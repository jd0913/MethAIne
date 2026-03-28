# -*- coding: utf-8 -*-
import streamlit as st
import os, math, json, warnings, uuid, logging, time, io, tempfile
import numpy as np
import pandas as pd
from glob import glob
from scipy.interpolate import PchipInterpolator
from scipy.stats import gaussian_kde

from joblib import Parallel, delayed
import dask
from dask.distributed import Client, LocalCluster
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, PageBreak, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY

from sklearn.linear_model import RidgeCV, LinearRegression
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel as C, WhiteKernel, RationalQuadratic
from sklearn.model_selection import TimeSeriesSplit
from statsmodels.robust.scale import mad
from statsmodels.tsa.stattools import acf
from statsmodels.tsa.arima.model import ARIMA
from arch import arch_model
from prophet import Prophet
import emcee

# --- SECTION 1: SETUP & LOGGING ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%H:%M:%S')
warnings.filterwarnings("ignore")

st.set_page_config(page_title="MethAIne 3.1 Dashboard", layout="wide")

# --- SECTION 2: STREAMLIT SIDEBAR (DASHBOARD) ---
st.sidebar.title("MethAIne 3.1 Dashboard")
st.sidebar.markdown("---")

st.sidebar.subheader("1. Run control & naming")
run_name = st.sidebar.text_input("Enter run name", value="default")

st.sidebar.subheader("2. Performance & accuracy settings")
fast_mode = st.sidebar.checkbox("Fast mode", value=True, help="Sacrifice accuracy for speed (1-2 mins).")
parallel_backend = st.sidebar.selectbox("Parallel processing library", ["joblib", "dask"])

st.sidebar.subheader("3. Data input")
upload_your_own_data = st.sidebar.checkbox("Upload your own CSV files", value=True)

st.sidebar.subheader("4. Advanced model parameters")
n_boot_full = st.sidebar.slider("Number of bootstraps", min_value=500, max_value=50000, value=10000, step=100)
poly_future_volatility_threshold = st.sidebar.slider("Future volatility threshold", min_value=1.0, max_value=20.0, value=7.0, step=0.5)

# --- SECTION 3: CONFIGURATION ---
_HIST_END_YEAR = 2025
_FUT_START_YEAR = _HIST_END_YEAR + 1

CONFIG = {
    "FAST_MODE": fast_mode,
    "PARALLEL_BACKEND": parallel_backend,
    "N_BOOT_FAST": 100,
    "N_BOOT_FULL": n_boot_full,
    "RNG_SEED": 42,
    "OBS_ERR_STD": 0.15,
    "FUTURE_NOISE_DAMPING": 0.6,
    "BLOCK_SIZE_DEFAULT": 5,
    "MODEL_DECAY_FACTOR": {"poly2": 1000, "pchip": 1000, "ridge": 1000, "prophet": 1000, "gp": 1000, "default": 1000},
    "POLY_FUTURE_VOLATILITY_THRESHOLD": poly_future_volatility_threshold,
    "HIST_FULL_START": 1940, "HIST_FULL_END": _HIST_END_YEAR,
    "CALIB_START": 1940, "CALIB_END": 1979,
    "MODERN_START": 1980, "MODERN_END": _HIST_END_YEAR,
    "FUT_START": _FUT_START_YEAR, "FUT_END": 2100,
    "FROZEN_VALIDATION_YEAR": 2014,
    "MIN_YEARS": 10,
}
np.random.seed(CONFIG['RNG_SEED'])

# --- SECTION 4: MethAIne LIBRARY FUNCTIONS ---

def hex_to_rgba(hex_color, opacity):
    hex_color = hex_color.lstrip('#')
    rgb = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    return f'rgba({rgb[0]}, {rgb[1]}, {rgb[2]}, {opacity})'

def get_run_config(config_dict):
    if config_dict["FAST_MODE"]:
        return {"N_BOOT": config_dict["N_BOOT_FAST"], "USE_EMCEE": False, "USE_GP": False, "USE_GARCH": False, "USE_PROPHET": False}
    else:
        return {"N_BOOT": config_dict["N_BOOT_FULL"], "USE_EMCEE": True, "USE_GP": True, "USE_GARCH": True, "USE_PROPHET": True}

def get_embedded_sample_data():
    years = np.arange(CONFIG['HIST_FULL_START'], CONFIG['HIST_FULL_END'] + 1)
    djf_temps = 0.5 + 0.03 * (years - 1940) + 0.5 * np.sin((years-1940)/15) + np.random.normal(0, 0.8, len(years))
    jja_temps = 18 + 0.04 * (years - 1940) + 0.0005 * (years - 1980)**2 + 0.4 * np.sin((years-1940)/20) + np.random.normal(0, 0.6, len(years))
    df = pd.DataFrame({"YEAR": years, "D-J-F": djf_temps, "J-J-A": jja_temps})
    string_io = io.StringIO()
    df.to_csv(string_io, index=False)
    string_io.seek(0)
    return string_io, "Demo City"

def preprocess_city(path_or_buffer, city_name=None, seasons=["D-J-F","J-J-A"]):
    if city_name is None:
        if hasattr(path_or_buffer, 'name'):
            city_name = path_or_buffer.name.replace(".csv","")
        else:
            city_name = "Uploaded_City"
    df = pd.read_csv(path_or_buffer)
    df.rename(columns={"DJF":"D-J-F", "JJA":"J-J-A"}, inplace=True)
    available_seasons = [s for s in seasons if s in df.columns]
    df = df[["YEAR"] + available_seasons].copy().replace([999.9,-999.9], np.nan)
    tt = df.melt(id_vars=["YEAR"], value_vars=available_seasons, var_name="SEASON", value_name="TEMP")
    tt["CITY"] = str(city_name)
    tt = tt.dropna(subset=["TEMP"])
    return tt

def generate_data_quality_report_st(all_city_data):
    report_output = []
    report_output.append("### --- DATA QUALITY REPORT ---")
    for city_df in all_city_data:
        city_name = str(city_df['CITY'].iloc[0])
        report_output.append(f"**Analysis for: {city_name}**")
        min_year = city_df['YEAR'].min()
        max_year = city_df['YEAR'].max()
        report_output.append(f"- Data spans from {min_year} to {max_year}.")
        for season in city_df['SEASON'].unique():
            season_df = city_df[city_df['SEASON'] == season]
            num_years = len(season_df)
            if num_years < CONFIG['MIN_YEARS']:
                report_output.append(f"  - ❌ {season}: Insufficient data ({num_years} years).")
            else:
                report_output.append(f"  - ✅ {season}: Data covers {num_years} years.")
    return "\n".join(report_output)

def choose_block_size(resid):
    if len(resid) < 10: return CONFIG['BLOCK_SIZE_DEFAULT']
    ac = acf(resid, nlags=min(50, len(resid)-1), fft=True, missing='conservative')
    lags = np.where(ac < math.exp(-1))[0]
    if len(lags) >= 1: return max(1, min(lags[0], 20))
    return CONFIG['BLOCK_SIZE_DEFAULT']

def block_bootstrap(x, y, block_size):
    n = len(x)
    nblocks = int(np.ceil(n / block_size))
    starts = np.random.randint(0, max(1, n - block_size + 1), size=nblocks)
    xb, yb = [], []
    for s in starts:
        end = min(s + block_size, n)
        xb.extend(x[s:end]); yb.extend(y[s:end])
    return np.array(xb[:n]), np.array(yb[:n])

def kde_fuse_slopes(slope_pool):
    if slope_pool.size < 5 or np.isnan(slope_pool).all(): return np.nanmean(slope_pool), (np.nan, np.nan)
    try:
        clean_pool = slope_pool[~np.isnan(slope_pool)]
        kde = gaussian_kde(clean_pool)
        x_range = np.linspace(np.percentile(clean_pool, 0.1), np.percentile(clean_pool, 99.9), 500)
        fused_mean = x_range[np.argmax(kde.pdf(x_range))]
        fused_ci = (np.nanpercentile(clean_pool, 2.5), np.nanpercentile(clean_pool, 97.5))
        return fused_mean, fused_ci
    except: return np.nanmean(slope_pool), (np.nan, np.nan)

def calculate_time_dependent_weights(model_weights, future_years):
    n_future = len(future_years)
    weights_td = {}
    for model_name, initial_weight in model_weights.items():
        decay = CONFIG["MODEL_DECAY_FACTOR"].get(model_name, CONFIG["MODEL_DECAY_FACTOR"]["default"])
        weights_td[model_name] = initial_weight * np.exp(-np.linspace(0, n_future / decay, n_future))
    return weights_td

def get_regional_adjustment_factor(city_name):
    name_str = str(city_name)
    return (len(name_str) - 6) * 0.01

def model_lin_modern(Xtr, ytr, Xte):
    m_mask = Xtr >= CONFIG['MODERN_START']
    if m_mask.sum() < 5: m_mask = np.ones_like(Xtr, dtype=bool)
    m = LinearRegression().fit(Xtr[m_mask].reshape(-1, 1), ytr[m_mask])
    return m.predict(Xte.reshape(-1, 1))

def model_poly2(Xtr, ytr, Xte):
    try: return np.poly1d(np.polyfit(Xtr, ytr, 2))(Xte)
    except: return np.poly1d(np.polyfit(Xtr, ytr, 1))(Xte)

def model_pchip(Xtr, ytr, Xte):
    try:
        o = np.argsort(Xtr)
        return PchipInterpolator(Xtr[o], ytr[o], extrapolate=True)(Xte)
    except: return model_poly2(Xtr, ytr, Xte)

def model_ridge(Xtr, ytr, Xte):
    try:
        xc = Xtr.mean()
        V = np.vstack([(Xtr - xc)**d for d in range(1, 4)]).T
        m = RidgeCV(alphas=[0.01, 0.1, 1.0]).fit(V, ytr)
        Vf = np.vstack([(Xte - xc)**d for d in range(1, 4)]).T
        return m.predict(Vf)
    except: return model_poly2(Xtr, ytr, Xte)

def make_stable(unstable_func):
    def stable_wrapper(Xtr, ytr, Xte):
        try:
            upred = unstable_func(Xtr, ytr, Xte)
            spred = model_lin_modern(Xtr, ytr, Xte)
            tw = np.linspace(1.0, 0.0, len(Xte))
            return (upred * tw) + (spred * (1 - tw))
        except: return model_lin_modern(Xtr, ytr, Xte)
    return stable_wrapper

def _model_gp_base(Xtr, ytr, Xte):
    try:
        k = C(1.0) * RationalQuadratic(length_scale=20.0) + WhiteKernel(noise_level=CONFIG['OBS_ERR_STD']**2)
        gp = GaussianProcessRegressor(kernel=k, normalize_y=True).fit(Xtr.reshape(-1, 1), ytr)
        return gp.predict(Xte.reshape(-1, 1))
    except: return model_poly2(Xtr, ytr, Xte)

def _model_prophet_base(Xtr, ytr, Xte):
    try:
        df = pd.DataFrame({'ds': pd.to_datetime(Xtr, format='%Y'), 'y': ytr})
        m = Prophet(yearly_seasonality=False, weekly_seasonality=False, daily_seasonality=False).fit(df)
        f = m.predict(pd.DataFrame({'ds': pd.to_datetime(Xte, format='%Y')}))
        return f['yhat'].values
    except: return model_poly2(Xtr, ytr, Xte)

def decadal_hindcast_errors(years, temps, model_funcs):
    errors = {name: [] for name in model_funcs.keys()}
    for d in range(CONFIG['CALIB_START'], CONFIG['FROZEN_VALIDATION_YEAR'], 10):
        tr_m = (years < d); te_m = (years >= d) & (years < d + 10)
        if te_m.sum() < 3 or tr_m.sum() < 5: continue
        Xtr, ytr, Xte, yte = years[tr_m], temps[tr_m], years[te_m], temps[te_m]
        for n, f in model_funcs.items():
            try: errors[n].append(np.sqrt(np.mean((f(Xtr, ytr, Xte) - yte)**2)))
            except: pass
    return {k: np.median(v) if v else np.nan for k, v in errors.items()}

def fit_ar_garch_and_simulate(resid, n_sim, use_garch=True):
    try:
        ar = ARIMA(resid, order=(1,0,0)).fit(enforce_stationarity=False)
        if use_garch and len(resid) > 30:
            g = arch_model(ar.resid, vol='Garch', p=1, q=1).fit(disp='off')
            sig = np.sqrt(g.forecast(horizon=n_sim).variance.values[-1])
            return ar.simulate(n_sim) + np.random.normal(0, sig, n_sim), "GARCH"
        return ar.simulate(n_sim), "ARIMA"
    except: return np.random.normal(0, np.std(resid), n_sim), "NOISE"

def fit_emcee_linear(years, temps):
    x = years - years.mean()
    def log_p(theta, x, y):
        inter, slope, ls = theta
        sig = np.exp(ls)
        if sig < 1e-5 or sig > 10: return -np.inf
        return -0.5 * np.sum(((y - (inter + slope * x)) / sig)**2 + 2 * ls)
    p0 = np.random.randn(32, 3) * 1e-3
    sampler = emcee.EnsembleSampler(32, 3, log_p, args=(x, temps))
    try:
        sampler.run_mcmc(p0, 500, progress=False)
        return sampler.get_chain(flat=True)[:, 1] * 10.0
    except: return np.array([])

def bootstrap_iteration(idx, years, t_obs, run_config, model_funcs, td_weights, q, resid, reg_adj):
    xb, yb = block_bootstrap(years, t_obs, choose_block_size(resid))
    preds = {n: f(xb, yb, run_config['future_years']) for n, f in model_funcs.items()}
    stacked = sum(td_weights[m] * np.array(preds[m]) for m in td_weights)
    stacked += (q(years.max()) - stacked[0])
    sim_n, _ = fit_ar_garch_and_simulate(resid, len(run_config['future_years']), run_config['USE_GARCH'])
    f_temps = stacked + CONFIG['FUTURE_NOISE_DAMPING'] * sim_n + reg_adj
    slope = np.polyfit(run_config['future_years'] - run_config['future_years'][0], f_temps, 1)[0] * 10.0
    return f_temps, slope

def run_projection_for_city(city_df, run_config):
    city_name = str(city_df["CITY"].iloc[0])
    h_end = city_df["YEAR"].max()
    f_years = np.arange(h_end + 1, CONFIG['FUT_END'] + 1)
    run_config['future_years'] = f_years
    reg_adj = get_regional_adjustment_factor(city_name)
    
    city_results = {}
    for season, g in city_df.groupby("SEASON"):
        yrs, tps = g.sort_values('YEAR')["YEAR"].values, g.sort_values('YEAR')["TEMP"].values
        if len(yrs) < CONFIG['MIN_YEARS']: continue
        
        t_obs = tps + np.random.normal(0, CONFIG['OBS_ERR_STD'], len(tps))
        q = np.poly1d(np.polyfit(yrs, t_obs, 2))
        resid = t_obs - q(yrs)
        
        m_funcs = {"poly2": make_stable(model_poly2), "pchip": make_stable(model_pchip), "ridge": make_stable(model_ridge)}
        if run_config['USE_GP']: m_funcs["gp"] = make_stable(_model_gp_base)
        if run_config['USE_PROPHET']: m_funcs["prophet"] = make_stable(_model_prophet_base)
        
        errs = decadal_hindcast_errors(yrs, t_obs, m_funcs)
        e_arr = np.array([errs.get(m, 10.0) for m in m_funcs])
        w = 1.0 / (e_arr + 1e-9); w /= w.sum()
        td_weights = calculate_time_dependent_weights({m: w[i] for i, m in enumerate(m_funcs)}, f_years)
        
        if CONFIG['PARALLEL_BACKEND'] == 'dask':
            res = dask.compute(*[dask.delayed(bootstrap_iteration)(i, yrs, t_obs, run_config, m_funcs, td_weights, q, resid, reg_adj) for i in range(run_config['N_BOOT'])])
        else:
            res = Parallel(n_jobs=-1)(delayed(bootstrap_iteration)(i, yrs, t_obs, run_config, m_funcs, td_weights, q, resid, reg_adj) for i in range(run_config['N_BOOT']))
            
        b_preds, b_slopes = np.array([r[0] for r in res]), np.array([r[1] for r in res])
        f_pool = b_slopes
        if run_config['USE_EMCEE']:
            f_pool = np.concatenate([b_slopes, fit_emcee_linear(yrs, t_obs)])
        
        f_mean, f_ci = kde_fuse_slopes(f_pool)
        ens_mean = np.nanmean(b_preds, axis=0)
        
        m_mask = (yrs >= CONFIG['MODERN_START'])
        modern_slope = np.polyfit(yrs[m_mask], tps[m_mask], 1)[0] * 10.0 if m_mask.sum() > 2 else np.nan
        
        city_results[season] = {
            "summary": {
                "CITY": city_name, "SEASON": season, "FUSED_SLOPE_C_per_dec": f_mean,
                "MODERN_SLOPE_C_per_dec": modern_slope, "FUSED_SLOPE_CI_LO": f_ci[0], "FUSED_SLOPE_CI_HI": f_ci[1],
                "INST_SLOPE_2100": np.polyfit(f_years[-5:] - f_years[-5:].mean(), ens_mean[-5:], 1)[0] * 10.0,
                "HIST_1940_1979_SLOPE_C_per_dec": np.polyfit(yrs[(yrs>=1940)&(yrs<=1979)], tps[(yrs>=1940)&(yrs<=1979)], 1)[0]*10 if len(yrs[(yrs>=1940)&(yrs<=1979)])>2 else np.nan,
                "ACCELERATION_RATE_C_per_dec": f_mean - modern_slope
            },
            "plot_data": {
                "years_full": yrs, "temps_full": tps, "future_years": f_years, "ens_mean": ens_mean,
                "ci_lower": np.nanpercentile(b_preds, 2.5, axis=0), "ci_upper": np.nanpercentile(b_preds, 97.5, axis=0)
            }
        }
    return city_results

# --- SECTION 5: VISUALIZATION ---

def generate_interactive_plot(all_results, eid):
    fig = go.Figure()
    season_colors = {"D-J-F": ["#5DADE2", "#21618C"], "J-J-A": ["#E67E22", "#873600"]}
    unique_cities = sorted(list(set(r['summary']['CITY'] for r in all_results)))
    for d in all_results:
        p, s = d["plot_data"], d["summary"]
        c_idx = unique_cities.index(s['CITY']) % 2
        base_clr = season_colors[s['SEASON']][c_idx]
        fill_clr = hex_to_rgba(base_clr, 0.15)
        
        fig.add_trace(go.Scatter(x=p['years_full'], y=p['temps_full'], mode='markers', marker=dict(color=base_clr, opacity=0.4), name=f"{s['CITY']} {s['SEASON']} Obs"))
        fig.add_trace(go.Scatter(x=p['future_years'], y=p['ens_mean'], mode='lines', line=dict(color=base_clr, width=3, dash='dash'), name=f"{s['CITY']} {s['SEASON']} Proj"))
        fig.add_trace(go.Scatter(x=np.concatenate([p['future_years'], p['future_years'][::-1]]), y=np.concatenate([p['ci_upper'], p['ci_lower'][::-1]]), fill='toself', fillcolor=fill_clr, line=dict(color='rgba(255,255,255,0)'), showlegend=False))
    fig.update_layout(title=f"MethAIne 3.1 Projections - ID: {eid}", template="plotly_white", hovermode="x unified")
    return fig

def generate_static_plot(all_results, eid):
    fig, ax = plt.subplots(figsize=(12, 6))
    sc = {"D-J-F": "#5DADE2", "J-J-A": "#E67E22"}
    for d in all_results:
        p, s = d["plot_data"], d["summary"]
        clr = sc[s['SEASON']]
        ax.scatter(p['years_full'], p['temps_full'], color=clr, alpha=0.3, s=10)
        ax.plot(p['future_years'], p['ens_mean'], color=clr, linestyle='--', linewidth=2)
        ax.fill_between(p['future_years'], p['ci_lower'], p['ci_upper'], color=clr, alpha=0.1)
    ax.set_title(f"MethAIne 3.1 Comparison - ID: {eid}")
    ax.grid(True, alpha=0.3)
    return fig

def generate_pdf_report(all_results, eid):
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    doc = SimpleDocTemplate(tmp.name)
    styles = getSampleStyleSheet()
    story = [Paragraph(f"MethAIne 3.1 Climate Risk Report", styles['h1']), Paragraph(f"Execution ID: {eid}", styles['h3']), Spacer(1, 12)]
    for r in all_results:
        s = r['summary']
        txt = f"<b>City: {s['CITY']} ({s['SEASON']})</b><br/>Future Trend: {s['FUSED_SLOPE_C_per_dec']:+.3f} °C/dec<br/>Modern Era: {s['MODERN_SLOPE_C_per_dec']:+.3f} °C/dec"
        story.append(Paragraph(txt, styles['Normal']))
        story.append(Spacer(1, 12))
    doc.build(story)
    return tmp.name

# --- SECTION 6: MAIN APP EXECUTION ---

def main():
    st.title("MethAIne 3.1 Dashboard")
    
    if upload_your_own_data:
        uploaded_files = st.file_uploader("Upload NASA/GISTEMP CSV Files", type="csv", accept_multiple_files=True)
    else:
        uploaded_files = None

    if st.button("🚀 Run MethAIne Analysis"):
        with st.spinner("Processing climate models..."):
            eid = str(uuid.uuid4())
            
            if uploaded_files:
                all_city_data = [preprocess_city(f) for f in uploaded_files]
            else:
                st.info("Running demonstration with demo data.")
                buf, name = get_embedded_sample_data()
                all_city_data = [preprocess_city(buf, city_name=name)]
            
            st.markdown(generate_data_quality_report_st(all_city_data))
            
            run_config = get_run_config(CONFIG)
            all_results = []
            for city_df in all_city_data:
                res = run_projection_for_city(city_df, run_config)
                for s, d in res.items(): all_results.append(d)
            
            if not all_results:
                st.error("No valid data to process.")
                return

            st.plotly_chart(generate_interactive_plot(all_results, eid), use_container_width=True)
            
            col1, col2 = st.columns(2)
            with col1:
                st.pyplot(generate_static_plot(all_results, eid))
            with col2:
                summary_df = pd.DataFrame([r['summary'] for r in all_results])
                st.dataframe(summary_df)

            csv = summary_df.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download Results CSV", csv, f"{run_name}_projections.csv", "text/csv")
            
            pdf_path = generate_pdf_report(all_results, eid)
            with open(pdf_path, "rb") as f:
                st.download_button("📥 Download PDF Report", f, f"{run_name}_report.pdf", "application/pdf")

if __name__ == "__main__":
    main()
