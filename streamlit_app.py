# -*- coding: utf-8 -*-
"""
MethAIne 3.1 (Official Streamlit Port - Stability Optimized)
COPYRIGHT OWNERS: JEFF MINSONA LEBA & YUG PATEL
"""

import streamlit as st
import os, math, json, warnings, uuid, logging, time, io, tempfile
import numpy as np
import pandas as pd
from glob import glob
from scipy.interpolate import PchipInterpolator
from scipy.stats import gaussian_kde

# Stability Fix: Set Matplotlib backend to Agg before any other imports
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines

from joblib import Parallel, delayed
import dask
from dask.distributed import Client, LocalCluster
import plotly.graph_objects as go
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
# Silence Prophet/Stan logging to prevent Streamlit buffer overflow
logging.getLogger('prophet').setLevel(logging.ERROR)
logging.getLogger('cmdstanpy').setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

st.set_page_config(page_title="MethAIne 3.1 Dashboard", layout="wide")

# --- SECTION 2: STREAMLIT SIDEBAR ---
st.sidebar.title("MethAIne 3.1 Dashboard")
st.sidebar.markdown("---")

st.sidebar.subheader("1. Run control & naming")
run_name = st.sidebar.text_input("Run Name", value="default")

st.sidebar.subheader("2. Performance & accuracy settings")
fast_mode = st.sidebar.checkbox("Fast mode", value=True, help="Sacrifice accuracy for speed (1-2 mins).")
parallel_backend = st.sidebar.selectbox("Parallel processing library", ["joblib", "dask"])

st.sidebar.subheader("3. Data input")
upload_your_own_data = st.sidebar.checkbox("Upload your own CSV files", value=True)

st.sidebar.subheader("4. Advanced model parameters")
n_boot_full = st.sidebar.slider("Number of bootstraps", 500, 50000, 10000, step=100)
poly_future_volatility_threshold = st.sidebar.slider("Volatility Threshold", 1.0, 20.0, 7.0, step=0.5)

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

# --- SECTION 4: HELPER FUNCTIONS ---

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

def generate_data_quality_report(all_city_data):
    report_lines = ["\n--- DATA QUALITY REPORT ---"]
    for city_df in all_city_data:
        city_name = city_df['CITY'].iloc[0]
        report_lines.append(f"\nAnalysis for: {city_name}")
        min_year_data = city_df['YEAR'].min()
        max_year_data = city_df['YEAR'].max()
        total_years_span = max_year_data - min_year_data + 1
        report_lines.append(f"  Data spans from {min_year_data} to {max_year_data}.")
        for season in city_df['SEASON'].unique():
            season_df = city_df[city_df['SEASON'] == season]
            num_years = len(season_df)
            missing_years = total_years_span - num_years
            if num_years < CONFIG['MIN_YEARS']:
                report_lines.append(f"  - ❌ {season}: Insufficient data ({num_years} years).")
                continue
            else:
                report_lines.append(f"  - ✅ {season}: Data covers {num_years} years.")
            if missing_years > 0:
                report_lines.append(f"    - ⚠️ Warning: {missing_years} missing year(s) detected.")
            temps = season_df['TEMP'].values
            median = np.median(temps)
            madv = mad(temps)
            if madv > 0:
                z_scores = 0.6745 * (temps - median) / madv
                outliers = np.where(np.abs(z_scores) > 3.5)[0]
                if len(outliers) > 0:
                    outlier_years = season_df['YEAR'].iloc[outliers].values
                    report_lines.append(f"    - ⚠️ Warning: {len(outliers)} potential outlier(s) detected in year(s): {outlier_years}")
    report_lines.append("--- END OF REPORT ---\n")
    return "\n".join(report_lines)

def choose_block_size(resid):
    if len(resid) < 10: return CONFIG['BLOCK_SIZE_DEFAULT']
    ac = acf(resid, nlags=min(50, len(resid)-1), fft=True, missing='conservative')
    lags = np.where(ac < math.exp(-1))[0]
    if len(lags) >= 1: return max(1, min(lags[0], 20))
    l2 = np.where(ac < 0.5)[0]
    if len(l2) >= 1: return max(1, min(int(l2[0]), 20))
    return CONFIG['BLOCK_SIZE_DEFAULT']

def block_bootstrap(x, y, block_size):
    n = len(x)
    if block_size <= 1:
        idx = np.random.choice(n, n, replace=True)
        return x[idx], y[idx]
    nblocks = int(np.ceil(n / block_size))
    starts = np.random.randint(0, max(1, n - block_size + 1), size=nblocks)
    xb, yb = [], []
    for s in starts:
        end = min(s + block_size, n)
        xb.extend(x[s:end]); yb.extend(y[s:end])
    xb, yb = np.array(xb), np.array(yb)
    if len(xb) < n:
        extra = np.random.choice(len(xb), n - len(xb))
        xb = np.concatenate([xb, xb[extra]]); yb = np.concatenate([yb, yb[extra]])
    return xb[:n], yb[:n]

def kde_fuse_slopes(slope_pool):
    if slope_pool.size < 50 or np.isnan(slope_pool).all(): return np.nanmean(slope_pool), (np.nan, np.nan)
    try:
        clean_pool = slope_pool[~np.isnan(slope_pool)]
        if len(clean_pool) < 5: return np.nanmean(slope_pool), (np.nan, np.nan)
        kde = gaussian_kde(clean_pool)
        x_range = np.linspace(np.percentile(clean_pool, 0.1), np.percentile(clean_pool, 99.9), 500)
        fused_mean = x_range[np.argmax(kde.pdf(x_range))]
        fused_ci = (np.nanpercentile(clean_pool, 2.5), np.nanpercentile(clean_pool, 97.5))
        return fused_mean, fused_ci
    except Exception: return np.nanmean(slope_pool), (np.nan, np.nan)

def calculate_time_dependent_weights(model_weights, future_years):
    n_future = len(future_years)
    weights_time_dependent = {}
    for model_name, initial_weight in model_weights.items():
        decay_factor = CONFIG["MODEL_DECAY_FACTOR"].get(model_name, CONFIG["MODEL_DECAY_FACTOR"]["default"])
        weights_time_dependent[model_name] = initial_weight * np.exp(-np.linspace(0, n_future / decay_factor, n_future))
    return weights_time_dependent

def get_regional_adjustment_factor(city_name):
    adjustment = (len(str(city_name)) - 6) * 0.01
    return adjustment

def model_lin_modern(Xtr, ytr, Xte):
    try:
        modern_mask = Xtr >= CONFIG['MODERN_START']
        if modern_mask.sum() < 5: modern_mask = np.ones_like(Xtr, dtype=bool)
        X_modern, y_modern = Xtr[modern_mask].reshape(-1, 1), ytr[modern_mask]
        model = LinearRegression().fit(X_modern, y_modern)
        return model.predict(Xte.reshape(-1, 1))
    except: return np.poly1d(np.polyfit(Xtr, ytr, 1))(Xte)

def model_poly2(Xtr, ytr, Xte):
    try: return np.poly1d(np.polyfit(Xtr, ytr, 2))(Xte)
    except: return np.poly1d(np.polyfit(Xtr, ytr, 1))(Xte)

def model_pchip(Xtr, ytr, Xte):
    try:
        order = np.argsort(Xtr)
        return PchipInterpolator(Xtr[order], ytr[order], extrapolate=True)(Xte)
    except: return model_poly2(Xtr, ytr, Xte)

def model_ridge(Xtr, ytr, Xte):
    try:
        X_center = Xtr.mean()
        V = np.vstack([(Xtr - X_center)**d for d in range(1, 4)]).T
        model = RidgeCV(alphas=[0.01, 0.1, 1.0], cv=TimeSeriesSplit(n_splits=3)).fit(V, ytr)
        Vf = np.vstack([(Xte - X_center)**d for d in range(1, 4)]).T
        return model.predict(Vf)
    except: return model_poly2(Xtr, ytr, Xte)

def make_stable(unstable_model_func):
    def stable_model_wrapper(Xtr, ytr, Xte):
        try:
            unstable_pred = unstable_model_func(Xtr, ytr, Xte)
            stable_anchor_pred = model_lin_modern(Xtr, ytr, Xte)
            n_future = len(Xte)
            transition_weight = np.linspace(1.0, 0.0, n_future)
            blended_pred = (unstable_pred * transition_weight) + (stable_anchor_pred * (1 - transition_weight))
            return blended_pred
        except:
            return model_lin_modern(Xtr, ytr, Xte)
    return stable_model_wrapper

def _model_gp_base(Xtr, ytr, Xte):
    try:
        kernel = C(1.0, (1e-3, 1e3)) * RationalQuadratic(length_scale=20.0, alpha=1.0) + WhiteKernel(noise_level=CONFIG['OBS_ERR_STD']**2)
        gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True, random_state=CONFIG['RNG_SEED'])
        gp.fit(Xtr.reshape(-1, 1), ytr)
        return gp.predict(Xte.reshape(-1, 1))
    except Exception:
        return model_poly2(Xtr, ytr, Xte)

def _model_prophet_base(Xtr, ytr, Xte):
    try:
        df_train = pd.DataFrame({'ds': pd.to_datetime(Xtr, format='%Y'), 'y': ytr})
        model = Prophet(yearly_seasonality=False, weekly_seasonality=False, daily_seasonality=False, uncertainty_samples=False)
        model.fit(df_train)
        future_df = pd.DataFrame({'ds': pd.to_datetime(Xte, format='%Y')})
        forecast = model.predict(future_df)
        return forecast['yhat'].values
    except Exception:
        return model_poly2(Xtr, ytr, Xte)

def decadal_hindcast_errors(years, temps, model_funcs):
    errors = {name: [] for name in model_funcs.keys()}
    for d in range(CONFIG['CALIB_START'], CONFIG['FROZEN_VALIDATION_YEAR'], 10):
        train_mask = (years < d); test_mask = (years >= d) & (years < d + 10)
        if test_mask.sum() < 3 or train_mask.sum() < 5: continue
        Xtr, ytr = years[train_mask], temps[train_mask]
        Xte, yte = years[test_mask], temps[test_mask]
        for name, func in model_funcs.items():
            try: errors[name].append(np.sqrt(np.mean((func(Xtr, ytr, Xte) - yte)**2)))
            except: errors[name].append(np.nan)
    med = {}
    for k, v in errors.items():
        clean_v = [x for x in v if not np.isnan(x)]
        med[k] = np.median(clean_v) if clean_v else np.nan
    return med

def fit_ar_garch_and_simulate(resid, n_sim, use_garch=True):
    if len(resid) < 30 or np.allclose(np.std(resid), 0):
        return np.random.normal(0, np.std(resid) if len(resid) > 1 else 0.1, size=n_sim), "SIMPLE_NOISE", None
    try:
        ar_model = ARIMA(resid, order=(1,0,0)).fit(enforce_stationarity=False)
        ar_resid = ar_model.resid
        if use_garch:
            garch_mod = arch_model(ar_resid, vol='Garch', p=1, q=1, dist='normal')
            garch_res = garch_mod.fit(disp='off')
            vol_forecast = garch_res.forecast(horizon=n_sim, reindex=False)
            sigma = np.sqrt(vol_forecast.variance.values[-1])
            return ar_model.simulate(n_sim) + np.random.normal(0, sigma, n_sim), "GARCH", None
        else:
            return np.random.choice(resid, size=n_sim, replace=True), "RESIDUAL_RESAMPLE", None
    except Exception as e:
        return np.random.normal(0, np.std(resid), size=n_sim), "SIMPLE_NOISE", None

def fit_emcee_linear(years, temps):
    x, y = np.array(years, dtype=float), np.array(temps, dtype=float)
    x0 = x.mean(); xc = x - x0
    def log_prob(theta, x, y):
        intercept, slope, log_sigma = theta
        model = intercept + slope * x
        sigma = np.exp(log_sigma)
        return -0.5 * np.sum(((y - model) / sigma)**2 + 2 * np.log(sigma) + np.log(2 * np.pi))
    try:
        A = np.vstack([xc, np.ones_like(xc)]).T
        ols = np.linalg.lstsq(A, y, rcond=None)[0]
        slope_init, intercept_init = ols[0], ols[1]
        log_sigma_init = np.log(np.std(y - (intercept_init + slope_init * xc)) + 1e-3)
    except: slope_init, intercept_init, log_sigma_init = 0.0, y.mean(), np.log(np.std(y) + 1e-3)
    ndim, p0 = 3, np.zeros((32, 3))
    p0[:,0] = intercept_init + 1e-3 * np.random.randn(32)
    p0[:,1] = slope_init + 1e-4 * np.random.randn(32)
    p0[:,2] = log_sigma_init + 1e-3 * np.random.randn(32)
    sampler = emcee.EnsembleSampler(32, ndim, log_prob, args=(xc, y))
    try:
        pos, _, _ = sampler.run_mcmc(p0, 250, progress=False)
        sampler.reset()
        sampler.run_mcmc(pos, 400, progress=False)
        chain = sampler.get_chain(flat=True)
        return chain[:,1] * 10.0 if chain.size > 0 else np.full(CONFIG['N_BOOT_FULL'] // 10, np.nan)
    except: return np.full(CONFIG['N_BOOT_FULL'] // 10, np.nan)

def bootstrap_iteration(iteration_index, years, temps_obs_err, run_config, model_funcs, weights_time_dependent, q, resid, regional_adjustment):
    xb, yb = block_bootstrap(years, temps_obs_err, choose_block_size(resid))
    preds = {name: func(xb, yb, run_config['future_years']) for name, func in model_funcs.items()}
    stacked = sum(weights_time_dependent[m] * np.array(preds[m]) for m in weights_time_dependent)
    offset = q(years.max()) - stacked[0]
    stacked += offset
    temps_future = stacked
    sim_noise, _, _ = fit_ar_garch_and_simulate(resid, len(run_config['future_years']), use_garch=run_config['USE_GARCH'])
    temps_future += CONFIG['FUTURE_NOISE_DAMPING'] * sim_noise
    temps_future += regional_adjustment
    slope = np.polyfit(run_config['future_years'] - run_config['future_years'][0], temps_future, 1)[0] * 10.0
    return temps_future, slope

def run_projection_for_city(city_df, run_config):
    city_name = city_df["CITY"].iloc[0]
    hist_end_year = city_df["YEAR"].max()
    future_years = np.arange(hist_end_year + 1, CONFIG['FUT_END'] + 1)
    run_config['future_years'] = future_years
    regional_adjustment = get_regional_adjustment_factor(city_name)
    city_results = {}
    for season, g_full in city_df.groupby("SEASON"):
        g_full = g_full.sort_values('YEAR')
        years_full, temps_full = g_full["YEAR"].values, g_full["TEMP"].values
        years, temps = years_full, temps_full
        if len(years) < CONFIG['MIN_YEARS']: continue
        temps_obs_err = temps + np.random.normal(0, CONFIG['OBS_ERR_STD'], size=len(temps))
        try: q = np.poly1d(np.polyfit(years, temps_obs_err, 2))
        except: q = np.poly1d(np.polyfit(years, temps_obs_err, 1))
        resid = temps_obs_err - q(years)
        model_funcs = {"poly2": make_stable(model_poly2), "pchip": make_stable(model_pchip), "ridge": make_stable(model_ridge)}
        if run_config['USE_GP']: model_funcs["gp"] = make_stable(_model_gp_base)
        if run_config['USE_PROPHET']: model_funcs["prophet"] = make_stable(_model_prophet_base)
        med_err = decadal_hindcast_errors(years, temps_obs_err, model_funcs)
        err_arr = np.array([med_err.get(m, np.nan) for m in model_funcs], dtype=float)
        if np.all(np.isnan(err_arr)):
            weights = {m: 1.0/len(model_funcs) for m in model_funcs}
        else:
            nanmask = np.isnan(err_arr)
            maxv = np.nanmax(err_arr) if not np.all(nanmask) else 1.0
            err_arr[nanmask] = maxv * 10.0
            inv = 1.0 / (err_arr + 1e-12)
            inv /= inv.sum()
            weights = {m: float(inv[i]) for i, m in enumerate(model_funcs)}
        for model_name in ['ridge']:
            if model_name in model_funcs and weights.get(model_name, 0) > 0:
                try:
                    pred_raw = model_funcs[model_name](years, temps_obs_err, future_years)
                    future_vol_range = np.abs(pred_raw.max() - pred_raw.min())
                    if future_vol_range > CONFIG['POLY_FUTURE_VOLATILITY_THRESHOLD']:
                        weights[model_name] *= 0.01
                except: pass
        sum_w = sum(weights.values()); weights = {k: v / sum_w for k, v in weights.items()} if sum_w > 0 else weights
        weights_time_dependent = calculate_time_dependent_weights(weights, future_years)
        
        # Stability Fix: Limit workers to prevent OOM crashes in Streamlit
        n_workers = min(os.cpu_count(), 4) if CONFIG['PARALLEL_BACKEND'] == 'joblib' else -1
        
        if CONFIG['PARALLEL_BACKEND'] == 'dask':
            lazy_results = [dask.delayed(bootstrap_iteration)(i, years, temps_obs_err, run_config, model_funcs, weights_time_dependent, q, resid, regional_adjustment) for i in range(run_config['N_BOOT'])]
            parallel_results = dask.compute(*lazy_results)
        else:
            parallel_results = Parallel(n_jobs=n_workers)(delayed(bootstrap_iteration)(i, years, temps_obs_err, run_config, model_funcs, weights_time_dependent, q, resid, regional_adjustment) for i in range(run_config['N_BOOT']))
        
        boot_preds = np.array([res[0] for res in parallel_results])
        boot_avg_slopes = np.array([res[1] for res in parallel_results])
        ens_mean = np.nanmean(boot_preds, axis=0)
        ci_lower = np.nanpercentile(boot_preds, 2.5, axis=0)
        ci_upper = np.nanpercentile(boot_preds, 97.5, axis=0)
        inst_slope_2100 = np.polyfit(future_years[-5:] - future_years[-5:].mean(), ens_mean[-5:], 1)[0] * 10.0
        fused_pool = boot_avg_slopes
        if run_config['USE_EMCEE']:
            emcee_slopes = fit_emcee_linear(years, temps_obs_err)
            fused_pool = np.concatenate([boot_avg_slopes, emcee_slopes])
        fused_mean, fused_ci = kde_fuse_slopes(fused_pool)
        calib_mask = (years_full >= CONFIG['CALIB_START']) & (years_full <= CONFIG['CALIB_END'])
        modern_mask = (years_full >= CONFIG['MODERN_START']) & (years_full <= hist_end_year)
        hist_slope_40_79 = np.polyfit(years_full[calib_mask], temps_full[calib_mask], 1)[0] * 10.0 if calib_mask.sum() >= 2 else np.nan
        modern_slope = np.polyfit(years_full[modern_mask], temps_full[modern_mask], 1)[0] * 10.0 if modern_mask.sum() >= 2 else np.nan
        acceleration_rate = fused_mean - modern_slope
        city_results[season] = {
            "summary": {
                "CITY": city_name, "SEASON": season, "FUSED_SLOPE_C_per_dec": fused_mean,
                "HIST_1940_1979_SLOPE_C_per_dec": hist_slope_40_79, f"MODERN_{CONFIG['MODERN_START']}_{hist_end_year}_SLOPE_C_per_dec": modern_slope,
                "INST_SLOPE_2100_C_per_dec": inst_slope_2100, "FUSED_SLOPE_CI_LO": fused_ci[0], "FUSED_SLOPE_CI_HI": fused_ci[1],
                "ACCELERATION_RATE_C_per_dec": acceleration_rate
            },
            "plot_data": {
                "years_full": years_full, "temps_full": temps_full, "future_years": future_years, "ens_mean": ens_mean,
                "ci_lower": ci_lower, "ci_upper": ci_upper
            }
        }
    return city_results

# --- SECTION 5: VISUALIZATION ---

def generate_interactive_plot(all_results, execution_id):
    fig = go.Figure()
    season_colors = {"D-J-F": ["rgba(93, 173, 226, 1)", "rgba(33, 97, 140, 1)"], "J-J-A": ["rgba(230, 126, 34, 1)", "rgba(135, 54, 0, 1)"]}
    unique_cities = sorted(list(set(r['summary']['CITY'] for r in all_results)))
    for data in all_results:
        p, s = data["plot_data"], data["summary"]
        city_name, season = s["CITY"], s["SEASON"]
        city_idx = unique_cities.index(city_name) % 2
        base_color = season_colors[season][city_idx]
        transparent_color = base_color.replace('1)', '0.15)')
        legend_group = f"{city_name}_{season}"
        fig.add_trace(go.Scatter(x=p['years_full'], y=p['temps_full'], mode='markers', marker=dict(color=base_color, size=5, opacity=0.4), name=f"{city_name} {season} Obs.", legendgroup=legend_group))
        hist_years = np.arange(p['years_full'].min(), p['years_full'].max() + 1)
        poly_fit = np.poly1d(np.polyfit(p["years_full"], p["temps_full"], 2))
        fig.add_trace(go.Scatter(x=hist_years, y=poly_fit(hist_years), mode='lines', line=dict(color=base_color, width=3, dash='solid'), name=f"{city_name} {season} Hist. Trend", legendgroup=legend_group))
        fig.add_trace(go.Scatter(x=p['future_years'], y=p['ens_mean'], mode='lines', line=dict(color=base_color, width=4, dash='dash'), name=f"{city_name} {season} Future Mean", legendgroup=legend_group))
        fig.add_trace(go.Scatter(x=np.concatenate([p['future_years'], p['future_years'][::-1]]), y=np.concatenate([p['ci_upper'], p['ci_lower'][::-1]]), fill='toself', fillcolor=transparent_color, line=dict(color='rgba(255,255,255,0)'), hoverinfo="none", showlegend=False, legendgroup=legend_group))
    fig.update_layout(title=f"MethAIne 3.1: Surface Temperature Projections — ID: {execution_id}", xaxis_title="Year", yaxis_title="Temperature (°C)", template="plotly_white", hovermode="x unified")
    return fig

def plot_city_on_axis(ax, city_results, title=None):
    season_colors = {"D-J-F": "#5DADE2", "J-J-A": "#E67E22"}
    hist_end_year = max(r['plot_data']['years_full'].max() for r in city_results)
    for data in city_results:
        p, s = data["plot_data"], data["summary"]
        color = season_colors[s['SEASON']]
        hist_years_for_trend = np.arange(p['years_full'].min(), p['years_full'].max() + 1)
        ax.scatter(p["years_full"], p["temps_full"], color=color, s=15, alpha=0.3)
        poly_hist_full = np.poly1d(np.polyfit(p["years_full"], p["temps_full"], 2))
        ax.plot(hist_years_for_trend, poly_hist_full(hist_years_for_trend), color=color, linewidth=1.5)
        ax.plot(p['future_years'], p["ens_mean"], color=color, linestyle='--', linewidth=2)
        ax.fill_between(p['future_years'], p["ci_lower"], p["ci_upper"], color=color, alpha=0.15)
    ax.set_xlabel("Year", fontsize=12)
    ax.grid(alpha=0.4)
    if title: ax.set_title(title, fontsize=14)
    scatter_handle = mlines.Line2D([], [], color='gray', marker='o', linestyle='None', markersize=6, label='Observed Data Points')
    solid_handle = mlines.Line2D([], [], color='black', linestyle='-', linewidth=2, label=f'Historical Trend Fit (1940–{hist_end_year})')
    dashed_handle = mlines.Line2D([], [], color='black', linestyle='--', linewidth=2, label=f'Future Ensemble Mean ({hist_end_year+1}–2100)')
    shade_blue = mpatches.Patch(color=season_colors["D-J-F"], alpha=0.4, label='Winter CI (D-J-F)')
    shade_orange = mpatches.Patch(color=season_colors["J-J-A"], alpha=0.4, label='Summer CI (J-J-A)')
    first_legend = ax.legend(handles=[scatter_handle, solid_handle, dashed_handle, shade_blue, shade_orange], loc='upper left', fontsize=9, title="Legend")
    ax.add_artist(first_legend)
    proj_handles = []
    for r in [d['summary'] for d in city_results]:
        color = season_colors[r['SEASON']]
        label = (f"{r['CITY']} {r['SEASON']} (Hist: {r['HIST_1940_1979_SLOPE_C_per_dec']:+.3f}, Avg: {r['FUSED_SLOPE_C_per_dec']:+.3f}, 2100: {r['INST_SLOPE_2100_C_per_dec']:+.3f})")
        proj_handles.append(mlines.Line2D([], [], color=color, linestyle='--', linewidth=2, label=label))
    ax.legend(handles=proj_handles, loc='lower right', fontsize=8, title="Seasonal Projections")

def generate_static_plot(all_results, execution_id):
    unique_cities = sorted(list(set(r['summary']['CITY'] for r in all_results)))
    n_cities = len(unique_cities)
    if n_cities == 2:
        fig, axes = plt.subplots(1, 2, figsize=(20, 8), sharey=True)
        fig.suptitle(f"MethAIne 3.1: Climate Projections Comparison — ID: {execution_id}", fontsize=16)
        for i, city_name in enumerate(unique_cities):
            ax = axes[i]
            city_results = [r for r in all_results if r['summary']['CITY'] == city_name]
            plot_city_on_axis(ax, city_results, city_name)
        axes[0].set_ylabel("Temperature (°C)", fontsize=12)
    else:
        fig, ax = plt.subplots(figsize=(16, 8))
        plot_city_on_axis(ax, all_results)
        ax.set_title(f"MethAIne 3.1: Multi-City Climate Projections — ID: {execution_id}", fontsize=14)
        ax.set_ylabel("Temperature (°C)", fontsize=12)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    return fig

def generate_pdf_report(all_results, execution_id, static_plot_path):
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    doc = SimpleDocTemplate(tmp.name)
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name='Justify', alignment=TA_JUSTIFY))
    story = []
    results_by_city = {}
    for r in all_results:
        city = r['summary']['CITY']
        if city not in results_by_city: results_by_city[city] = []
        results_by_city[city].append(r)
    for city_name, city_results in results_by_city.items():
        summary_data = [d['summary'] for d in city_results]
        story.append(Paragraph(f"MethAIne 3.1: Climate Risk Report for {city_name}", styles['h1']))
        story.append(Paragraph(f"Execution ID: {execution_id}", styles['h3']))
        story.append(Spacer(1, 0.25*inch))
        story.append(Paragraph("<b>Executive Summary</b>", styles['h2']))
        jja_summary = next((item for item in summary_data if item["SEASON"] == "J-J-A"), summary_data[0])
        modern_slope_key = [k for k in jja_summary.keys() if k.startswith('MODERN_')][0]
        modern_slope = jja_summary[modern_slope_key]
        future_slope = jja_summary['FUSED_SLOPE_C_per_dec']
        end_slope = jja_summary['INST_SLOPE_2100_C_per_dec']
        summary_text = f"This report presents a long-term climate projection for {city_name} through 2100. The analysis reveals a significant warming trend during the modern era. "
        if future_slope > 0.05: summary_text += f"The future projection indicates a continued warming trend, with an average rate of <b>{future_slope:+.3f} °C per decade</b>. "
        else: summary_text += "The future projection indicates a stabilization or potential plateau of the warming trend. "
        if end_slope < -0.05: summary_text += "The rate of warming is expected to decelerate and potentially show a slight cooling trend by the end of the century as the trend stabilizes."
        elif end_slope < future_slope: summary_text += "The rate of warming is expected to decelerate towards the end of the century."
        else: summary_text += "The rate of warming is expected to continue or accelerate towards the end of the century."
        story.append(Paragraph(summary_text, styles['Justify']))
        story.append(Spacer(1, 0.25*inch))
        story.append(Image(static_plot_path, width=7*inch, height=3.5*inch))
        story.append(PageBreak())
    doc.build(story)
    return tmp.name

# --- SECTION 6: MAIN CONTROLLER (WITH CACHING) ---

@st.cache_data(show_spinner=False)
def execute_methaine_engine(uploaded_files_data, config_params):
    # Reconstruct all_city_data from bytes
    all_city_data = []
    if uploaded_files_data:
        for file_name, file_bytes in uploaded_files_data.items():
            buf = io.BytesIO(file_bytes)
            all_city_data.append(preprocess_city(buf, city_name=file_name.replace(".csv","")))
    else:
        sample_buffer, city_name = get_embedded_sample_data()
        all_city_data = [preprocess_city(sample_buffer, city_name=city_name)]

    # Update config with data-specific years
    max_year = max(df['YEAR'].max() for df in all_city_data)
    config_params['HIST_FULL_END'] = max_year
    config_params['MODERN_END'] = max_year
    config_params['FUT_START'] = max_year + 1

    run_config = get_run_config(config_params)
    full_config = {**config_params, **run_config}
    
    quality_report = generate_data_quality_report(all_city_data)
    
    all_results = []
    for city_df in all_city_data:
        city_results = run_projection_for_city(city_df, full_config)
        for season, data in city_results.items():
            all_results.append(data)
            
    return all_results, quality_report, config_params

def main():
    if upload_your_own_data:
        uploaded_files = st.file_uploader("Upload NASA/GISTEMP CSV Files", type="csv", accept_multiple_files=True)
    else:
        uploaded_files = None

    if st.button("🚀 Run MethAIne Analysis"):
        # Prepare data for caching (Streamlit can't cache UploadedFile objects directly)
        files_data = {}
        if uploaded_files:
            for f in uploaded_files:
                files_data[f.name] = f.getvalue()
        
        with st.spinner("Processing climate models... This may take several minutes."):
            EXECUTION_ID = str(uuid.uuid4())
            
            # Run the engine (Cached)
            all_results, quality_report, updated_config = execute_methaine_engine(files_data, CONFIG)

            if not all_results:
                st.error("No results generated. Check your data quality.")
                return

            # Display Quality Report
            st.text_area("Data Quality Report", quality_report, height=200)

            # Interactive Plot
            st.plotly_chart(generate_interactive_plot(all_results, EXECUTION_ID), use_container_width=True)
            
            # Static Plot
            static_fig = generate_static_plot(all_results, EXECUTION_ID)
            st.pyplot(static_fig)

            # Summary Table
            summary_list = []
            for r in all_results:
                s = r['summary'].copy()
                modern_key = [k for k in s.keys() if k.startswith('MODERN_')][0]
                s['MODERN_SLOPE_C_per_dec'] = s.pop(modern_key)
                summary_list.append(s)
            
            df_out = pd.DataFrame(summary_list)
            st.subheader("Final Climate Slope Summary (°C/decade)")
            st.dataframe(df_out)
            
            # Downloads
            st.download_button("📥 Download CSV Results", df_out.to_csv(index=False), f"{run_name}_projections.csv", "text/csv")
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_img:
                static_fig.savefig(tmp_img.name, dpi=300)
                pdf_path = generate_pdf_report(all_results, EXECUTION_ID, tmp_img.name)
            
            with open(pdf_path, "rb") as f:
                st.download_button("📥 Download PDF Report", f, f"{run_name}_report.pdf", "application/pdf")
            
            # Cleanup
            plt.close(static_fig)

if __name__ == "__main__":
    main()
