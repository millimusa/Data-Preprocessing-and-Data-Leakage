# =============================================================================
#  Data Preprocessing and Data Leakage in Multivariate Time Series Forecasting
#
#  Experiments of the paper on the Beijing PM2.5 data set
#  (UCI Machine Learning Repository, Liang et al., 2015).
#
#  Single-cell Google Colab script.
#  Recommended runtime: Runtime > Change runtime type > T4 GPU.
#  Expected duration on a T4 runtime: about 80 to 90 minutes.
#  All tables, figures and forecasts are written to ./pm25_results and zipped
#  into pm25_results.zip, which is downloaded automatically at the end.
# =============================================================================
import os, sys, io, time, json, zipfile, warnings, logging, subprocess, urllib.request, random, platform
warnings.filterwarnings("ignore")
LOCAL = os.environ.get("PM25_LOCAL") == "1"      # used only for offline testing outside Colab
FAST = os.environ.get("PM25_FAST") == "1"        # quick smoke test (few models, few epochs)


def _need(module, pip_name):
    try:
        __import__(module)
    except ImportError:
        if not LOCAL:
            subprocess.run([sys.executable, "-m", "pip", "install", "-q", pip_name], check=False)


for _m, _p in [("statsmodels", "statsmodels"), ("xgboost", "xgboost"), ("prophet", "prophet")]:
    _need(_m, _p)

import numpy as np
import pandas as pd
import matplotlib
if LOCAL:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.patches import FancyBboxPatch, Patch
from matplotlib.lines import Line2D
from scipy import stats
from scipy.signal import savgol_coeffs
from sklearn.impute import KNNImputer
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor

try:
    from xgboost import XGBRegressor
except ImportError:
    XGBRegressor = None
try:
    from statsmodels.tsa.arima.model import ARIMA as SARIMAX      # ARIMA with exogenous regressors
    from statsmodels.tsa.stattools import adfuller
except ImportError:
    SARIMAX = None
try:
    from prophet import Prophet
    for _lg in ("cmdstanpy", "prophet", "prophet.plot"):
        logging.getLogger(_lg).setLevel(logging.ERROR)
        logging.getLogger(_lg).propagate = False
except ImportError:
    Prophet = None
try:
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    import tensorflow as tf
    from tensorflow import keras
    tf.get_logger().setLevel("ERROR")
except ImportError:
    tf = None

# -----------------------------------------------------------------------------
# 1. Experimental settings
# -----------------------------------------------------------------------------
SEED = 42
LOOKBACK = 24                                   # hours of history per forecast
TEST_START = pd.Timestamp("2014-01-01 00:00:00")
VAL_FRAC = 0.10                                 # last 10% of training windows for early stopping
DL_SEEDS = (0,) if FAST else (0, 1, 2)
DL_EPOCHS, DL_BATCH, DL_PATIENCE, DL_LR = (2 if FAST else 50), 512, 5, 1e-3
SG_WINDOW, SG_ORDER = 23, 2                     # causal Savitzky-Golay: noise variance factor ~ 1/3
MA_WINDOW, ES_ALPHA = 3, 0.5                    # noise variance factor 1/3 for both
RUN_SENSITIVITY = True
SENS_RATES = (0.10,) if FAST else (0.10, 0.20, 0.30)
SENS_MODELS = ("LR", "RF", "XGBoost", "LSTM")
OUT = "pm25_results"
FIG_TITLES = False                              # paper-ready figures: descriptions go into the captions
os.makedirs(OUT, exist_ok=True)

random.seed(SEED)
np.random.seed(SEED)

REFERENCE = {"imputation": "linear", "anomaly": "none", "encoding": "onehot",
             "normalization": "minmax", "smoothing": "none"}
STAGES = {
    "imputation": ["mean", "linear", "knn"],
    "anomaly": ["none", "zscore", "iqr", "iforest"],
    "encoding": ["label", "onehot", "target"],
    "normalization": ["none", "minmax", "zscore", "robust"],
    "smoothing": ["none", "ma", "es", "sg"],          # smoothing of the series (inputs and training targets)
    "smoothing_in": ["none", "ma", "es", "sg"],       # robustness check: smoothing of the model inputs only
}
STAGE_KEY = {"smoothing_in": "smoothing"}           # configuration key used by an analysis block
IF_CONTAMINATION = 0.02                             # share of training hours isolated by Isolation Forest
STAGE_TITLES = {"imputation": "Missing value imputation", "anomaly": "Anomaly handling",
                "encoding": "Categorical encoding", "normalization": "Normalization",
                "smoothing": "Smoothing", "smoothing_in": "Smoothing (inputs only)"}
METHOD_LABELS = {
    ("imputation", "mean"): "Mean", ("imputation", "linear"): "Linear interp.", ("imputation", "knn"): "KNN",
    ("anomaly", "none"): "None", ("anomaly", "zscore"): "Z-score", ("anomaly", "iqr"): "IQR",
    ("anomaly", "iforest"): "Isolation F.",
    ("encoding", "label"): "Label", ("encoding", "onehot"): "One-hot", ("encoding", "target"): "Target",
    ("normalization", "none"): "None", ("normalization", "minmax"): "Min-max",
    ("normalization", "zscore"): "Z-score", ("normalization", "robust"): "Robust",
    ("smoothing", "none"): "None", ("smoothing", "ma"): "Moving avg.", ("smoothing", "es"): "Exp. smoothing",
    ("smoothing", "sg"): "Savitzky-Golay",
    ("smoothing_in", "none"): "None", ("smoothing_in", "ma"): "Moving avg.",
    ("smoothing_in", "es"): "Exp. smoothing", ("smoothing_in", "sg"): "Savitzky-Golay",
}
ML_MODELS = ["LR", "DT", "RF", "XGBoost", "kNN"]
DL_MODELS = ["1D-CNN", "LSTM", "GRU"]
STAT_MODELS = ["ARIMAX", "Prophet"]
MODELS = ML_MODELS + DL_MODELS + STAT_MODELS
FAMILY = {**{m: "ML" for m in ML_MODELS}, **{m: "DL" for m in DL_MODELS},
          **{m: "Statistical" for m in STAT_MODELS}, "Persistence": "Baseline"}
MET = ["DEWP", "TEMP", "PRES", "Iws", "Is", "Ir"]
CATS = ["NE", "NW", "SE", "cv"]

# Figure style (colour-blind validated categorical palette, recessive chrome)
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
PAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
FAM_COL = {"ML": PAL[0], "DL": PAL[1], "Statistical": PAL[2], "Baseline": MUTED}
plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 300, "savefig.bbox": "tight", "font.size": 8.5,
    "font.family": "DejaVu Sans", "axes.titlesize": 9.5, "axes.titleweight": "bold",
    "axes.labelsize": 8.5, "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "axes.labelcolor": INK,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False, "xtick.color": INK2, "ytick.color": INK2,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.frameon": False, "legend.fontsize": 7.5,
    "text.color": INK, "axes.titlelocation": "left", "figure.facecolor": "white",
})
DIVERGING = LinearSegmentedColormap.from_list(
    "gain_loss", ["#1c5cab", "#6da7ec", "#f0efec", "#f19c93", "#b8302f"])
SEQUENTIAL = LinearSegmentedColormap.from_list("blues", ["#f5f9fe", "#b7d3f6", "#6da7ec", "#2a78d6", "#104281"])


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def title(ax, text, **kw):
    if FIG_TITLES:
        ax.set_title(text, **kw)


def save_fig(fig, name):
    fig.savefig(os.path.join(OUT, name))
    if not LOCAL:
        plt.show()
    plt.close(fig)


# -----------------------------------------------------------------------------
# 2. Data
# -----------------------------------------------------------------------------
def load_data():
    df = None
    for p in ["PRSA_data_2010.1.1-2014.12.31.csv", "pollution.csv", os.environ.get("PM25_CSV", "")]:
        if p and os.path.exists(p):
            df = pd.read_csv(p)
            break
    urls = ["https://archive.ics.uci.edu/static/public/381/beijing+pm2+5+data.zip",
            "https://archive.ics.uci.edu/ml/machine-learning-databases/00381/PRSA_data_2010.1.1-2014.12.31.csv",
            "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pollution.csv"]
    for u in urls:
        if df is not None:
            break
        try:
            raw = urllib.request.urlopen(u, timeout=60).read()
            if u.endswith(".zip"):
                with zipfile.ZipFile(io.BytesIO(raw)) as z:
                    df = pd.read_csv(z.open([n for n in z.namelist() if n.endswith(".csv")][0]))
            else:
                df = pd.read_csv(io.BytesIO(raw))
        except Exception as e:
            log(f"download failed ({u}): {e}")
    if df is None:
        raise RuntimeError("Beijing PM2.5 data could not be loaded.")
    df["datetime"] = pd.to_datetime(df[["year", "month", "day", "hour"]])
    df = df.set_index("datetime").rename(columns={"pm2.5": "pm25"})
    df = df[["pm25"] + MET + ["cbwd"]].astype({c: float for c in ["pm25"] + MET})
    assert len(df) == 43824, "unexpected number of records"
    return df.loc["2010-01-02":]           # PM2.5 was not recorded at all on 1 January 2010


def calendar_features(idx):
    h = idx.hour.values.astype(float)
    d = idx.dayofyear.values.astype(float)
    return np.column_stack([np.sin(2 * np.pi * h / 24), np.cos(2 * np.pi * h / 24),
                            np.sin(2 * np.pi * d / 365.25), np.cos(2 * np.pi * d / 365.25)])


# -----------------------------------------------------------------------------
# 3. Preprocessing operators (all fitted on the training period only)
# -----------------------------------------------------------------------------
def impute_train(y_tr, method, ctx_tr):
    """Imputation of the training history. y_tr: pd.Series with NaN; ctx_tr: covariates for KNN."""
    if not y_tr.isna().any():
        return y_tr.copy()
    if method == "mean":
        return y_tr.fillna(y_tr.mean())
    if method == "linear":
        return y_tr.interpolate(method="linear", limit_direction="both")
    if method == "knn":
        M = np.column_stack([y_tr.values, ctx_tr])
        mu, sd = np.nanmean(M, axis=0), np.nanstd(M, axis=0)
        sd[sd == 0] = 1.0
        out = KNNImputer(n_neighbors=5).fit_transform((M - mu) / sd)
        return pd.Series(out[:, 0] * sd[0] + mu[0], index=y_tr.index)
    raise ValueError(method)


def detect_anomalies(y_tr, y_te, method, feat_tr, feat_te, seed=SEED):
    """Returns boolean flags for the training and test periods and the fitted thresholds."""
    if method == "none":
        return np.zeros(len(y_tr), bool), np.zeros(len(y_te), bool), {}
    if method in ("zscore", "iqr"):
        if method == "zscore":
            mu, sd = float(y_tr.mean()), float(y_tr.std())
            lo, hi = mu - 3 * sd, mu + 3 * sd
        else:
            q1, q3 = np.percentile(y_tr.values, [25, 75])
            lo, hi = q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1)
        f_tr = (y_tr.values < lo) | (y_tr.values > hi)
        f_te = (y_te.values < lo) | (y_te.values > hi)
        return f_tr, f_te, {"lower": lo, "upper": hi}
    if method == "iforest":
        mu, sd = feat_tr.mean(0), feat_tr.std(0)
        sd[sd == 0] = 1.0
        iso = IsolationForest(n_estimators=200, contamination=IF_CONTAMINATION, random_state=seed, n_jobs=-1)
        iso.fit((feat_tr - mu) / sd)
        return iso.predict((feat_tr - mu) / sd) == -1, iso.predict((feat_te - mu) / sd) == -1, {}
    raise ValueError(method)


def encode_wind(cbwd, method, y_tr_clean, tr):
    if method == "label":                    # alphabetical integer codes, as LabelEncoder would assign
        return cbwd.map({c: i for i, c in enumerate(CATS)}).values.astype(float)[:, None]
    if method == "onehot":
        return np.column_stack([(cbwd.values == c).astype(float) for c in CATS])
    if method == "target":                   # smoothed mean PM2.5 per wind regime (training period)
        s = pd.Series(y_tr_clean.values, index=cbwd.index[tr])
        g = s.groupby(cbwd[tr].values).agg(["mean", "count"])
        m, gm = 20.0, float(s.mean())
        enc = (g["count"] * g["mean"] + m * gm) / (g["count"] + m)
        return cbwd.map(enc).values.astype(float)[:, None]
    raise ValueError(method)


class Identity:
    def fit(self, X):
        return self

    def transform(self, X):
        return np.asarray(X, float)

    def inverse_transform(self, X):
        return np.asarray(X, float)


SCALERS = {"none": Identity, "minmax": MinMaxScaler, "zscore": StandardScaler, "robust": RobustScaler}
SG_COEF = savgol_coeffs(SG_WINDOW, SG_ORDER, pos=SG_WINDOW - 1, use="dot")   # evaluated at the newest point


def smooth(z, method):
    """Causal (one-sided) smoothers: every value depends only on the present and the past."""
    if method == "none":
        return z.copy()
    if method == "ma":
        return pd.Series(z).rolling(MA_WINDOW, min_periods=1).mean().values
    if method == "es":
        return pd.Series(z).ewm(alpha=ES_ALPHA, adjust=False).mean().values
    if method == "sg":
        out = z.copy()
        out[SG_WINDOW - 1:] = np.lib.stride_tricks.sliding_window_view(z, SG_WINDOW) @ SG_COEF
        return out
    raise ValueError(method)


def preprocess(df, cfg, extra_nan=None):
    tr = np.asarray(df.index < TEST_START)
    te = ~tr
    y = df["pm25"].copy()
    if extra_nan is not None:
        y = y.mask(extra_nan)
    met = df[MET].values.astype(float)
    cal = calendar_features(df.index)
    onehot = np.column_stack([(df["cbwd"].values == c).astype(float) for c in CATS])
    ctx = np.column_stack([met, onehot, cal])
    info = {"n_missing_train": int(y[tr].isna().sum()), "n_missing_test": int(y[te].isna().sum())}
    # Stage 1: imputation of the training history; causal carry-forward in the test period
    y_tr = impute_train(y[tr], cfg["imputation"], ctx[tr])
    y_te = pd.concat([y_tr.iloc[-1:], y[te]]).ffill().iloc[1:]
    # Stage 2: anomaly detection (thresholds learnt on training data), flagged values treated as missing
    feat = np.column_stack([np.r_[y_tr.values, y_te.values], met[:, :3], np.log1p(met[:, 3])])
    f_tr, f_te, thr = detect_anomalies(y_tr, y_te, cfg["anomaly"], feat[tr], feat[te])
    info.update({"n_flag_train": int(f_tr.sum()), "n_flag_test": int(f_te.sum()),
                 **{f"thr_{k}": float(v) for k, v in thr.items()}})
    if f_tr.any():
        y_tr = impute_train(y_tr.mask(f_tr), cfg["imputation"], ctx[tr])
    if f_te.any():
        y_te = pd.concat([y_tr.iloc[-1:], y_te.mask(f_te)]).ffill().iloc[1:]
    y_clean = np.r_[y_tr.values, y_te.values]
    # Stage 3: encoding of the wind direction
    wind = encode_wind(df["cbwd"], cfg["encoding"], y_tr, tr)
    # Stage 4: normalization of every input column and of the target
    Xcov = np.column_stack([met, wind, cal])
    sx = SCALERS[cfg["normalization"]]().fit(Xcov[tr])
    sy = SCALERS[cfg["normalization"]]().fit(y_clean[tr][:, None])
    Xcov = sx.transform(Xcov)
    z = sy.transform(y_clean[:, None]).ravel()
    # Stage 5: causal smoothing of the PM2.5 series. By default the smoothed series supplies both the
    # model inputs and the training targets; with smooth_target=False only the inputs are smoothed.
    z_s = smooth(z, cfg["smoothing"])
    z_tgt = z_s if cfg.get("smooth_target", True) else z
    return dict(z=z_tgt, z_in=z_s, Xcov=Xcov, n_cov=len(MET) + wind.shape[1], n_wind=wind.shape[1], sy=sy,
                y_clean=y_clean, flags=(f_tr, f_te), info=info, tr=tr, te=te)


def make_windows(P):
    z_in, X, L = P["z_in"], P["Xcov"], LOOKBACK
    T = len(z_in)
    origin = np.arange(L - 1, T - 1)
    target = origin + 1
    lags = np.column_stack([z_in[origin - k] for k in range(L)])
    X_tab = np.column_stack([lags, X[origin, :P["n_cov"]], X[target, P["n_cov"]:]])
    seq = np.column_stack([z_in, X]).astype(np.float32)
    X_seq = np.lib.stride_tricks.sliding_window_view(seq, (L, seq.shape[1]))[:, 0][: len(origin)]
    is_test = P["te"][target]
    return dict(X_tab=X_tab, X_seq=X_seq, y=P["z"][target], target=target, is_test=is_test)


# -----------------------------------------------------------------------------
# 4. Forecasting models
# -----------------------------------------------------------------------------
def make_ml(name, seed=SEED):
    if name == "LR":
        return LinearRegression()
    if name == "DT":
        return DecisionTreeRegressor(max_depth=12, min_samples_leaf=20, random_state=seed)
    if name == "RF":
        return RandomForestRegressor(n_estimators=20 if FAST else 150, max_depth=20, min_samples_leaf=5,
                                     max_features=0.33, n_jobs=-1, random_state=seed)
    if name == "XGBoost":
        return XGBRegressor(n_estimators=50 if FAST else 500, learning_rate=0.05, max_depth=6, subsample=0.8,
                            colsample_bytree=0.8, tree_method="hist", n_jobs=-1, random_state=seed,
                            verbosity=0)
    if name == "kNN":
        return KNeighborsRegressor(n_neighbors=10, n_jobs=-1)
    raise ValueError(name)


def build_net(kind, L, F, seed):
    keras.utils.set_random_seed(seed)
    inp = keras.Input(shape=(L, F))
    if kind == "1D-CNN":
        x = keras.layers.Conv1D(64, 3, activation="relu")(inp)
        x = keras.layers.Conv1D(64, 3, activation="relu")(x)
        x = keras.layers.Flatten()(x)
        x = keras.layers.Dense(64, activation="relu")(x)
    elif kind == "LSTM":
        x = keras.layers.LSTM(64)(inp)
    elif kind == "GRU":
        x = keras.layers.GRU(64)(inp)
    else:
        raise ValueError(kind)
    net = keras.Model(inp, keras.layers.Dense(1)(x))
    net.compile(optimizer=keras.optimizers.Adam(DL_LR), loss="mse")
    return net


def run_dl(kind, W, seeds=DL_SEEDS):
    tr_idx = np.where(~W["is_test"])[0]
    n_val = int(round(VAL_FRAC * len(tr_idx)))
    fit_idx, val_idx = tr_idx[:-n_val], tr_idx[-n_val:]
    Xf, yf = np.ascontiguousarray(W["X_seq"][fit_idx]), W["y"][fit_idx].astype(np.float32)
    Xv, yv = np.ascontiguousarray(W["X_seq"][val_idx]), W["y"][val_idx].astype(np.float32)
    Xt = np.ascontiguousarray(W["X_seq"][W["is_test"]])
    preds, epochs = [], []
    for s in seeds:
        net = build_net(kind, Xf.shape[1], Xf.shape[2], s)
        stop = keras.callbacks.EarlyStopping(monitor="val_loss", patience=DL_PATIENCE, restore_best_weights=True)
        hist = net.fit(Xf, yf, validation_data=(Xv, yv), epochs=DL_EPOCHS, batch_size=DL_BATCH,
                       callbacks=[stop], verbose=0, shuffle=True)
        preds.append(net.predict(Xt, batch_size=4096, verbose=0).ravel())
        epochs.append(int(np.argmin(hist.history["val_loss"]) + 1))
        keras.backend.clear_session()
    return np.array(preds), epochs


def arimax_exog(P):
    X, k = P["Xcov"], P["n_cov"]
    keep = k - 1 if P["n_wind"] > 1 else k                           # drop one dummy (collinear with the constant)
    lagged = np.vstack([np.full((1, keep), np.nan), X[:-1, :keep]])  # covariates known at the forecast origin
    return np.column_stack([lagged, X[:, k:]])                       # plus calendar terms of the target hour


def select_arima_order(P):
    """AIC-based choice of (p, 0, q) on the training period of the reference configuration."""
    E = arimax_exog(P)
    rows = np.arange(LOOKBACK, int(P["tr"].sum()))
    grid = []
    for p in (1, 2, 3):
        for q in (0, 1, 2):
            try:
                r = SARIMAX(P["z"][rows], exog=E[rows], order=(p, 0, q), trend="c").fit(method="innovations_mle")
                grid.append({"p": p, "d": 0, "q": q, "aic": float(r.aic), "bic": float(r.bic)})
            except Exception as e:
                log(f"ARIMA({p},0,{q}) failed: {e}")
    grid = pd.DataFrame(grid).sort_values("aic")
    grid.to_csv(os.path.join(OUT, "arima_order_selection.csv"), index=False)
    best = grid.iloc[0]
    return (int(best.p), 0, int(best.q))


def run_arimax(P, order):
    """Regression with ARMA errors: FGLS for the regression part, innovations MLE for the ARMA part.
    Parameters are estimated once on the training period; the test period is filtered with them fixed,
    which yields genuine one-step-ahead forecasts."""
    E = arimax_exog(P)
    rows = np.arange(LOOKBACK, int(P["tr"].sum()))
    te_idx = np.where(P["te"])[0]
    res = SARIMAX(P["z"][rows], exog=E[rows], order=order, trend="c").fit(method="innovations_mle")
    res_all = res.append(P["z"][te_idx], exog=E[te_idx])
    return np.asarray(res_all.predict(start=len(rows), end=len(rows) + len(te_idx) - 1))


def run_prophet(P, index):
    """Prophet (trend + daily, weekly and yearly seasonality) with the recent PM2.5 history and the
    meteorological covariates known at the forecast origin as extra regressors."""
    z, X, k = P["z"], P["Xcov"], P["n_cov"]
    d = pd.DataFrame({"ds": index.values, "y": z})
    for lag in (1, 2, 3, 24):
        d[f"lag{lag}"] = pd.Series(P["z_in"]).shift(lag).values
    lagged = np.vstack([np.full((1, k), np.nan), X[:-1, :k]])
    for j in range(k):
        d[f"x{j}"] = lagged[:, j]
    regs = [c for c in d.columns if c not in ("ds", "y")]
    n_tr = int(P["tr"].sum())
    m = Prophet(daily_seasonality=True, weekly_seasonality=True, yearly_seasonality=True, uncertainty_samples=0)
    for c in regs:
        m.add_regressor(c)
    m.fit(d.iloc[LOOKBACK:n_tr])
    return m.predict(d.iloc[np.where(P["te"])[0]].drop(columns="y"))["yhat"].values


# -----------------------------------------------------------------------------
# 5. Evaluation
# -----------------------------------------------------------------------------
def scores(y_true, y_pred):
    m = np.isfinite(y_true)
    e = y_pred[m] - y_true[m]
    pos = y_true[m] > 0
    return (float(np.mean(np.abs(e))), float(np.sqrt(np.mean(e ** 2))),
            float(100 * np.mean(np.abs(e[pos]) / y_true[m][pos])))


def dm_test(e_variant, e_reference, h=1):
    """Diebold-Mariano test on absolute errors with Newey-West variance and the HLN correction.
    A positive statistic means that the variant is less accurate than the reference."""
    d = np.abs(e_variant) - np.abs(e_reference)
    n = len(d)
    dc = d - d.mean()
    lag = int(np.floor(4 * (n / 100) ** (2 / 9)))
    lrv = np.dot(dc, dc) / n
    for k in range(1, lag + 1):
        lrv += 2 * (1 - k / (lag + 1)) * np.dot(dc[k:], dc[:-k]) / n
    stat = d.mean() / np.sqrt(lrv / n)
    stat *= np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    p = 2 * stats.t.sf(abs(stat), df=n - 1)
    return float(stat), float(p)


def build_configs():
    cfgs = [("reference", REFERENCE["imputation"], dict(REFERENCE))]
    for stage, methods in STAGES.items():
        key = STAGE_KEY.get(stage, stage)
        for mth in methods:
            if mth != REFERENCE[key]:
                c = dict(REFERENCE)
                c[key] = mth
                if stage == "smoothing_in":
                    c["smooth_target"] = False
                cfgs.append((stage, mth, c))
    if FAST:
        keep = {("reference", "linear"), ("imputation", "knn"), ("anomaly", "iforest"), ("encoding", "target"),
                ("normalization", "none"), ("smoothing", "sg"), ("smoothing_in", "ma")}
        cfgs = [c for c in cfgs if (c[0], c[1]) in keep]
    return cfgs


def holm(p):
    """Holm step-down adjustment of a vector of p-values."""
    p = np.asarray(p, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def available_models():
    ok = list(ML_MODELS)
    if XGBRegressor is None:
        ok.remove("XGBoost")
    if tf is not None:
        ok += DL_MODELS
    if SARIMAX is not None:
        ok.append("ARIMAX")
    if Prophet is not None:
        ok.append("Prophet")
    return ok


# -----------------------------------------------------------------------------
# 6. Main experiment
# -----------------------------------------------------------------------------
T0 = time.time()
df = load_data()
IDX = df.index
TR = np.asarray(IDX < TEST_START)
TE = ~TR
TE_IDX = np.where(TE)[0]
Y_RAW = df["pm25"].values
Y_TEST = Y_RAW[TE_IDX]
PERSIST = df["pm25"].ffill().values[TE_IDX - 1]         # last observed value at the forecast origin
RUN_MODELS = available_models()
gpu = [] if tf is None else tf.config.list_physical_devices("GPU")
log(f"records: {len(df)} | training hours: {TR.sum()} | test hours: {TE.sum()} | "
    f"observed test targets: {np.isfinite(Y_TEST).sum()} | models: {RUN_MODELS} | GPU: {bool(gpu)}")

adf_stat, adf_p = (np.nan, np.nan)
if SARIMAX is not None:
    _y = df["pm25"][TR].interpolate(limit_direction="both").values
    adf_stat, adf_p = adfuller(_y[-8760:], maxlag=48, autolag="AIC")[:2]

CONFIGS = build_configs()
rows, preds, seed_preds, prep_info = [], {}, {}, []
arima_order = (1, 0, 0) if FAST else None
for ci, (stage, method, cfg) in enumerate(CONFIGS):
    cid = "reference" if stage == "reference" else f"{stage}:{method}"
    t_cfg = time.time()
    P = preprocess(df, cfg)
    W = make_windows(P)
    prep_info.append({"config": cid, **P["info"]})
    Xtr, ytr = W["X_tab"][~W["is_test"]], W["y"][~W["is_test"]]
    Xte = W["X_tab"][W["is_test"]]
    y_app = P["sy"].inverse_transform(P["z"][TE_IDX][:, None]).ravel()     # preprocessed test target
    if "ARIMAX" in RUN_MODELS and arima_order is None:
        arima_order = select_arima_order(P)
        log(f"ARIMAX order selected by AIC: {arima_order}")
    for name in RUN_MODELS:
        if name == "ARIMAX" and stage == "smoothing_in":
            continue                     # ARIMAX models the series itself, so inputs cannot be smoothed alone
        t0 = time.time()
        extra = {}
        try:
            if name in ML_MODELS:
                raw = make_ml(name).fit(Xtr, ytr).predict(Xte)
            elif name in DL_MODELS:
                sp, ep = run_dl(name, W)
                sp = np.array([P["sy"].inverse_transform(s[:, None]).ravel() for s in sp])
                seed_preds[(cid, name)] = np.clip(sp, 0, None).astype(np.float32)
                seed_mae = [scores(Y_TEST, np.clip(s, 0, None))[0] for s in sp]
                extra = {"seed_sd_MAE": float(np.std(seed_mae, ddof=1)) if len(sp) > 1 else np.nan,
                         "epochs": ";".join(map(str, ep))}
                raw = None
                yhat = np.clip(sp.mean(0), 0, None)
            elif name == "ARIMAX":
                raw = run_arimax(P, arima_order)
            elif name == "Prophet":
                raw = run_prophet(P, IDX)
            if raw is not None:
                yhat = np.clip(P["sy"].inverse_transform(np.asarray(raw)[:, None]).ravel(), 0, None)
            if not np.all(np.isfinite(yhat)):
                raise ValueError("non-finite forecasts")
        except Exception as e:
            log(f"  {cid} | {name} FAILED: {e}")
            continue
        preds[(cid, name)] = yhat.astype(np.float32)
        mae, rmse, mape = scores(Y_TEST, yhat)
        mae_a, rmse_a, mape_a = scores(np.where(np.isfinite(Y_TEST), y_app, np.nan), yhat)
        rows.append({"config": cid, "stage": stage, "method": method, "model": name, "family": FAMILY[name],
                     "MAE": mae, "RMSE": rmse, "MAPE": mape, "MAE_apparent": mae_a, "RMSE_apparent": rmse_a,
                     "MAPE_apparent": mape_a, "seconds": time.time() - t0, **extra})
        log(f"  [{ci + 1}/{len(CONFIGS)}] {cid:24s} {name:8s} MAE={mae:7.3f} RMSE={rmse:7.3f} "
            f"MAPE={mape:6.2f}% ({time.time() - t0:5.1f}s)")
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "results_all.csv"), index=False)
    log(f"configuration {cid} done in {time.time() - t_cfg:.0f}s (total {(time.time() - T0) / 60:.1f} min)")

res = pd.DataFrame(rows)
mae_p, rmse_p, mape_p = scores(Y_TEST, PERSIST)
persist_row = {"model": "Persistence", "family": "Baseline", "MAE": mae_p, "RMSE": rmse_p, "MAPE": mape_p}
pd.DataFrame(prep_info).to_csv(os.path.join(OUT, "preprocessing_info.csv"), index=False)
np.savez_compressed(os.path.join(OUT, "test_forecasts.npz"), y_test=Y_TEST, persistence=PERSIST,
                    **{f"{c}|{m}": v for (c, m), v in preds.items()})

# -----------------------------------------------------------------------------
# 7. Tables and significance tests
# -----------------------------------------------------------------------------
obs = np.isfinite(Y_TEST)
dm_rows = []
for (cid, name), yhat in preds.items():
    if cid == "reference" or ("reference", name) not in preds:
        continue
    st, p = dm_test(yhat[obs] - Y_TEST[obs], preds[("reference", name)][obs] - Y_TEST[obs])
    dm_rows.append({"config": cid, "model": name, "DM": st, "p_value": p})
dm = pd.DataFrame(dm_rows, columns=["config", "model", "DM", "p_value"])
dm["p_holm"] = np.nan
for name, g in dm.groupby("model"):
    dm.loc[g.index, "p_holm"] = holm(g["p_value"].values)      # family-wise control within each model
dm.to_csv(os.path.join(OUT, "dm_tests.csv"), index=False)
res = res.merge(dm, on=["config", "model"], how="left")
ref = res[res.config == "reference"].set_index("model")
res["MAE_change_pct"] = 100 * (res["MAE"] / res["model"].map(ref["MAE"]) - 1)
res["RMSE_change_pct"] = 100 * (res["RMSE"] / res["model"].map(ref["RMSE"]) - 1)
res["MAPE_change_pct"] = 100 * (res["MAPE"] / res["model"].map(ref["MAPE"]) - 1)
res.to_csv(os.path.join(OUT, "results_all.csv"), index=False)

order_models = [m for m in MODELS if m in ref.index]
tab_ref = ref.loc[order_models, ["family", "MAE", "RMSE", "MAPE"]].copy()
if "seed_sd_MAE" in ref:
    tab_ref["seed_sd_MAE"] = ref.loc[order_models, "seed_sd_MAE"]
tab_ref.loc["Persistence"] = ["Baseline", mae_p, rmse_p, mape_p] + ([np.nan] if "seed_sd_MAE" in tab_ref else [])
tab_ref.round(3).to_csv(os.path.join(OUT, "table_reference.csv"))


def stage_frame(stage):
    """All results of one stage, with the reference configuration standing in for the reference method."""
    part = res[res.stage == stage].copy()
    r = res[res.config == "reference"].copy()
    r["stage"], r["method"] = stage, REFERENCE[STAGE_KEY.get(stage, stage)]
    if stage == "smoothing_in":
        r = r[r.model != "ARIMAX"]
    return pd.concat([r, part])


for stage, methods in STAGES.items():
    fr = stage_frame(stage)
    if fr.empty:
        continue
    wide = fr.pivot_table(index="model", columns="method", values=["MAE", "RMSE", "MAPE"])
    wide = wide.reindex(index=[m for m in order_models if m in wide.index])
    wide = wide.reindex(columns=[(mt, m) for mt in ["MAE", "RMSE", "MAPE"] for m in methods if (mt, m) in wide.columns])
    wide.round(3).to_csv(os.path.join(OUT, f"table_{stage}.csv"))

# -----------------------------------------------------------------------------
# 8. Missing-data sensitivity: extra gaps drawn from the empirical gap-length distribution
# -----------------------------------------------------------------------------
sens_rows, imp_rows = [], []
if RUN_SENSITIVITY:
    miss = df["pm25"].isna().values
    n_tr = int(TR.sum())
    runs = (np.diff(np.r_[0, miss[:n_tr].astype(int), 0]))
    gap_len = np.where(runs == -1)[0] - np.where(runs == 1)[0]
    for rate in SENS_RATES:
        rng = np.random.default_rng(SEED + int(rate * 100))
        extra = np.zeros(len(df), bool)
        while (extra & ~miss)[:n_tr].sum() < rate * n_tr:
            L = int(rng.choice(gap_len))
            s = int(rng.integers(LOOKBACK, n_tr - L))
            extra[s:s + L] = True
        extra &= ~miss
        for imp in STAGES["imputation"]:
            cfg = dict(REFERENCE)
            cfg["imputation"] = imp
            P = preprocess(df, cfg, extra_nan=extra)
            err = P["y_clean"][extra] - Y_RAW[extra]
            imp_rows.append({"rate": rate, "imputation": imp, "imputation_MAE": float(np.mean(np.abs(err))),
                             "imputation_RMSE": float(np.sqrt(np.mean(err ** 2))), "n_masked": int(extra.sum())})
            W = make_windows(P)
            Xtr, ytr = W["X_tab"][~W["is_test"]], W["y"][~W["is_test"]]
            Xte = W["X_tab"][W["is_test"]]
            for name in SENS_MODELS:
                if name not in RUN_MODELS:
                    continue
                if name in ML_MODELS:
                    raw = make_ml(name).fit(Xtr, ytr).predict(Xte)
                else:
                    raw = run_dl(name, W, seeds=(0,))[0][0]
                yhat = np.clip(P["sy"].inverse_transform(np.asarray(raw)[:, None]).ravel(), 0, None)
                mae, rmse, mape = scores(Y_TEST, yhat)
                sens_rows.append({"rate": rate, "imputation": imp, "model": name, "MAE": mae, "RMSE": rmse,
                                  "MAPE": mape})
                log(f"  sensitivity rate={rate:.2f} {imp:6s} {name:8s} MAE={mae:.3f}")
    # rate 0 = original data (single-seed LSTM for comparability)
    for imp in STAGES["imputation"]:
        cid = "reference" if imp == REFERENCE["imputation"] else f"imputation:{imp}"
        for name in SENS_MODELS:
            if (cid, name) in preds:
                yhat = seed_preds[(cid, name)][0] if (cid, name) in seed_preds else preds[(cid, name)]
                mae, rmse, mape = scores(Y_TEST, yhat)
                sens_rows.append({"rate": 0.0, "imputation": imp, "model": name, "MAE": mae, "RMSE": rmse,
                                  "MAPE": mape})
    pd.DataFrame(sens_rows).to_csv(os.path.join(OUT, "sensitivity_missing.csv"), index=False)
    pd.DataFrame(imp_rows).to_csv(os.path.join(OUT, "imputation_error.csv"), index=False)

# -----------------------------------------------------------------------------
# 9. Figures
# -----------------------------------------------------------------------------
def fig_workflow():
    fig, ax = plt.subplots(figsize=(7.4, 3.7))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 52)
    ax.axis("off")

    def box(x, y, w, h, title, lines, face="#f5f9fe", edge="#86b6ef", bold_line=None, note=None):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.25,rounding_size=1.2",
                                    fc=face, ec=edge, lw=0.9))
        ax.text(x + w / 2, y + h - 2.0, title, ha="center", va="top", fontsize=7.3, weight="bold", color=INK)
        for i, ln in enumerate(lines):
            ax.text(x + w / 2, y + h - 6.6 - i * 3.3, ln, ha="center", va="top", fontsize=6.5,
                    color=INK if ln == bold_line else INK2, weight="bold" if ln == bold_line else "normal")
        if note:
            ax.text(x + w / 2, y + 1.2, note, ha="center", va="bottom", fontsize=5.9, color=MUTED, style="italic")

    def arrow(x0, y0, x1, y1):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=0.9, shrinkA=0, shrinkB=0))

    steps = [("1  Imputation", ["Mean", "Linear interp.", "KNN"], "Linear interp.", None),
             ("2  Anomalies", ["None", "Z-score (3 SD)", "IQR (1.5 IQR)", "Isolation Forest"], "None", None),
             ("3  Encoding", ["Label", "One-hot", "Target"], "One-hot", None),
             ("4  Normalization", ["None", "Min-max", "Z-score", "Robust"], "Min-max", None),
             ("5  Smoothing", ["None", "Moving avg.", "Exp. smoothing", "Savitzky-Golay"], "None",
              "series, or inputs only")]
    ax.text(0.8, 51.5, "Preprocessing stages (reference choice in bold; one stage is varied at a time)",
            fontsize=7.5, color=INK2, va="top")
    w, gap, x0 = 17.0, 3.1, 0.8
    for i, (t, ls, b, note) in enumerate(steps):
        x = x0 + i * (w + gap)
        box(x, 25, w, 21.5, t, ls, bold_line=b, note=note)
        if i < 4:
            arrow(x + w + 0.4, 35.5, x + w + gap - 0.4, 35.5)
    lower = [("Supervised windows", ["24 h of PM2.5, weather,", "wind and calendar terms", "target: PM2.5 at t + 1 h"]),
             ("Forecasting models", ["ML: LR, DT, RF, XGBoost, kNN", "DL: 1D-CNN, LSTM, GRU",
                                     "Statistical: ARIMAX, Prophet"]),
             ("Evaluation on 2014", ["raw hourly observations", "MAE, RMSE, MAPE",
                                     "Diebold-Mariano vs. reference"])]
    lw_, xs = 29.8, [0.8, 35.0, 69.2]
    for i, (t, ls) in enumerate(lower):
        box(xs[i], 1.0, lw_, 16.0, t, ls, face="#fbf3ee" if i == 1 else "#f7f7f5",
            edge="#f0b89f" if i == 1 else AXIS)
        if i < 2:
            arrow(xs[i] + lw_ + 0.4, 8.6, xs[i + 1] - 0.4, 8.6)
    xl = x0 + 4 * (w + gap) + w / 2
    ax.plot([xl, xl, xs[0] + lw_ / 2, xs[0] + lw_ / 2], [24.7, 20.6, 20.6, 18.2], color=MUTED, lw=0.9)
    ax.annotate("", xy=(xs[0] + lw_ / 2, 17.35), xytext=(xs[0] + lw_ / 2, 18.4),
                arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=0.9, shrinkA=0, shrinkB=0))
    ax.text(57, 21.6, "all preprocessing methods fitted on 2010-2013 only and applied to 2014", fontsize=6.4,
            color=MUTED, ha="center", va="bottom", style="italic")
    save_fig(fig, "fig01_workflow.png")


def fig_data():
    fig = plt.figure(figsize=(7.4, 5.4))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.15, 1], hspace=0.45, wspace=0.3)
    ax = fig.add_subplot(gs[0, :])
    s = df["pm25"]
    ax.axvspan(TEST_START, IDX[-1], color="#f0efec", lw=0)
    ax.plot(IDX, s.values, color=PAL[0], lw=0.35, label="Hourly PM2.5")
    mi = IDX[s.isna().values]
    ax.plot(mi, np.full(len(mi), -38), "|", color=PAL[7], ms=5, mew=0.6, label="Missing hour")
    ax.set_ylim(-70, 1000)
    ax.set_ylabel("PM2.5 (µg/m³)")
    ax.set_title("(a) Hourly PM2.5 concentration, 2010-2014", pad=4)
    ax.text(IDX[int(TR.sum() * 0.5)], 940, "Training and validation (2010-2013)", ha="center", fontsize=7.5,
            color=INK2)
    ax.text(TEST_START + (IDX[-1] - TEST_START) / 2, 940, "Test (2014)", ha="center", fontsize=7.5, color=INK2)
    ax.legend(loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=2, markerscale=1.2)
    ax2 = fig.add_subplot(gs[1, 0])
    mr = s.isna().groupby([IDX.year, IDX.month]).mean().unstack() * 100
    ax2.imshow(mr.values, cmap=SEQUENTIAL, aspect="auto", vmin=0, vmax=max(20, np.nanmax(mr.values)))
    for i in range(mr.shape[0]):
        for j in range(mr.shape[1]):
            v = mr.values[i, j]
            ax2.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=6.2,
                     color="white" if v > 12 else INK2)
    ax2.set_xticks(range(12))
    ax2.set_xticklabels(list("JFMAMJJASOND"))
    ax2.set_yticks(range(mr.shape[0]))
    ax2.set_yticklabels(mr.index)
    ax2.grid(False)
    ax2.set_title("(b) Missing PM2.5 hours per month (%)", pad=4)
    ax3 = fig.add_subplot(gs[1, 1])
    v = s[TR].dropna().values
    ax3.hist(v, bins=np.arange(0, 1000, 10), color=PAL[0], lw=0)
    ax3.set_yscale("log")
    q1, q3 = np.percentile(v, [25, 75])
    thr = [(q3 + 1.5 * (q3 - q1), "IQR fence", PAL[1], "right"), (v.mean() + 3 * v.std(), "mean + 3 SD", PAL[6], "left")]
    for x, lab, c, side in thr:
        ax3.axvline(x, color=c, lw=1.1)
        ax3.text(x + (-10 if side == "right" else 10), 2500, f"{lab}\n{x:.0f} µg/m³", fontsize=6.6, color=c,
                 ha=side, va="top")
    ax3.set_xlabel("PM2.5 (µg/m³), training period")
    ax3.set_ylabel("Hours (log scale)")
    ax3.set_title("(c) Distribution and outlier thresholds", pad=4)
    save_fig(fig, "fig02_data.png")


def fig_anomaly():
    win = (IDX >= "2013-01-06") & (IDX < "2013-01-20")
    fig, axes = plt.subplots(3, 1, figsize=(7.4, 5.8), sharex=True)
    names = {"zscore": "Z-score (3 SD)", "iqr": "IQR (1.5 IQR)", "iforest": "Isolation Forest"}
    for ax, meth, col in zip(axes, ["zscore", "iqr", "iforest"], [PAL[6], PAL[1], PAL[2]]):
        cfg = dict(REFERENCE)
        cfg["anomaly"] = meth
        P = preprocess(df, cfg)
        f_tr = P["flags"][0]
        flag = np.r_[f_tr, P["flags"][1]]
        ax.plot(IDX[win], df["pm25"].values[win], color=MUTED, lw=0.9)
        ax.plot(IDX[win], P["y_clean"][win], color=col, lw=1.3)
        fw = win & flag
        ax.plot(IDX[fw], df["pm25"].values[fw], "o", ms=3.0, mfc=col, mec="white", mew=0.4)
        ax.set_title(f"{names[meth]}: {100 * f_tr.mean():.1f}% of training hours flagged", pad=3)
        ax.set_ylabel("PM2.5 (µg/m³)")
        if "thr_upper" in P["info"]:
            ax.axhline(P["info"]["thr_upper"], color=col, lw=0.8, alpha=0.6)
    handles = [Line2D([], [], color=MUTED, lw=1.2, label="Observed"),
               Line2D([], [], color=INK2, lw=1.4, label="Series after replacement (colour of each method)"),
               Line2D([], [], color=INK2, marker="o", lw=0, ms=4, label="Flagged hour")]
    fig.legend(handles=handles, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.03))
    axes[-1].xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%d %b"))
    fig.tight_layout()
    save_fig(fig, "fig03_anomaly_examples.png")


def fig_smoothing():
    win = (IDX >= "2014-02-10") & (IDX < "2014-02-15")
    y = df["pm25"].ffill().values
    fig, ax = plt.subplots(figsize=(7.4, 3.0))
    ax.plot(IDX[win], y[win], color=INK, lw=1.0, label="Observed", marker="o", ms=1.8)
    for meth, col in zip(["ma", "es", "sg"], [PAL[0], PAL[1], PAL[2]]):
        ax.plot(IDX[win], smooth(y, meth)[win], color=col, lw=1.3, label=METHOD_LABELS[("smoothing", meth)])
    ax.set_ylabel("PM2.5 (µg/m³)")
    title(ax, "Causal smoothers tuned to the same white-noise attenuation (variance factor 1/3)", pad=4)
    ax.legend(ncol=4, loc="upper left")
    ax.xaxis.set_major_locator(matplotlib.dates.DayLocator())
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%d %b"))
    save_fig(fig, "fig04_smoothing_example.png")


def _fmt(v):
    if abs(v) >= 99.5:
        return f"{v:+.0f}"
    if abs(v) < 0.95:
        return f"{v:+.2f}"
    return f"{v:+.1f}"


def fig_heatmap():
    var = [(s, m) for s in STAGES for m in STAGES[s] if m != REFERENCE[STAGE_KEY.get(s, s)]]
    var = [(s, m) for s, m in var if (f"{s}:{m}" in set(res.config))]
    models = [m for m in order_models if m in set(res.model)]
    M = np.full((len(models), len(var)), np.nan)
    S = np.zeros_like(M, dtype=bool)
    for i, mo in enumerate(models):
        for j, (s, m) in enumerate(var):
            r = res[(res.config == f"{s}:{m}") & (res.model == mo)]
            if len(r):
                M[i, j] = r["MAE_change_pct"].iloc[0]
                S[i, j] = (r["p_holm"].iloc[0] < 0.05) and abs(M[i, j]) >= 0.1   # ties below 0.1% not flagged
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    lim = 20
    ax.imshow(np.ma.masked_invalid(np.clip(M, -lim, lim)), cmap=DIVERGING, norm=TwoSlopeNorm(0, -lim, lim),
              aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if np.isfinite(M[i, j]):
                v = M[i, j]
                ax.text(j, i, _fmt(v) + ("*" if S[i, j] else ""), ha="center", va="center", fontsize=5.9,
                        color="white" if abs(v) > 12 else INK)
            else:
                ax.text(j, i, "n/a", ha="center", va="center", fontsize=5.9, color=MUTED)
    ax.set_xticks(range(len(var)))
    ax.set_xticklabels([METHOD_LABELS[v] for v in var], rotation=40, ha="right")
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models)
    ax.grid(False)
    start = 0
    for s in STAGES:
        n = sum(1 for v in var if v[0] == s)
        if not n:
            continue
        a, b = start, start + n - 1
        start += n
        lab = {"imputation": "Imputation", "anomaly": "Anomalies", "encoding": "Encoding",
               "normalization": "Normalization", "smoothing": "Smoothing", "smoothing_in": "Smoothing\n(inputs only)"}[s]
        ax.text((a + b) / 2, -0.62, lab, ha="center", va="bottom", fontsize=6.8, weight="bold", color=INK)
        if b < len(var) - 1:
            ax.axvline(b + 0.5, color="white", lw=2.5)
    for k in [len(ML_MODELS), len(ML_MODELS) + len(DL_MODELS)]:
        if k < len(models):
            ax.axhline(k - 0.5, color="white", lw=2.5)
    sm = plt.cm.ScalarMappable(cmap=DIVERGING, norm=TwoSlopeNorm(0, -lim, lim))
    cb = fig.colorbar(sm, ax=ax, fraction=0.025, pad=0.015)
    cb.set_label("Change in MAE vs. reference (%)", fontsize=7.2)
    cb.outline.set_visible(False)
    if FIG_TITLES:
        fig.suptitle("Change in test MAE relative to the reference pipeline "
                     "(* Holm-adjusted DM p < 0.05 and |change| >= 0.1%; colours capped at ±20%)", fontsize=8.2,
                     weight="bold", x=0.02, ha="left", y=1.03)
    save_fig(fig, "fig05_mae_change_heatmap.png")


def fig_stage_range():
    models = [m for m in order_models if m in set(res.model)]
    stages = list(STAGES)
    R = np.full((len(models), len(stages)), np.nan)
    for i, mo in enumerate(models):
        refm = ref.loc[mo, "MAE"]
        for j, s in enumerate(stages):
            v = stage_frame(s)
            v = v[v.model == mo]["MAE"]
            if len(v) > 1:
                R[i, j] = 100 * (v.max() - v.min()) / refm
    fig, ax = plt.subplots(figsize=(5.6, 4.0))
    vmax = np.nanpercentile(R, 90)
    ax.imshow(np.ma.masked_invalid(np.clip(R, 0, vmax)), cmap=SEQUENTIAL, vmin=0, vmax=vmax, aspect="auto")
    for i in range(R.shape[0]):
        for j in range(R.shape[1]):
            if np.isfinite(R[i, j]):
                ax.text(j, i, f"{R[i, j]:.1f}" if R[i, j] < 99.5 else f"{R[i, j]:.0f}", ha="center",
                        va="center", fontsize=6.6, color="white" if R[i, j] > 0.6 * vmax else INK)
            else:
                ax.text(j, i, "n/a", ha="center", va="center", fontsize=6.3, color=MUTED)
    lab = {"imputation": "Imputation", "anomaly": "Anomalies", "encoding": "Encoding",
           "normalization": "Normalization", "smoothing": "Smoothing", "smoothing_in": "Smoothing\n(inputs)"}
    ax.set_xticks(range(len(stages)))
    ax.set_xticklabels([lab[s] for s in stages], rotation=0, fontsize=6.8)
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models)
    ax.grid(False)
    title(ax, "Spread of test MAE across the methods of each stage\n(max minus min, % of reference MAE)", pad=6)
    save_fig(fig, "fig06_stage_sensitivity.png")


def fig_forecasts():
    fam_best = {}
    r0 = res[res.config == "reference"]
    for fam in ["ML", "DL", "Statistical"]:
        rr = r0[r0.family == fam]
        if len(rr):
            fam_best[fam] = rr.sort_values("MAE").iloc[0]["model"]
    obs_s = pd.Series(Y_TEST, index=IDX[TE_IDX])
    roll = obs_s.rolling(168).std()
    end = roll.idxmax()
    win = (IDX[TE_IDX] > end - pd.Timedelta(hours=168)) & (IDX[TE_IDX] <= end)
    fig, ax = plt.subplots(figsize=(7.4, 3.1))
    ax.plot(IDX[TE_IDX][win], Y_TEST[win], color=INK, lw=1.3, label="Observed")
    for (fam, mo), col in zip(fam_best.items(), [PAL[0], PAL[1], PAL[2]]):
        ax.plot(IDX[TE_IDX][win], preds[("reference", mo)][win], color=col, lw=1.0, label=f"{mo} ({fam})")
    ax.set_ylabel("PM2.5 (µg/m³)")
    title(ax, "One-hour-ahead forecasts in the most volatile test week (reference pipeline)", pad=4)
    ax.legend(ncol=4, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%d %b"))
    save_fig(fig, "fig07_forecast_week.png")


def fig_apparent():
    var = [("anomaly", m) for m in ["zscore", "iqr", "iforest"]] + [("smoothing", m) for m in ["ma", "es", "sg"]]
    var = [v for v in var if f"{v[0]}:{v[1]}" in set(res.config)]
    if not var:
        return
    raw_c, app_c = [], []
    for s, m in var:
        r = res[res.config == f"{s}:{m}"].set_index("model")
        base = ref.loc[r.index, "MAE"]
        raw_c.append(np.median(100 * (r["MAE"] / base - 1)))
        app_c.append(np.median(100 * (r["MAE_apparent"] / base - 1)))
    y = np.arange(len(var))
    fig, ax = plt.subplots(figsize=(6.4, 3.3))
    ax.barh(y + 0.19, app_c, height=0.36, color=PAL[0], label="Scored against the preprocessed series")
    ax.barh(y - 0.19, raw_c, height=0.36, color=PAL[1], label="Scored against the raw observations")
    for yy, a, b in zip(y, app_c, raw_c):
        ax.text(a + (0.6 if a >= 0 else -0.6), yy + 0.19, f"{a:+.1f}%", va="center",
                ha="left" if a >= 0 else "right", fontsize=6.8, color=INK2)
        ax.text(b + (0.6 if b >= 0 else -0.6), yy - 0.19, f"{b:+.1f}%", va="center",
                ha="left" if b >= 0 else "right", fontsize=6.8, color=INK2)
    ax.axvline(0, color=AXIS, lw=0.9)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{'Anomaly' if s == 'anomaly' else 'Smoothing'}: {METHOD_LABELS[(s, m)]}" for s, m in var])
    ax.invert_yaxis()
    ax.set_xlabel("Median change in MAE across models vs. reference (%)")
    title(ax, "Apparent versus real effect of anomaly removal and smoothing", pad=4)
    ax.legend(loc="upper center", bbox_to_anchor=(0.45, -0.2), ncol=2)
    xl = ax.get_xlim()
    ax.set_xlim(min(xl[0], -3) - 4, xl[1] + 6)
    save_fig(fig, "fig08_apparent_vs_real.png")


def fig_sensitivity():
    if not sens_rows:
        return
    sr, ir = pd.DataFrame(sens_rows), pd.DataFrame(imp_rows)
    mods = [m for m in SENS_MODELS if m in set(sr.model)]
    fig, axes = plt.subplots(1, len(mods) + 1, figsize=(7.4, 2.8))
    cols = {"mean": PAL[1], "linear": PAL[0], "knn": PAL[2]}
    for imp in STAGES["imputation"]:
        d = ir[ir.imputation == imp].sort_values("rate")
        axes[0].plot(100 * d.rate, d.imputation_MAE, marker="o", ms=3.5, color=cols[imp], lw=1.3,
                     label=METHOD_LABELS[("imputation", imp)])
    axes[0].set_title("(a) Imputation error", pad=3, fontsize=8)
    axes[0].set_xticks([10, 20, 30])
    axes[0].set_ylabel("MAE (µg/m³)")
    axes[0].set_xlabel("Added gaps (%)")
    for k, mo in enumerate(mods, 1):
        for imp in STAGES["imputation"]:
            d = sr[(sr.model == mo) & (sr.imputation == imp)].sort_values("rate")
            axes[k].plot(100 * d.rate, d.MAE, marker="o", ms=3.5, color=cols[imp], lw=1.3)
        axes[k].set_title(f"({chr(97 + k)}) {mo}" + (" (one seed)" if mo in DL_MODELS else ""), pad=3, fontsize=8)
        axes[k].set_xticks([0, 10, 20, 30])
        axes[k].set_xlabel("Added gaps (%)")
    axes[1].set_ylabel("Test MAE (µg/m³)")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.08))
    fig.tight_layout(w_pad=0.6)
    save_fig(fig, "fig09_missing_sensitivity.png")


def fig_reference():
    t = tab_ref.copy()
    t = t.sort_values("MAE")
    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    ax.barh(range(len(t)), t["MAE"], color=[FAM_COL[f] for f in t["family"]], height=0.62)
    for i, (mo, r) in enumerate(t.iterrows()):
        ax.text(r["MAE"] + 0.1, i, f"{r['MAE']:.2f}", va="center", fontsize=6.8, color=INK2)
    ax.set_yticks(range(len(t)))
    ax.set_yticklabels(t.index)
    ax.invert_yaxis()
    ax.set_xlabel("Test MAE (µg/m³)")
    ax.set_xlim(0, t["MAE"].max() * 1.12)
    ax.legend(handles=[Patch(color=FAM_COL[f], label=f) for f in ["ML", "DL", "Statistical", "Baseline"]],
              loc="lower right")
    title(ax, "Accuracy under the reference pipeline", pad=4)
    save_fig(fig, "fig10_reference_mae.png")


for f in (fig_workflow, fig_data, fig_anomaly, fig_smoothing, fig_heatmap, fig_stage_range, fig_forecasts,
          fig_apparent, fig_sensitivity, fig_reference):
    try:
        f()
    except Exception as e:
        log(f"figure {f.__name__} failed: {e}")

# -----------------------------------------------------------------------------
# 10. Run information and download
# -----------------------------------------------------------------------------
info = {"minutes": round((time.time() - T0) / 60, 1), "python": platform.python_version(),
        "numpy": np.__version__, "pandas": pd.__version__,
        "sklearn": __import__("sklearn").__version__,
        "xgboost": __import__("xgboost").__version__ if XGBRegressor else None,
        "statsmodels": __import__("statsmodels").__version__ if SARIMAX else None,
        "prophet": __import__("prophet").__version__ if Prophet else None,
        "tensorflow": tf.__version__ if tf else None, "gpu": [g.name for g in gpu],
        "arima_order": arima_order, "adf_stat_2013": float(adf_stat), "adf_p_2013": float(adf_p),
        "sg_noise_factor": float(SG_COEF @ SG_COEF), "configs": [c[0] + ":" + c[1] for c in CONFIGS],
        "persistence": persist_row, "fast_mode": FAST}
with open(os.path.join(OUT, "run_info.json"), "w") as fh:
    json.dump(info, fh, indent=2, default=str)
print(tab_ref.round(2).to_string())
print("\n=== RESULTS (config, model, MAE, RMSE, MAPE, MAE_apparent, p_holm) ===")
print(res[["config", "model", "MAE", "RMSE", "MAPE", "MAE_apparent", "p_holm"]].round(4).to_csv(index=False))
zip_path = "pm25_results.zip"
with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
    for fn in sorted(os.listdir(OUT)):
        zf.write(os.path.join(OUT, fn), arcname=os.path.join(OUT, fn))
log(f"finished in {info['minutes']} min -> {zip_path}")
if not FAST:
    try:
        from google.colab import files
        files.download(zip_path)
    except Exception:
        pass
