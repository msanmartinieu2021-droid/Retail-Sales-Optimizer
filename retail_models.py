import re
import numpy as np
import pandas as pd
from dataclasses import dataclass

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from lightgbm import LGBMRegressor

from scipy.sparse import coo_matrix
from implicit.als import AlternatingLeastSquares


# =========================================================
# Utilities
# =========================================================
def _ohe_dense():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)

def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(s).strip().lower())

def _first_existing(df: pd.DataFrame, candidates: list[str]) -> str | None:
    cols = {_norm(c): c for c in df.columns}
    for cand in candidates:
        key = _norm(cand)
        if key in cols:
            return cols[key]
    return None

def _find_by_keywords(df: pd.DataFrame, keywords: list[str]) -> str | None:
    ncols = [(c, _norm(c)) for c in df.columns]
    for kw in keywords:
        nkw = _norm(kw)
        for c, nc in ncols:
            if nkw in nc:
                return c
    return None

def _coerce_bool01(x: pd.Series) -> pd.Series:
    if x.dtype == bool:
        return x.astype(int)
    s = x.astype(str).str.lower().str.strip()
    return s.isin(["1", "true", "t", "yes", "y"]).astype(int)

def _safe_num(x, default=0.0) -> pd.Series:
    if isinstance(x, pd.Series):
        return pd.to_numeric(x, errors="coerce").fillna(default)
    if isinstance(x, pd.DataFrame):
        return pd.to_numeric(x.iloc[:, 0], errors="coerce").fillna(default)
    if x is None or isinstance(x, (int, float, np.number)):
        return pd.Series([default if x is None else float(x)])
    return pd.to_numeric(pd.Series(x), errors="coerce").fillna(default)

def _clip01(x: pd.Series) -> pd.Series:
    return x.clip(lower=0.0, upper=1.0)


# =========================================================
# Generalizable Schema (auto-detect)
# =========================================================
@dataclass
class RetailSchema:
    order_id: str = "order_id"
    customer_id: str = "customer_id"
    date: str = "date"
    product_category: str = "product_category"

    unit_price: str = "unit_price"
    quantity: str = "quantity"
    total_amount: str = "total_amount"

    discount_amount: str = "discount_amount"
    discount_percent: str = "discount_percent"

    city: str = "city"
    gender: str = "gender"
    age: str = "age"
    returning: str = "is_returning_customer"
    payment_method: str = "payment_method"
    device_type: str = "device_type"
    session_minutes: str = "session_minutes"
    pages_viewed: str = "pages_viewed"
    delivery_days: str = "delivery_time_days"
    rating: str = "customer_rating"


def infer_schema(df_raw: pd.DataFrame) -> dict:
    df = df_raw

    customer = (
        _first_existing(df, ["customer_id", "Customer_ID", "cust_id", "user_id", "client_id"])
        or _find_by_keywords(df, ["customer", "cust", "user", "client"])
    )

    category = (
        _first_existing(df, ["product_category", "Product_Category", "category", "product_type", "item_category"])
        or _find_by_keywords(df, ["category", "productcat", "product_category", "itemcat"])
    )

    order_id = (
        _first_existing(df, ["order_id", "Order_ID", "transaction_id", "invoice_id", "receipt_id"])
        or _find_by_keywords(df, ["order", "transaction", "invoice"])
    )

    date = (
        _first_existing(df, ["date", "Date", "order_date", "timestamp", "datetime"])
        or _find_by_keywords(df, ["date", "time", "timestamp"])
    )

    unit_price = (
        _first_existing(df, ["unit_price", "Unit_Price", "price", "item_price", "unitprice"])
        or _find_by_keywords(df, ["unitprice", "unit_price", "price"])
    )

    quantity = (
        _first_existing(df, ["quantity", "Quantity", "qty", "units"])
        or _find_by_keywords(df, ["quantity", "qty", "units"])
    )

    total_amount = (
        _first_existing(df, ["total_amount", "Total_Amount", "amount", "sales", "revenue", "total"])
        or _find_by_keywords(df, ["totalamount", "total_amount", "revenue", "sales", "amount", "total"])
    )

    discount_amount = (
        _first_existing(df, ["discount_amount", "Discount_Amount", "discount", "promo_amount"])
        or _find_by_keywords(df, ["discountamount", "discount_amount", "discount"])
    )

    city = _first_existing(df, ["city", "City", "region", "state"]) or _find_by_keywords(df, ["city", "region", "state"])
    gender = _first_existing(df, ["gender", "Gender", "sex"]) or _find_by_keywords(df, ["gender", "sex"])
    age = _first_existing(df, ["age", "Age"]) or _find_by_keywords(df, ["age"])
    returning = _first_existing(df, ["is_returning_customer", "Is_Returning_Customer", "returning", "repeat"]) or _find_by_keywords(df, ["return"])
    payment_method = _first_existing(df, ["payment_method", "Payment_Method", "payment"]) or _find_by_keywords(df, ["payment"])
    device_type = _first_existing(df, ["device_type", "Device_Type", "device"]) or _find_by_keywords(df, ["device"])
    session_minutes = _first_existing(df, ["session_duration_minutes", "Session_Duration_Minutes", "session_minutes"]) or _find_by_keywords(df, ["session", "duration"])
    pages_viewed = _first_existing(df, ["pages_viewed", "Pages_Viewed", "pageviews"]) or _find_by_keywords(df, ["pages", "views"])
    delivery_days = _first_existing(df, ["delivery_time_days", "Delivery_Time_Days", "delivery_days"]) or _find_by_keywords(df, ["delivery"])
    rating = _first_existing(df, ["customer_rating", "Customer_Rating", "rating", "review_score"]) or _find_by_keywords(df, ["rating", "review"])

    return {
        "customer_id": customer,
        "product_category": category,
        "order_id": order_id,
        "date": date,
        "unit_price": unit_price,
        "quantity": quantity,
        "total_amount": total_amount,
        "discount_amount": discount_amount,
        "city": city,
        "gender": gender,
        "age": age,
        "returning": returning,
        "payment_method": payment_method,
        "device_type": device_type,
        "session_minutes": session_minutes,
        "pages_viewed": pages_viewed,
        "delivery_days": delivery_days,
        "rating": rating,
    }


# =========================================================
# Realistic Syntethic Discount
# =========================================================
def simulate_discount_percent(df: pd.DataFrame, s: RetailSchema, seed: int = 42) -> pd.Series:
    rng = np.random.default_rng(seed)
    n = len(df)

    cat = df[s.product_category].astype(str).str.lower()
    promo_heavy = ["fashion", "apparel", "clothing", "beauty", "cosmetic", "home", "garden"]
    cat_bonus = cat.apply(lambda x: 0.20 if any(k in x for k in promo_heavy) else 0.0).values

    unit_price = _safe_num(df[s.unit_price], 0.0).values
    price_score = np.clip(np.log1p(unit_price) / 8.0, 0, 1)

    dow = _safe_num(df.get("dow", pd.Series(np.zeros(n))), 0).values
    is_weekend = np.isin(dow, [5, 6]).astype(float)

    month = _safe_num(df.get("month", pd.Series(np.ones(n))), 1).values
    campaign = np.isin(month, [11, 12, 1]).astype(float)

    eng_series = df["engagement_score"] if "engagement_score" in df.columns else pd.Series(np.zeros(n), index=df.index)
    eng = _safe_num(eng_series, 0.0).values

    denom = (np.percentile(eng, 90) + 1e-9) if n else 1.0
    eng_score = np.clip(eng / denom, 0, 1)

    z = (
        -1.1
        + 0.9 * price_score
        + 0.8 * is_weekend
        + 0.9 * campaign
        + cat_bonus
        - 0.6 * eng_score
        + rng.normal(0, 0.3, size=n)
    )
    p = 1.0 / (1.0 + np.exp(-z))
    has_disc = rng.uniform(0, 1, size=n) < p

    choices = np.array([0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40])
    probs = np.array([0.18, 0.22, 0.18, 0.16, 0.14, 0.10, 0.02])
    disc_vals = rng.choice(choices, size=n, p=probs)

    out = np.where(has_disc, disc_vals, 0.0)
    out = out + rng.normal(0, 0.01, size=n)
    out = np.clip(out, 0.0, 0.50)
    return pd.Series(out, index=df.index)


# =========================================================
# Parse & Clean
# =========================================================
def parse_and_clean_general(df_raw: pd.DataFrame, seed: int = 42):
    warnings = []
    df = df_raw.copy()
    mapping = infer_schema(df)

    if mapping["customer_id"] is None:
        raise ValueError("I can't find a Customer ID Column. You need customer_id / user_id / client_id.")
    if mapping["product_category"] is None:
        raise ValueError("I can't find a category/product column. You need product_category / category.")

    s = RetailSchema()

    rename_map = {}
    for canon, src in mapping.items():
        if src is not None:
            canon_name = getattr(s, canon) if hasattr(s, canon) else canon
            rename_map[src] = canon_name
    df = df.rename(columns=rename_map)

    df[s.customer_id] = df[s.customer_id].astype(str)
    df[s.product_category] = df[s.product_category].astype(str)

    if s.order_id not in df.columns:
        df[s.order_id] = np.arange(len(df)).astype(str)
        warnings.append("No order_id column; artificial column has been created.")

    if s.date in df.columns:
        df[s.date] = pd.to_datetime(df[s.date], errors="coerce")
    else:
        df[s.date] = pd.NaT
        warnings.append("No date columns; some temporal features will be 0/NaN.")

    if s.quantity not in df.columns:
        df[s.quantity] = 1.0
        warnings.append("No quantity column; quantity=1 was assumed per row.")
    df[s.quantity] = _safe_num(df[s.quantity], 1.0).clip(lower=0)

    if s.total_amount in df.columns:
        df[s.total_amount] = _safe_num(df[s.total_amount], 0.0).clip(lower=0)
    else:
        df[s.total_amount] = np.nan

    if s.unit_price in df.columns:
        df[s.unit_price] = _safe_num(df[s.unit_price], np.nan).clip(lower=0)
    else:
        df[s.unit_price] = np.nan

    if df[s.total_amount].isna().all() and not df[s.unit_price].isna().all():
        df[s.total_amount] = (df[s.unit_price] * df[s.quantity]).astype(float)
        warnings.append("No total_amount amount; calculated as unit_price * quantity.")
    if df[s.unit_price].isna().all() and not df[s.total_amount].isna().all():
        denom = df[s.quantity].replace(0, np.nan)
        df[s.unit_price] = (df[s.total_amount] / denom).fillna(df[s.total_amount].median())
        warnings.append("No unit_price column; estimated as total_amount / quantity.")

    if df[s.unit_price].isna().all():
        df[s.unit_price] = 1.0
        warnings.append("No unit_price column and could not calculate it; unit_price=1 was set.")
    if df[s.total_amount].isna().all():
        df[s.total_amount] = (df[s.unit_price] * df[s.quantity]).astype(float)
        warnings.append("No total_amount column and could not obtain it; calculated as unit_price * quantity.")

    if df[s.date].notna().any():
        df["dow"] = df[s.date].dt.dayofweek.fillna(0).astype(int)
        df["month"] = df[s.date].dt.month.fillna(1).astype(int)
        df["is_weekend"] = df["dow"].isin([5, 6]).astype(int)
    else:
        df["dow"] = 0
        df["month"] = 1
        df["is_weekend"] = 0

    if s.session_minutes in df.columns:
        df[s.session_minutes] = _safe_num(df[s.session_minutes], 0.0).clip(lower=0)
    else:
        df[s.session_minutes] = 0.0
    if s.pages_viewed in df.columns:
        df[s.pages_viewed] = _safe_num(df[s.pages_viewed], 0.0).clip(lower=0)
    else:
        df[s.pages_viewed] = 0.0
    df["engagement_score"] = 0.6 * np.log1p(df[s.session_minutes]) + 0.4 * np.log1p(df[s.pages_viewed])

    df["discount_is_synthetic"] = 0
    if s.discount_amount in df.columns:
        df[s.discount_amount] = _safe_num(df[s.discount_amount], 0.0).clip(lower=0)
        base = (df[s.unit_price] * df[s.quantity]).replace(0, np.nan)
        df[s.discount_percent] = (df[s.discount_amount] / base).fillna(0.0)
        df[s.discount_percent] = _clip01(df[s.discount_percent]).clip(upper=0.95)
    else:
        df[s.discount_percent] = simulate_discount_percent(df, s, seed=seed)
        df["discount_is_synthetic"] = 1
        warnings.append("There was no discount; synthetic discount_percent was generated(solo demo).")

    df[s.discount_percent] = _safe_num(df[s.discount_percent], 0.0).clip(lower=0.0, upper=0.95)

    df["order_prediscount"] = (df[s.unit_price] * df[s.quantity]).astype(float)
    df["value_lost_discount"] = df["order_prediscount"] * df[s.discount_percent]
    df["net_revenue"] = (df["order_prediscount"] * (1.0 - df[s.discount_percent])).astype(float)

    for c in ["order_prediscount", "net_revenue", "value_lost_discount"]:
        df[f"log_{c}"] = np.log1p(_safe_num(df[c], 0.0).clip(lower=0))

    med_eng = float(np.nanmedian(df["engagement_score"].values)) if len(df) else 0.0
    df["high_engagement_flag"] = (df["engagement_score"] > med_eng).astype(int)
    denom = df[s.session_minutes].replace(0, np.nan)
    df["pages_per_minute"] = (df[s.pages_viewed] / denom).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    if s.delivery_days in df.columns:
        df[s.delivery_days] = _safe_num(df[s.delivery_days], 0.0).clip(lower=0)
        med_del = float(np.nanmedian(df[s.delivery_days].values)) if len(df) else 0.0
        df["delivery_delay_index"] = (df[s.delivery_days] - med_del).fillna(0.0)
        q25 = float(np.nanquantile(df[s.delivery_days].values, 0.25)) if len(df) else 0.0
        df["fast_delivery_flag"] = (df[s.delivery_days] <= q25).astype(int)
    else:
        df[s.delivery_days] = 0.0
        df["delivery_delay_index"] = 0.0
        df["fast_delivery_flag"] = 0

    if s.rating in df.columns:
        df[s.rating] = _safe_num(df[s.rating], 0.0)
        df["low_rating_risk"] = (df[s.rating] <= 2).astype(int)
    else:
        df[s.rating] = 0.0
        df["low_rating_risk"] = 0

    if s.returning in df.columns:
        df[s.returning] = _coerce_bool01(df[s.returning])
    else:
        df[s.returning] = 0

    if s.age in df.columns:
        df[s.age] = _safe_num(df[s.age], np.nan)
    else:
        df[s.age] = np.nan

    df = df.dropna(subset=[s.customer_id, s.product_category]).reset_index(drop=True)
    return df, s, mapping, warnings


# =========================================================
# Customer features
# =========================================================
def build_customer_features(df: pd.DataFrame, s: RetailSchema) -> pd.DataFrame:
    d = df.copy()
    cat_counts = d.groupby([s.customer_id, s.product_category]).size().reset_index(name="n")
    cat_div = cat_counts.groupby(s.customer_id)[s.product_category].nunique().rename("category_diversity")
    top_share = cat_counts.groupby(s.customer_id)["n"].apply(lambda x: float((x / x.sum()).max())).rename("top_category_share")

    ref_date = d[s.date].max() if d[s.date].notna().any() else pd.Timestamp("1970-01-01")

    agg = d.groupby(s.customer_id).agg(
        orders=(s.order_id, "nunique"),
        total_spend=("net_revenue", "sum"),
        avg_basket=("net_revenue", "mean"),
        total_qty=(s.quantity, "sum"),
        avg_discount_percent=(s.discount_percent, "mean"),
        avg_engagement=("engagement_score", "mean"),
        avg_delivery_days=(s.delivery_days, "mean"),
        avg_rating=(s.rating, "mean"),
        last_purchase=(s.date, "max"),
        first_purchase=(s.date, "min"),
        returning=(s.returning, "max"),
    ).reset_index()

    if "last_purchase" in agg:
        agg["recency_days"] = (ref_date - agg["last_purchase"]).dt.days
    else:
        agg["recency_days"] = 0

    agg["tenure_days"] = (agg["last_purchase"] - agg["first_purchase"]).dt.days if "first_purchase" in agg else 0
    agg["orders_per_month"] = agg["orders"] / (agg["tenure_days"].clip(lower=1) / 30.0)

    agg = agg.merge(cat_div, on=s.customer_id, how="left").merge(top_share, on=s.customer_id, how="left")
    agg["category_diversity"] = agg["category_diversity"].fillna(1)
    agg["top_category_share"] = agg["top_category_share"].fillna(1.0)

    for c in ["total_spend", "avg_basket", "total_qty", "recency_days", "orders"]:
        agg[f"log_{c}"] = np.log1p(_safe_num(agg[c], 0.0).clip(lower=0))

    return agg.drop(columns=["last_purchase", "first_purchase"], errors="ignore")


def make_customer_prepipe(cust_feat: pd.DataFrame, id_col: str) -> ColumnTransformer:
    cols = [c for c in cust_feat.columns if c != id_col]
    numeric = [c for c in cols if pd.api.types.is_numeric_dtype(cust_feat[c])]
    categorical = [c for c in cols if c not in numeric]

    num_pipe = Pipeline([("imp", SimpleImputer(strategy="median")), ("scaler", RobustScaler())])
    cat_pipe = Pipeline([("imp", SimpleImputer(strategy="most_frequent")), ("ohe", _ohe_dense())])

    return ColumnTransformer(
        [("num", num_pipe, numeric), ("cat", cat_pipe, categorical)],
        remainder="drop",
    )


# =========================================================
# Segmentation + balance
# =========================================================
def _balanced_assign_from_centers(X: np.ndarray, centers: np.ndarray) -> np.ndarray:
    X = np.asarray(X)
    centers = np.asarray(centers)
    n = X.shape[0]
    K = centers.shape[0]

    # defensiva: si K==0 o n==0 o K >= n (no tiene sentido hacer balance estricto)
    if n == 0 or K == 0 or K >= n:
        return np.zeros(n, dtype=int) if K > 0 else np.zeros(n, dtype=int)

    # cuota por cluster (distribución lo más uniforme posible)
    base = n // K
    rem = n - base * K
    quota = np.array([base + (1 if i < rem else 0) for i in range(K)], dtype=int)

    # distancia cuadrada (vectorizada)
    x2 = (X * X).sum(axis=1, keepdims=True)
    c2 = (centers * centers).sum(axis=1, keepdims=True).T
    d2 = x2 + c2 - 2.0 * (X @ centers.T)

    # preferencias ordenadas por distancia (cluster más cercano primero)
    prefs = np.argsort(d2, axis=1)

    # margen entre el mejor y el segundo (para priorizar asignaciones "fuertes")
    best = d2[np.arange(n), prefs[:, 0]]
    second = d2[np.arange(n), prefs[:, 1]] if K > 1 else best + 1.0
    margin = second - best
    order = np.argsort(-margin)

    assigned = -np.ones(n, dtype=int)
    remaining = quota.copy()

    # asignar respetando cuotas
    for idx in order:
        for k in prefs[idx]:
            if remaining[k] > 0:
                assigned[idx] = int(k)
                remaining[k] -= 1
                break

    # si quedan por asignar (raro), repartir cíclicamente
    if (assigned == -1).any():
        leftovers = np.where(assigned == -1)[0]
        for i, idx in enumerate(leftovers):
            assigned[idx] = int(i % K)

    return assigned

def train_segmentation(X: np.ndarray, k_min=3, k_max=10, pca_var=0.90, min_prop=0.08, random_state=42):
    pca = PCA(n_components=pca_var, random_state=random_state)
    Xr = pca.fit_transform(X)

    best_score = -1e9
    best_model = None
    best_labels = None
    best_k = None
    best_metrics = None

    for k in range(k_min, k_max + 1):
        km = KMeans(n_clusters=k, n_init=25, random_state=random_state)
        labels = km.fit_predict(Xr)

        counts = np.bincount(labels, minlength=k)
        props = counts / counts.sum()
        min_cluster_prop = float(props.min())

        try:
            sil = float(silhouette_score(Xr, labels))
        except Exception:
            sil = -1.0

        balance_penalty = min(1.0, float(min_cluster_prop / max(min_prop, 1e-9)))
        composite = sil * balance_penalty

        if composite > best_score:
            best_score = composite
            best_model = km
            best_labels = labels
            best_k = k
            best_metrics = {
                "k": int(best_k),
                "silhouette": float(sil),
                "min_segment_prop": float(min_cluster_prop),
                "balance_penalty": float(balance_penalty),
            }

    return pca, best_model, best_labels, best_metrics


def balance_labels_if_requested(X: np.ndarray, pca: PCA, seg_model: KMeans, labels: np.ndarray, do_balance: bool) -> np.ndarray:
    # si no piden balance o no hay modelo, devolver labels originales
    if not do_balance:
        return labels
    if seg_model is None or not hasattr(seg_model, "cluster_centers_"):
        return labels

    # asegurarnos que X es array y tiene filas
    X = np.asarray(X)
    if X.shape[0] == 0:
        return labels

    # pca defensiva: si pca es None o no está ajustado, no transformamos
    try:
        Xr = pca.transform(X) if pca is not None else X
    except Exception:
        return labels

    centers = seg_model.cluster_centers_
    # si las dimensiones no casan, evitamos el rebalanceo por seguridad
    if Xr.shape[1] != centers.shape[1]:
        return labels

    # si K es demasiado grande en relación a n, no balanceamos (muy costoso o ilógico)
    n = Xr.shape[0]
    K = centers.shape[0]
    if K <= 0 or K >= n:
        return labels

    # límite de seguridad: si n*K es excesivo, saltar rebalanceo (evita cuelgues)
    if n * K > 10_000_000:  # umbral seguro; ajustable
        return labels

    try:
        new_labels = _balanced_assign_from_centers(Xr, centers)
        return np.asarray(new_labels).astype(int)
    except Exception:
        return labels


# =========================================================
# Optimal discount per segment
# =========================================================
def build_demand_table(df: pd.DataFrame, s: RetailSchema):
    d = df.copy()
    y = np.log1p(_safe_num(d[s.quantity], 0.0).clip(lower=0)).astype(float)

    feats = pd.DataFrame({
        "product_category": d[s.product_category].astype(str),
        "unit_price": _safe_num(d[s.unit_price], 0.0).astype(float),
        "discount_percent": _safe_num(d[s.discount_percent], 0.0).astype(float),
        "is_weekend": _safe_num(d.get("is_weekend", 0), 0).astype(int),
        "month": _safe_num(d.get("month", 1), 1).astype(int),
        "engagement_score": _safe_num(d.get("engagement_score", pd.Series(np.zeros(len(d)))), 0.0).values,
        "delivery_days": _safe_num(d.get(s.delivery_days, 0.0), 0.0).astype(float),
        "rating": _safe_num(d.get(s.rating, 0.0), 0.0).astype(float),
        "returning": _safe_num(d.get(s.returning, 0), 0).astype(int),
    })

    for opt in [s.city, s.gender, s.payment_method, s.device_type]:
        if opt in d.columns:
            feats[opt] = d[opt].astype(str)
    for optn in [s.age, s.session_minutes, s.pages_viewed]:
        if optn in d.columns:
            feats[optn] = _safe_num(d[optn], np.nan)

    return feats, y


def train_demand_model(feats: pd.DataFrame, y: pd.Series, random_state=42) -> Pipeline:
    cat_cols = [c for c in feats.columns if feats[c].dtype == "object"]
    num_cols = [c for c in feats.columns if c not in cat_cols]

    pre = ColumnTransformer(
        [
            ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("scaler", RobustScaler())]), num_cols),
            ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")), ("ohe", _ohe_dense())]), cat_cols),
        ],
        remainder="drop",
    )

    model = LGBMRegressor(
        n_estimators=450,
        learning_rate=0.05,
        num_leaves=63,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=random_state,
        n_jobs=-1,
    )

    pipe = Pipeline([("pre", pre), ("model", model)])
    pipe.fit(feats, y)
    return pipe


def recommend_discount_by_segment(
    df: pd.DataFrame,
    cust_segments: pd.DataFrame,
    s: RetailSchema,
    candidate=(0.0, 0.05, 0.10, 0.15, 0.20, 0.25),
    min_rows_segment: int = 120,
    max_rows_per_segment: int = 4000,
    max_segments_to_process: int = 50,   # <-- límite de seguridad
):
    d = df.merge(cust_segments[[s.customer_id, "segment"]], on=s.customer_id, how="left").copy()

    out = []
    has_synth = ("discount_is_synthetic" in d.columns) and (int(d["discount_is_synthetic"].max()) == 1)

    if has_synth:
        seg_stats = d.groupby("segment").agg(
            avg_price=(s.unit_price, "mean"),
            avg_qty=(s.quantity, "mean"),
            avg_spend=("net_revenue", "mean"),
            n=("segment", "size"),
        ).reset_index()
        seg_map = {int(r["segment"]): r for _, r in seg_stats.iterrows()}

    # lista de segmentos ordenada
    seg_list = sorted(d["segment"].dropna().unique())

    # si hay demasiados segmentos, procesamos sólo los más grandes (evita cuelgues)
    if len(seg_list) > int(max_segments_to_process):
        size_map = d.groupby("segment").size()
        top_segments = size_map.sort_values(ascending=False).head(int(max_segments_to_process)).index.tolist()
        seg_list = sorted([s for s in top_segments])

    for seg in seg_list:
        seg_df = d[d["segment"] == seg].copy()

        if len(seg_df) < int(min_rows_segment):
            continue

        if len(seg_df) > int(max_rows_per_segment):
            seg_df = seg_df.sample(n=int(max_rows_per_segment), random_state=42)

        price = _safe_num(seg_df[s.unit_price], 0.0).values
        price_mean = float(np.mean(price)) if len(price) else 0.0

        best_r, best_rev = None, -1e18

        if has_synth:
            rrow = seg_map.get(int(seg), None)
            avg_spend = float(rrow["avg_spend"]) if rrow is not None else 0.0
            avg_qty = float(rrow["avg_qty"]) if rrow is not None else 1.0

            beta = np.clip((price_mean / (avg_spend + 1e-6)) * 2.0 + 0.8, 0.6, 3.0)

            for r in candidate:
                q_pred = avg_qty * float(np.exp(beta * r))
                rev = price_mean * (1.0 - r) * q_pred
                if rev > best_rev:
                    best_rev, best_r = rev, r

            out.append({
                "segment": int(seg),
                "recommended_discount_percent": float(best_r),
                "expected_avg_revenue_per_order": float(best_rev),
                "n_orders_segment": int(len(seg_df)),
                "method": "synthetic_elasticity_prior",
            })

        else:
            feats, y = build_demand_table(seg_df, s)
            pipe = train_demand_model(feats, y)

            for r in candidate:
                feats_sim = feats.copy()
                feats_sim["discount_percent"] = float(r)
                logq = pipe.predict(feats_sim)
                q_pred = np.expm1(np.clip(logq, -5, 8))
                rev = float(np.mean(price * (1.0 - r) * q_pred))
                if rev > best_rev:
                    best_rev, best_r = rev, r

            out.append({
                "segment": int(seg),
                "recommended_discount_percent": float(best_r),
                "expected_avg_revenue_per_order": float(best_rev),
                "n_orders_segment": int(len(seg_df)),
                "method": "segment_demand_model",
            })

    return pd.DataFrame(out).sort_values("segment").reset_index(drop=True)

# =========================================================
# ALS
# =========================================================
def build_interaction_matrix(df: pd.DataFrame, s: RetailSchema, weight: str = "quantity"):
    d = df.copy()
    users = d[s.customer_id].astype("category")
    items = d[s.product_category].astype("category")

    if weight == "quantity" and s.quantity in d.columns:
        w = _safe_num(d[s.quantity], 1.0).clip(lower=1).astype(float)
    else:
        w = pd.Series(np.ones(len(d), dtype=float), index=d.index)

    user_index = {u: i for i, u in enumerate(users.cat.categories)}
    item_map = {i: it for i, it in enumerate(items.cat.categories)}

    rows = users.cat.codes.values
    cols = items.cat.codes.values
    mat = coo_matrix((w.values, (rows, cols))).tocsr()
    return mat, user_index, item_map


def train_als(mat, factors=96, iterations=25, regularization=0.10, random_state=42):
    als = AlternatingLeastSquares(
        factors=factors,
        iterations=iterations,
        regularization=regularization,
        random_state=random_state,
    )
    als.fit(mat)
    return als


def recommend_categories(customer_id: str, als, mat, user_index: dict, item_map: dict, topn=5):
    """
    FIX robusto para distintas versiones de implicit:
    - recs puede venir como lista de tuplas (item, score)
    - o como array Nx2
    - o como array Nx3+  -> nos quedamos con las 2 primeras columnas
    """
    if customer_id not in user_index:
        return []

    u = user_index[customer_id]
    recs = als.recommend(u, mat[u], N=topn)

    arr = np.asarray(recs)
    if arr.ndim == 1:
        # caso raro: lista de tuplas -> convertir a Nx2
        try:
            arr = np.array(list(recs), dtype=float)
        except Exception:
            return []

    if arr.ndim == 2 and arr.shape[1] >= 2:
        item_ids = arr[:, 0].astype(int)
        scores = arr[:, 1].astype(float)
    else:
        # fallback defensivo
        out = []
        for r in recs:
            if isinstance(r, (list, tuple)) and len(r) >= 2:
                out.append((r[0], float(r[1])))
        item_ids = [int(x[0]) for x in out]
        scores = [float(x[1]) for x in out]

    out = []
    for item_idx, score in zip(item_ids, scores):
        out.append((item_map.get(int(item_idx), f"item_{item_idx}"), float(score)))
    return out
