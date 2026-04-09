import streamlit as st
import pandas as pd
import numpy as np

from retail_models import (
    parse_and_clean_general,
    build_customer_features,
    make_customer_prepipe,
    train_segmentation,
    balance_labels_if_requested,
    recommend_discount_by_segment,
    build_interaction_matrix,
    train_als,
    recommend_categories,
)

# =========================
# Config
# =========================
st.set_page_config(page_title="Retail Sales Optimizer", layout="wide")

st.title("Retail Sales Optimizer")
st.caption(
    "Upload your dataset → column auto-detection → variables creation → segmentation → discount optimization → categories recommendation"
)

# =========================
# Session state
# =========================
if "disc_policy" not in st.session_state:
    st.session_state.disc_policy = None
if "disc_policy_meta" not in st.session_state:
    st.session_state.disc_policy_meta = None

if "als_bundle" not in st.session_state:
    st.session_state.als_bundle = None
if "als_meta" not in st.session_state:
    st.session_state.als_meta = None

# =========================
# Sidebar
# =========================
with st.sidebar:
    st.header("1) Dataset")
    uploaded = st.file_uploader("Upload your CSV", type=["csv"])
    default_path = "/content/ecommerce_customer_behavior_dataset.csv"
    use_default = st.checkbox("Use example dataset if no other csv is uploaded", value=True)
    st.divider()

    st.header("2) Segmentation")
    st.caption("More segments = more detail, but can be less stable.")
    k_min = st.slider("Min segments", 2, 10, 3)
    k_max = st.slider("Max segments", 3, 15, 10)
    pca_var = st.slider("Compression level (PCA)", 0.70, 0.99, 0.90)
    balance_segments = st.checkbox("Balance segments (similar sizes)", value=True)
    st.caption("ONLY balance the final assignation (do not change the model), to avoid tiny clusters.")
    st.divider()

    st.header("3) Optimal Discounts")
    candidate_pct = st.multiselect(
        "Discounts to test (%)",
        options=[0, 5, 10, 15, 20, 25, 30, 35, 40],
        default=[0, 5, 10, 15, 20, 25],
    )
    min_rows_segment = st.slider("Minimun rows per segment to estimate", 30, 400, 120, step=10)

    # ✅ NUEVO: límites anti-cuelgue (sin romper tu UI)
    st.caption("Anti-feeze (recommended if K is increased manually):")
    max_rows_per_segment = st.slider("Max rows per segment (simulation)", 500, 20000, 4000, step=500)
    global_train_max_rows = st.slider("Max rows to train global demand", 5000, 200000, 50000, step=5000)

    st.divider()

    st.header("4) Recomendations (ALS)")
    topn = st.slider("Nº of recommended categories", 3, 10, 5)
    factors = st.slider("Recommendator quality (factors)", 32, 192, 96, step=16)

# =========================
# Load dataset
# =========================
@st.cache_data(show_spinner=False)
def load_df(uploaded_file, use_default_path: bool, default_path: str):
    if uploaded_file is not None:
        return pd.read_csv(uploaded_file), "CSV uploaded"
    if use_default_path:
        return pd.read_csv(default_path), "Example dataset"
    return None, "none"

df_raw, src = load_df(uploaded, use_default, default_path)
if df_raw is None:
    st.info("Upload a dataset or click the example dataset.")
    st.stop()

st.subheader("1) Dataset")
st.caption(f"Fuente: **{src}**  |  Filas: **{len(df_raw):,}**  |  Columnas: **{len(df_raw.columns):,}**")
st.dataframe(df_raw.head(30), use_container_width=True)

# =========================
# Clean + schema inference
# =========================
@st.cache_data(show_spinner=False)
def clean_and_infer(df_in: pd.DataFrame):
    return parse_and_clean_general(df_in, seed=42)

with st.spinner("Auto-detecting columns + creating variables…"):
    df, s, mapping, warns = clean_and_infer(df_raw)

if warns:
    with st.expander("Warnings (per missing columns / inferences)"):
        for w in warns:
            st.warning(w)

with st.expander("Mapping detected (columna original → correct format)"):
    show_map = {k: v for k, v in mapping.items() if v is not None}
    st.json(show_map)

# =========================
# Feature preview
# =========================
st.subheader("Variables created (preview)")
cols_show = [
    s.customer_id, s.product_category, s.unit_price, s.quantity, s.total_amount,
    s.discount_percent, "discount_is_synthetic",
    "order_prediscount", "net_revenue", "value_lost_discount",
    "engagement_score", "high_engagement_flag",
    "delivery_delay_index", "fast_delivery_flag", "low_rating_risk",
    "dow", "month", "is_weekend",
]
cols_show = [c for c in cols_show if c in df.columns]
st.dataframe(df[cols_show].head(25), use_container_width=True)

if int(df.get("discount_is_synthetic", pd.Series([0])).max()) == 1:
    st.warning("⚠️ There was no discount; synthetic discount_percent was generated(solo demo)")

# =========================
# Segmentation
# =========================
st.subheader("2) Segmentation")

@st.cache_data(show_spinner=False)
def compute_customer_features_cached(df_in: pd.DataFrame):
    return build_customer_features(df_in, s)

@st.cache_resource(show_spinner=False)
def compute_segmentation_cached(cust_feat: pd.DataFrame, kmin: int, kmax: int, pvar: float):
    pre = make_customer_prepipe(cust_feat, id_col=s.customer_id)
    X = pre.fit_transform(cust_feat)
    pca, seg_model, labels, metrics = train_segmentation(X, k_min=kmin, k_max=kmax, pca_var=pvar)
    return pre, X, pca, seg_model, labels, metrics

with st.spinner("Building features per client + segmenting…"):
    cust_feat = compute_customer_features_cached(df)
    pre, X, pca, seg_model, labels, metrics = compute_segmentation_cached(cust_feat, k_min, k_max, pca_var)

labels = np.asarray(labels).astype(int)
labels = balance_labels_if_requested(X, pca, seg_model, labels, do_balance=balance_segments)

cust_feat = cust_feat.copy()
cust_feat["segment"] = labels.astype(int)

st.write(metrics)

seg_counts = cust_feat["segment"].value_counts().sort_index()
st.bar_chart(seg_counts)

st.caption("Summary per segment (to interprete who's who):")
seg_profile = (
    cust_feat.groupby("segment")
    .agg(
        clientes=(s.customer_id, "count"),
        gasto_total=("total_spend", "mean"),
        recencia_dias=("recency_days", "mean"),
        descuento_medio=("avg_discount_percent", "mean"),
        engagement=("avg_engagement", "mean"),
        entrega_media=("avg_delivery_days", "mean"),
        diversidad=("category_diversity", "mean"),
    )
    .reset_index()
    .sort_values("segment")
)
st.dataframe(seg_profile, use_container_width=True)

# =========================
# STEP 3 — Discount policy (button + meta invalidation)
# =========================
st.subheader("3) Optimal discount per segment")

candidate = tuple([x / 100.0 for x in sorted(candidate_pct)]) if candidate_pct else (0.0, 0.05, 0.10, 0.15, 0.20, 0.25)

meta_now = (src, int(k_min), int(k_max), float(pca_var), bool(balance_segments), tuple(candidate), int(min_rows_segment), int(max_rows_per_segment), int(global_train_max_rows))
if st.session_state.disc_policy_meta is not None and st.session_state.disc_policy_meta != meta_now:
    st.session_state.disc_policy = None
    st.session_state.disc_policy_meta = None

colA, colB = st.columns([1, 3])
with colA:
    run_discount = st.button("Calculate optimal disocunts", type="primary")
with colB:
    st.caption("Tip: adjust segmentation and then press the button. No recalculations when moving sliders.")

df_seg = df.merge(cust_feat[[s.customer_id, "segment"]], on=s.customer_id, how="left")
orders_per_segment = df_seg.groupby("segment").size().reset_index(name="n_orders_segment")

if run_discount:
    with st.spinner("Calculation optimal disocunts per segment…"):
        cust_seg = cust_feat[[s.customer_id, "segment"]].copy()
        cust_seg[s.customer_id] = cust_seg[s.customer_id].astype(str)
        cust_seg["segment"] = cust_seg["segment"].astype(int)

        pol = recommend_discount_by_segment(
            df=df,
            cust_segments=cust_seg,
            s=s,
            candidate=candidate,
            min_rows_segment=int(min_rows_segment),
            max_rows_per_segment=int(max_rows_per_segment),
        )
        st.session_state.disc_policy = pol
        st.session_state.disc_policy_meta = meta_now

disc_policy = st.session_state.disc_policy

all_segments = sorted(cust_feat["segment"].dropna().astype(int).unique().tolist())

if disc_policy is None:
    st.info("Press the button to generate the policy.")
else:
    base = pd.DataFrame({"segment": all_segments})
    base = base.merge(orders_per_segment, on="segment", how="left")
    base["n_orders_segment"] = base["n_orders_segment"].fillna(0).astype(int)

    disc_policy = disc_policy.copy()
    disc_policy["segment"] = disc_policy["segment"].astype(int)

    out = base.merge(disc_policy, on="segment", how="left")

    out["estado"] = "OK"
    out.loc[out["recommended_discount_percent"].isna(), "estado"] = "Not calculated (not enough data in the segment)"
    out["Descuento recomendado (%)"] = (out["recommended_discount_percent"] * 100).round(1)

    out = out.drop(columns=["recommended_discount_percent"], errors="ignore")
    out = out.sort_values("segment").reset_index(drop=True)

    st.dataframe(out, use_container_width=True)
    st.caption("If a segment appears as 'Not calculated', it is because there are not enough data in the segment. Lower K or decrease minimun per segment.")

# =========================
# STEP 4 — ALS (button)
# =========================
st.subheader("4) Category Recommendator(ALS)")

als_meta_now = (src, int(factors))
if st.session_state.als_meta is not None and st.session_state.als_meta != als_meta_now:
    st.session_state.als_bundle = None
    st.session_state.als_meta = None

colA, colB = st.columns([1, 3])
with colA:
    run_als = st.button("Train recommendator", type="primary")
with colB:
    st.caption("Only train when pressing (avoiding constant recalculation).")

if run_als:
    with st.spinner("Training ALS…"):
        mat, user_index, item_map = build_interaction_matrix(df, s, weight="quantity")
        als = train_als(mat, factors=int(factors))
        st.session_state.als_bundle = (als, mat, user_index, item_map)
        st.session_state.als_meta = als_meta_now

als_bundle = st.session_state.als_bundle

if als_bundle is None:
    st.info("Press 'Train recommendator' to activate recommendations.")
else:
    als, mat, user_index, item_map = als_bundle

    customer_ids = cust_feat[s.customer_id].dropna().astype(str).unique().tolist()
    cid = st.selectbox("Choose a Customer_ID", customer_ids)

    seg = int(cust_feat.loc[cust_feat[s.customer_id].astype(str) == cid, "segment"].iloc[0])
    st.info(f"Segment of Client: **{seg}**")

    if disc_policy is not None and len(disc_policy) > 0:
        row = disc_policy[disc_policy["segment"] == seg]
        if len(row):
            pct = float(row["recommended_discount_percent"].iloc[0] * 100)
            st.success(f"Discount recomendated for this segment: **{pct:.1f}%**")
        else:
            st.warning("I do not have a policy discount for this segment (not enough data).")

    if st.button("Category Recommendator"):
        recs = recommend_categories(cid, als, mat, user_index, item_map, topn=int(topn))
        if not recs:
            st.warning("No recommendations (cliente not found or not enough interactions).")
        else:
            recs_df = pd.DataFrame(recs, columns=["Recommended category", "Score"])
            recs_df["Score"] = pd.to_numeric(recs_df["Score"], errors="coerce")

            if recs_df["Score"].notna().any():
                mx = recs_df["Score"].max()
                if mx and mx > 0:
                    recs_df["Afinidad (%)"] = (recs_df["Score"] / mx * 100).round(1)
                else:
                    recs_df["Afinidad (%)"] = 0.0
            else:
                recs_df["Afinidad (%)"] = None

            st.dataframe(recs_df, use_container_width=True)
            st.caption("Score/Afinidad = estimated preference of the client by caegory, based in historical behavior.")
