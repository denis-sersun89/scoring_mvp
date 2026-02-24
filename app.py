import streamlit as st
import json
import pandas as pd
import uuid
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timezone
from scoring_engine import LogisticScoringModel
import numpy as np
from sklearn.metrics import roc_curve, auc



st.set_page_config(page_title="Corporate Scoring System", layout="wide")

st.title("🏦 Corporate Scoring System")

# --------------------------------------------------
# FILES
# --------------------------------------------------

MODEL_PATH = "model.json"
DB_PATH = "clients_db.json"
HISTORY_PATH = "history.json"
MODEL_CHANGE_LOG_PATH = "model_change_log.json"

# --------------------------------------------------
# LOADERS
# --------------------------------------------------

@st.cache_data
def load_clients():
    with open(DB_PATH, "r") as f:
        return json.load(f)

def load_history():
    try:
        with open(HISTORY_PATH, "r") as f:
            return json.load(f)
    except:
        return []

def save_history(history):
    with open(HISTORY_PATH, "w") as f:
        json.dump(history, f, indent=4)

def load_model():
    with open(MODEL_PATH, "r") as f:
        return json.load(f)

def save_model(model_data):
    with open(MODEL_PATH, "w") as f:
        json.dump(model_data, f, indent=4)

def load_model_change_log():
    try:
        with open(MODEL_CHANGE_LOG_PATH, "r") as f:
            return json.load(f)
    except:
        return []

def save_model_change_log(change_log):
    with open(MODEL_CHANGE_LOG_PATH, "w") as f:
        json.dump(change_log, f, indent=4)

# --------------------------------------------------
# UTILITIES
# --------------------------------------------------

def now_utc():
    return datetime.now(timezone.utc).isoformat()

def format_dt(dt_str):
    if not dt_str:
        return ""
    return pd.to_datetime(dt_str).strftime("%Y-%m-%d %H:%M")

def migrate_history(history):
    changed = False
    for i, h in enumerate(history):
        if "calculation_no" not in h:
            h["calculation_no"] = i + 1
            changed = True
        if "calculation_uuid" not in h:
            h["calculation_uuid"] = str(uuid.uuid4())
            changed = True
    if changed:
        save_history(history)
    return history

# --------------------------------------------------
# MODEL
# --------------------------------------------------

model = LogisticScoringModel(MODEL_PATH)
clients_db = load_clients()

# --------------------------------------------------
# NAVIGATION
# --------------------------------------------------

# if "role" not in st.session_state:
#     st.session_state.role = "User"

# st.sidebar.title("Navigation")

# if st.sidebar.button("👤 User"):
#     st.session_state.role = "User"

# if st.sidebar.button("📝 Validator"):
#     st.session_state.role = "Validator"

# if st.sidebar.button("📜 History"):
#     st.session_state.role = "History Viewer"

# if st.sidebar.button("⚙ Admin"):
#     st.session_state.role = "Admin"

# role = st.session_state.role


role = st.sidebar.radio(
    "Navigation",
    ["User", "Validator", "History Viewer", "Admin"]
)
st.session_state.role = role

# ==================================================
# USER MODE
# ==================================================

if role == "User":

    st.header("New Scoring")

    data_source = st.radio(
        "Select data source",
        ["Client ID (Database)", "Excel Upload"]
    )

    # ---- SAFE DEFAULTS ----
    input_data = None
    client_name = None
    client_id = None
    period_raw = None
    formatted_period = "N/A"
    quarter_label = None
    industry = None  # 🔹 NEW

    # --------------------------------------------------
    # Period formatter (global for this block)
    # --------------------------------------------------

    def format_period(period_raw):
        try:
            dt = pd.to_datetime(period_raw, errors="coerce")
            if pd.isna(dt):
                return period_raw, None

            formatted = dt.strftime("%d %b %Y")
            quarter = f"Q{((dt.month-1)//3)+1} {dt.year}"
            return formatted, quarter
        except:
            return period_raw, None

    # ---------------------------------------------
    # DATABASE MODE
    # ---------------------------------------------
    if data_source == "Client ID (Database)":

        client_id = st.text_input("Client ID")

        if client_id and client_id in clients_db:
            client_data = clients_db[client_id]
            client_name = client_data["company_name"]

            # 🔹 NEW
            industry = client_data.get("industry")

            if not industry:
                st.error("Industry missing in database")
                st.stop()

            period_raw = client_data.get("financial_period")
            if period_raw:
                formatted_period, quarter_label = format_period(period_raw)

            input_data = {
                "debt_ratio": client_data["debt_ratio"],
                "cash_ratio": client_data["cash_ratio"],
                "max_dpd_2y": client_data["max_dpd_2y"],
                "industry": industry  # 🔹 NEW
            }

    # ---------------------------------------------
    # EXCEL MODE
    # ---------------------------------------------
    else:

        uploaded_file = st.file_uploader(
            "Upload Excel file",
            type=["xlsx"]
        )

        if uploaded_file:

            try:
                df_excel = pd.read_excel(
                    uploaded_file,
                    header=None,
                    engine="openpyxl"
                )

                # Header data
                client_name = str(df_excel.iloc[0, 1])
                client_id = str(df_excel.iloc[1, 1])
                period_raw = str(df_excel.iloc[2, 1])
                industry = str(df_excel.iloc[6, 1])  

                formatted_period, quarter_label = format_period(period_raw)

                # Financial data
                debt_ratio = pd.to_numeric(df_excel.iloc[3, 1], errors="coerce")
                cash_ratio = pd.to_numeric(df_excel.iloc[4, 1], errors="coerce")
                max_dpd = pd.to_numeric(df_excel.iloc[5, 1], errors="coerce")
                

                if pd.isna(debt_ratio) or pd.isna(cash_ratio) or pd.isna(max_dpd):
                    st.error("Invalid numeric values in Excel file")
                    st.stop()

                debt_ratio = float(debt_ratio)
                cash_ratio = float(cash_ratio)
                max_dpd = float(max_dpd)

                input_data = {
                    "debt_ratio": debt_ratio,
                    "cash_ratio": cash_ratio,
                    "max_dpd_2y": max_dpd,
                    "industry": industry  # 🔹 NEW
                }

                if not industry:
                    st.error("Industry not provided in Excel file")
                    st.stop()

                st.success("Excel successfully loaded")

            except Exception as e:
                st.error(f"Excel format error: {e}")

    # ---------------------------------------------
    # DISPLAY INPUT (CLEAN UI)
    # ---------------------------------------------
    if input_data:

        st.markdown("### 🏢 Client Information")

        colA, colB, colC, colD = st.columns(4)

        colA.metric("Client Name", client_name if client_name else "-")
        colB.metric("Client ID / IDNP", client_id if client_id else "-")
        colC.metric("Industry", industry if industry else "-")
        colD.metric("Financial Period", formatted_period)

        if quarter_label:
            st.caption(f"Reporting Quarter: {quarter_label}")

        st.subheader("📥 Client Indicators")

        col1, col2, col3 = st.columns(3)

        col1.metric("Debt Ratio", f"{input_data['debt_ratio']:.2f}")
        col2.metric("Cash Ratio", f"{input_data['cash_ratio']:.2f}")
        col3.metric("Max DPD (2y)", input_data["max_dpd_2y"])

        if st.button("Calculate & Submit"):

            result = model.score(input_data)

            history = migrate_history(load_history())

            next_no = max(
                [h["calculation_no"] for h in history],
                default=0
            ) + 1

            record = {
                "calculation_uuid": str(uuid.uuid4()),
                "calculation_no": next_no,
                "timestamp": now_utc(),
                "client_id": client_id,
                "client_name": client_name,
                "industry": industry,
                "financial_period": period_raw,   # 🔹 сохраняем период
                "input_data": input_data,
                "model_output": result,
                "hard_rules": {
                    "startup": False,
                    "negative_info": False
                },
                "validator_comment": "",
                "final_rating": result["rating"],
                "status": "PENDING"
            }

            history.append(record)
            save_history(history)

            st.success("Calculation submitted")
# ==================================================
# VALIDATOR
# ==================================================

elif role == "Validator":

    st.header("Validation Panel")

    history = migrate_history(load_history())
    pending = [h for h in history if h["status"] == "PENDING"]

    if not pending:
        st.info("No pending cases")
    else:

        display_map = {
            f"CR-{h['calculation_no']:05d} | {h['client_name']}":
                h["calculation_uuid"]
            for h in pending
        }

        selected = st.selectbox("Select", list(display_map.keys()))
        selected_uuid = display_map[selected]

        record = next(h for h in history if h["calculation_uuid"] == selected_uuid)

        st.write("PD:", f"{record['model_output']['pd']:.2%}")
        st.write("Model Rating:", record["model_output"]["rating"])

        startup = st.checkbox("Startup")
        negative = st.checkbox("Negative Info")
        comment = st.text_area("Comment")

        if st.button("Validate"):

            final_rating = record["model_output"]["rating"]

            if negative:
                final_rating = 9
            elif startup:
                final_rating = max(final_rating, 7)

            for h in history:
                if h["calculation_uuid"] == selected_uuid:
                    h["hard_rules"]["startup"] = startup
                    h["hard_rules"]["negative_info"] = negative
                    h["validator_comment"] = comment
                    h["final_rating"] = final_rating
                    h["status"] = "VALIDATED"
                    h["validated_timestamp"] = now_utc()

            save_history(history)
            st.success("Validated")

# ==================================================
# HISTORY VIEWER
# ==================================================

elif role == "History Viewer":

    st.header("📊 Risk Monitoring Dashboard")

    history = migrate_history(load_history())

    if not history:
        st.info("Empty")
        st.stop()

    df = pd.DataFrame(history)

    df["timestamp_dt"] = pd.to_datetime(
        df["timestamp"],
        errors="coerce",
        utc=True
    )

    if "validated_timestamp" in df.columns:
        ts_values = df["validated_timestamp"].to_numpy()
    else:
        ts_values = pd.Series([None] * len(df)).to_numpy()

    df["validated_dt"] = pd.to_datetime(
        ts_values,
        errors="coerce",
        utc=True
    )
    # ---------------- KPI ----------------

    total = len(df)
    validated = (df["status"] == "VALIDATED").sum()

    avg_pd = (
        df["model_output"]
        .apply(lambda x: x["pd"])
        .mean()
    )

    validated_df = df[df["status"] == "VALIDATED"]

    high_risk = (
        (validated_df["final_rating"] >= 8).mean() * 100
        if not validated_df.empty else 0
    )

    df["processing_time_min"] = (
        (df["validated_dt"] - df["timestamp_dt"])
        .dt.total_seconds() / 60
    )

    avg_sla = df["processing_time_min"].mean()

    col1, col2, col3, col4, col5 = st.columns(5)

    col1.metric("Total", total)
    col2.metric("Validated", validated)
    col3.metric("Avg PD", f"{avg_pd:.2%}")
    col4.metric("High Risk %", f"{high_risk:.1f}%")
    col5.metric("Avg SLA (min)", f"{avg_sla:.1f}" if not pd.isna(avg_sla) else "N/A")

    st.markdown("---")

    # ---------------- Charts ----------------

    st.subheader("PD Distribution")
    fig_pd = px.histogram(
        df["model_output"].apply(lambda x: x["pd"]),
        nbins=20
    )
    st.plotly_chart(fig_pd, use_container_width=True)

    st.subheader("Rating Distribution")
    fig_rating = px.histogram(df["final_rating"])
    st.plotly_chart(fig_rating, use_container_width=True)

    st.markdown("---")

    # ---------------- Table ----------------

    table_df = pd.DataFrame({
        "UUID": df["calculation_uuid"],
        "Client": df["client_name"],
        "Industry": df.get("industry"),
        "Calculation": df["calculation_no"].apply(lambda x: f"CR-{x:05d}"),
        "PD (%)": df["model_output"].apply(lambda x: round(x["pd"]*100,2)),
        "Final Rating": df["final_rating"],
        "Status": df["status"],
        "Created": df["timestamp_dt"].dt.strftime("%Y-%m-%d %H:%M")
    })

    table_df = table_df.sort_values("Created", ascending=False)

    page_size = 10
    total_pages = (len(table_df)-1)//page_size+1

    if "page" not in st.session_state:
        st.session_state.page = 1

    colA, colB, colC = st.columns([1,2,1])

    if colA.button("◀ Prev") and st.session_state.page > 1:
        st.session_state.page -= 1

    colB.write(f"Page {st.session_state.page} of {total_pages}")

    if colC.button("Next ▶") and st.session_state.page < total_pages:
        st.session_state.page += 1

    start = (st.session_state.page-1)*page_size
    end = start+page_size

    page_df = table_df.iloc[start:end]

    st.dataframe(page_df.drop(columns=["UUID"]), use_container_width=True)

    # ---------------- DETAILS ----------------

    display_map = {
        f"{row['Calculation']} | {row['Client']}":
            row["UUID"]
        for _, row in page_df.iterrows()
    }

    selected = st.selectbox("Select for details", list(display_map.keys()))
    selected_uuid = display_map[selected]

    record = next(h for h in history if h["calculation_uuid"] == selected_uuid)

    st.markdown("---")
    st.subheader("📄 Details")

    st.write("Client:", record["client_name"])
    st.write("PD:", f"{record['model_output']['pd']:.2%}")
    st.write("Final Rating:", record["final_rating"])
    st.write("Created:", format_dt(record["timestamp"]))
    st.write("Validated:", format_dt(record.get("validated_timestamp")))

    # -------- Waterfall Explainability --------

    st.subheader("📉 Explainability - Waterfall")

    contrib = record["model_output"]["contributions"]

    base = record["model_output"]["logit_score"] - sum(
        v["contribution"] for v in contrib.values()
    )

    measures = ["absolute"]
    values = [base]
    labels = ["Intercept"]

    for k, v in contrib.items():
        measures.append("relative")
        values.append(v["contribution"])
        labels.append(k)

    measures.append("total")
    values.append(0)
    labels.append("Final Logit")

    fig = go.Figure(go.Waterfall(
        measure=measures,
        x=labels,
        y=values
    ))

    st.plotly_chart(fig, use_container_width=True)

# ==================================================
# ADMIN MODE
# ==================================================

elif role == "Admin":

    st.header("⚙ Model Administration")

    model_data = load_model()

    # ==================================================
    # 📊 MODEL OVERVIEW
    # ==================================================

    st.markdown("## 📊 Model Overview")

    col1, col2, col3 = st.columns(3)

    col1.metric("Variables", len(model_data["variables"]))
    col2.metric("Rating Buckets", len(model_data["master_scale"]))
    col3.metric(
        "Calibration Slope",
        f"{model_data['model']['calibration_slope']:.3f}"
    )

    st.markdown("---")

    # ==================================================
    # 🔧 GLOBAL PARAMETERS
    # ==================================================

    with st.expander("Global Model Parameters", expanded=True):

        col1, col2, col3 = st.columns(3)

        intercept = col1.number_input(
            "Intercept",
            value=float(model_data["model"]["intercept"]),
            format="%.6f"
        )

        cal_intercept = col2.number_input(
            "Calibration Intercept",
            value=float(model_data["model"]["calibration_intercept"]),
            format="%.6f"
        )

        cal_slope = col3.number_input(
            "Calibration Slope",
            value=float(model_data["model"]["calibration_slope"]),
            format="%.6f"
        )

    st.markdown("---")

    st.markdown("## 📈 Variables")

    for var_name, var_data in model_data["variables"].items():

        with st.expander(f"{var_name}", expanded=False):

            # ----- Coefficient -----
            coef = st.number_input(
                f"{var_name} Coefficient",
                value=float(var_data["coefficient"]),
                format="%.6f",
                key=f"coef_{var_name}"
            )

            model_data["variables"][var_name]["coefficient"] = coef

            st.markdown("### WoE Bins")

            bins = var_data.get("bins", [])

            if not bins:
                st.warning("No bins configured for this variable.")
                continue

            for i, bin_data in enumerate(bins):

                col1, col2 = st.columns([3, 1])

                # ==================================================
                # NUMERIC VARIABLE
                # ==================================================
                if "min" in bin_data or "max" in bin_data:

                    bin_min = bin_data.get("min")
                    bin_max = bin_data.get("max")

                    if bin_min is None and bin_max is not None:
                        label = f"(-∞ , {bin_max}]"
                    elif bin_min is not None and bin_max is None:
                        label = f"({bin_min} , +∞)"
                    elif bin_min is not None and bin_max is not None:
                        label = f"({bin_min} , {bin_max}]"
                    else:
                        label = f"Bin {i}"

                # ==================================================
                # CATEGORICAL VARIABLE
                # ==================================================
                elif "values" in bin_data:

                    categories = bin_data["values"]

                    if isinstance(categories, list):
                        label = f"[ {', '.join(map(str, categories))} ]"
                    else:
                        label = f"[ {categories} ]"

                else:
                    label = f"Bin {i}"

                col1.markdown(f"**{label}**")

                woe = col2.number_input(
                    f"WoE {var_name}_{i}",
                    value=float(bin_data["woe"]),
                    format="%.6f",
                    key=f"woe_{var_name}_{i}"
                )

                model_data["variables"][var_name]["bins"][i]["woe"] = woe
    # ==================================================
    # 🏦 MASTER SCALE (WITH VALIDATION)
    # ==================================================

    st.markdown("## 🏦 Rating Master Scale")

    with st.expander("Edit Master Scale Buckets", expanded=False):

        scale_error = False

        for i, scale in enumerate(model_data["master_scale"]):

            rating = scale["rating"]

            st.markdown(f"### Rating {rating}")

            col1, col2 = st.columns(2)

            min_pd = col1.number_input(
                f"Min PD (Rating {rating})",
                value=float(scale["min_pd"]),
                format="%.6f",
                key=f"min_pd_{i}"
            )

            max_pd = col2.number_input(
                f"Max PD (Rating {rating})",
                value=float(scale["max_pd"]),
                format="%.6f",
                key=f"max_pd_{i}"
            )

            # 🔴 Validation
            if min_pd > max_pd:
                scale_error = True
                st.error(f"⚠ Rating {rating}: Min PD > Max PD")

            model_data["master_scale"][i]["min_pd"] = min_pd
            model_data["master_scale"][i]["max_pd"] = max_pd
    # ==================================================
    # 📊 PD CURVE PREVIEW
    # ==================================================

    st.markdown("## 📊 PD Curve Preview")

    logits = np.linspace(-5, 5, 200)

    old_model = load_model()
    old_ci = old_model["model"]["calibration_intercept"]
    old_cs = old_model["model"]["calibration_slope"]

    old_pd = 1 / (1 + np.exp(-(old_ci + old_cs * logits)))
    new_pd = 1 / (1 + np.exp(-(cal_intercept + cal_slope * logits)))

    fig_curve = go.Figure()

    fig_curve.add_trace(go.Scatter(
        x=logits,
        y=old_pd,
        name="Current"
    ))

    fig_curve.add_trace(go.Scatter(
        x=logits,
        y=new_pd,
        name="New",
        line=dict(dash="dash")
    ))

    fig_curve.update_layout(
        xaxis_title="Logit",
        yaxis_title="PD",
        height=400
    )

    st.plotly_chart(fig_curve, use_container_width=True)

    st.markdown("---")

    # ==================================================
    # 📈 IMPACT SIMULATION
    # ==================================================

    st.markdown("## 📈 Impact Simulation")

    sample_logit = st.slider(
        "Example Logit Score",
        -5.0, 5.0, 0.0, 0.1
    )

    old_pd_value = 1 / (1 + np.exp(-(old_ci + old_cs * sample_logit)))
    new_pd_value = 1 / (1 + np.exp(-(cal_intercept + cal_slope * sample_logit)))

    col1, col2 = st.columns(2)

    col1.metric("Current PD", f"{old_pd_value:.2%}")
    col2.metric(
        "New PD",
        f"{new_pd_value:.2%}",
        delta=f"{(new_pd_value - old_pd_value):.2%}"
    )

    st.markdown("---")

    # ==================================================
    # 📊 ROC + GINI + KS                   !!!!!!!!!!!!
    # ==================================================

    # neavind y_true real, simulam random pentru a vedea cum se schimba AUC/Gini/KS cu noua calibratie
    st.markdown("## 📊 Model Validation Analytics")

    np.random.seed(42)
    n = 1000
    logits_sim = np.random.normal(0, 1.2, n)

    pd_sim = 1 / (1 + np.exp(-(old_ci + old_cs * logits_sim)))
    y_true = np.random.binomial(1, pd_sim)

    fpr, tpr, _ = roc_curve(y_true, pd_sim)
    roc_auc = auc(fpr, tpr)
    gini = 2 * roc_auc - 1

    fig_roc = go.Figure()
    fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, name=f"ROC AUC={roc_auc:.3f}"))
    fig_roc.add_trace(go.Scatter(
        x=[0,1], y=[0,1],
        line=dict(dash="dash"),
        name="Random"
    ))

    fig_roc.update_layout(height=400)
    st.plotly_chart(fig_roc, use_container_width=True)

    col1, col2 = st.columns(2)
    col1.metric("AUC", f"{roc_auc:.3f}")
    col2.metric("Gini", f"{gini:.3f}")

    # KS

    df_val = pd.DataFrame({
        "pd": pd_sim,
        "target": y_true
    }).sort_values("pd")

    df_val["cum_good"] = ((df_val["target"] == 0).cumsum()) / (y_true == 0).sum()
    df_val["cum_bad"] = ((df_val["target"] == 1).cumsum()) / (y_true == 1).sum()
    df_val["ks"] = df_val["cum_bad"] - df_val["cum_good"]

    ks_stat = df_val["ks"].max()

    fig_ks = go.Figure()
    fig_ks.add_trace(go.Scatter(y=df_val["cum_good"], name="Good"))
    fig_ks.add_trace(go.Scatter(y=df_val["cum_bad"], name="Bad"))

    fig_ks.update_layout(
        title=f"KS Curve (KS={ks_stat:.3f})",
        height=400
    )

    st.plotly_chart(fig_ks, use_container_width=True)
    st.metric("KS Statistic", f"{ks_stat:.3f}")

    st.markdown("---")

    # ==================================================
    # 💾 SAVE
    # ==================================================

    st.markdown("## 💾 Save Changes")

    confirm = st.checkbox("I confirm model parameter changes")

    if st.button("Save Model Changes", type="primary"):

        if not confirm:
            st.warning("Please confirm changes.")
        elif scale_error:
            st.error("Fix Master Scale errors before saving.")
        else:

            old_model = load_model()

            model_data["model"]["intercept"] = intercept
            model_data["model"]["calibration_intercept"] = cal_intercept
            model_data["model"]["calibration_slope"] = cal_slope

            save_model(model_data)

            change_log = load_model_change_log()
            change_log.append({
                "timestamp": datetime.now().isoformat(),
                "old_model": old_model,
                "new_model": model_data
            })
            save_model_change_log(change_log)

            st.success("Model updated successfully.")