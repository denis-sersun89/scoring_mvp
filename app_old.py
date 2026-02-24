import streamlit as st
import json
import pandas as pd
import uuid
from datetime import datetime, timezone
from scoring_engine import LogisticScoringModel

st.set_page_config(page_title="Corporate Scoring System")

st.title("🏦 Corporate Scoring System")

# --------------------------------------------------
# Файлы
# --------------------------------------------------

MODEL_PATH = "model.json"
DB_PATH = "clients_db.json"
HISTORY_PATH = "history.json"

# --------------------------------------------------
# Excel конфигурация (фиксированный шаблон)
# --------------------------------------------------

EXCEL_CONFIG = {
    "sheet_name": "Scoring",
    "cells": {
        "debt_ratio": "B2",
        "cash_ratio": "B3",
        "max_dpd_2y": "D7"
    }
}

model = LogisticScoringModel(MODEL_PATH)

# --------------------------------------------------
# Загрузка данных
# --------------------------------------------------

@st.cache_data
def load_clients():
    with open(DB_PATH, "r") as f:
        return json.load(f)

def load_history():
    with open(HISTORY_PATH, "r") as f:
        return json.load(f)

def save_history(history):
    with open(HISTORY_PATH, "w") as f:
        json.dump(history, f, indent=4)

clients_db = load_clients()


MODEL_CHANGE_LOG_PATH = "model_change_log.json"

def load_model():
    with open(MODEL_PATH, "r") as f:
        return json.load(f)

def save_model(model_data):
    with open(MODEL_PATH, "w") as f:
        json.dump(model_data, f, indent=4)

def load_model_change_log():
    with open(MODEL_CHANGE_LOG_PATH, "r") as f:
        return json.load(f)

def save_model_change_log(log):
    with open(MODEL_CHANGE_LOG_PATH, "w") as f:
        json.dump(log, f, indent=4)

# --------------------------------------------------
# Excel reader
# --------------------------------------------------

def read_excel_cell(file, sheet_name, cell_reference):
    df = pd.read_excel(
        file,
        sheet_name=sheet_name,
        engine="openpyxl",
        header=None
    )

    column_letter = ''.join(filter(str.isalpha, cell_reference))
    row_number = int(''.join(filter(str.isdigit, cell_reference)))

    column_index = ord(column_letter.upper()) - ord('A')

    return df.iloc[row_number - 1, column_index]

# --------------------------------------------------
# Роли
# --------------------------------------------------

# --------------------------------------------------
# Vertical Menu (Sidebar Buttons)
# --------------------------------------------------

if "role" not in st.session_state:
    st.session_state.role = "User"

st.sidebar.title("Navigation")

if st.sidebar.button("👤 User"):
    st.session_state.role = "User"

if st.sidebar.button("📝 Validator"):
    st.session_state.role = "Validator"

if st.sidebar.button("📜 History"):
    st.session_state.role = "History Viewer"

if st.sidebar.button("⚙ Admin"):
    st.session_state.role = "Admin"

st.sidebar.markdown("---")
st.sidebar.markdown(f"**Current section:** {st.session_state.role}")

role = st.session_state.role





# ==================================================
# USER MODE
# ==================================================

if role == "User":

    st.header("New Scoring")

    data_source = st.radio(
        "Select data source",
        ["Client ID (Database)", "Excel Upload"]
    )

    input_data = None
    client_name = None

    # ---------------------------
    # Client ID mode
    # ---------------------------

    if data_source == "Client ID (Database)":

        client_id = st.text_input("Enter Client ID")

        if client_id and client_id in clients_db:

            client_data = clients_db[client_id]
            client_name = client_data.get("company_name")

            st.json(client_data)

            input_data = {
                "debt_ratio": client_data["debt_ratio"],
                "cash_ratio": client_data["cash_ratio"],
                "max_dpd_2y": client_data["max_dpd_2y"]
            }

        elif client_id:
            st.error("Client not found")

    # ---------------------------
    # Excel mode
    # ---------------------------

    else:

        uploaded_file = st.file_uploader("Upload Excel (.xlsx)", type=["xlsx"])

        if uploaded_file:

            try:
                xls = pd.ExcelFile(uploaded_file)

                if EXCEL_CONFIG["sheet_name"] not in xls.sheet_names:
                    st.error("Required sheet not found")
                else:

                    debt_ratio = read_excel_cell(
                        uploaded_file,
                        EXCEL_CONFIG["sheet_name"],
                        EXCEL_CONFIG["cells"]["debt_ratio"]
                    )

                    cash_ratio = read_excel_cell(
                        uploaded_file,
                        EXCEL_CONFIG["sheet_name"],
                        EXCEL_CONFIG["cells"]["cash_ratio"]
                    )

                    max_dpd = read_excel_cell(
                        uploaded_file,
                        EXCEL_CONFIG["sheet_name"],
                        EXCEL_CONFIG["cells"]["max_dpd_2y"]
                    )

                    input_data = {
                        "debt_ratio": float(debt_ratio),
                        "cash_ratio": float(cash_ratio),
                        "max_dpd_2y": float(max_dpd)
                    }

                    client_name = "Excel Client"

                    st.success("Excel data extracted successfully")
                    st.write(input_data)

            except Exception as e:
                st.error(f"Excel error: {e}")

    # ---------------------------
    # Submit calculation
    # ---------------------------

    if input_data and st.button("Calculate & Submit"):

        result = model.score(input_data)
        history = load_history()

        # ---- вычисляем следующий номер ----
        if history:
            last_no = max(h.get("calculation_no", 0) for h in history)
            next_no = last_no + 1
        else:
            next_no = 1

        record = {
            "calculation_uuid": str(uuid.uuid4()),   # технический ID
            "calculation_no": next_no, # бизнес-номер для удобства
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "client_id": client_id if data_source == "Client ID (Database)" else None,
            "client_name": client_name,
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

        st.success("Calculation submitted for validation")

# ==================================================
# VALIDATOR MODE
# ==================================================

elif role == "Validator":

    st.header("Validation Panel")

    history = load_history()
    pending = [h for h in history if h["status"] == "PENDING"]

    if not pending:
        st.info("No pending calculations")

    else:

        # -------------------------------------------------
        # Человекочитаемый список
        # -------------------------------------------------
        display_map = {
            f"#{h['calculation_no']} | {h['client_name']} (ID: {h.get('client_id','Excel')})":
                h["calculation_uuid"]
            for h in pending
        }

        selected_display = st.selectbox(
            "Select calculation",
            list(display_map.keys())
        )

        selected_uuid = display_map[selected_display]

        selected_record = next(
            h for h in pending if h["calculation_uuid"] == selected_uuid
        )

        st.write("Client:", selected_record["client_name"])
        st.write("Initial Rating:", selected_record["model_output"]["rating"])
        st.write("PD:", f"{selected_record['model_output']['pd']:.2%}")

        st.subheader("Hard Rules")

        startup = st.checkbox("Startup")
        negative_info = st.checkbox("Negative Information")

        comment = st.text_area("Validator comment")

        if st.button("Validate"):

            final_rating = selected_record["model_output"]["rating"]

            if negative_info:
                final_rating = 9
            elif startup:
                final_rating = max(final_rating, 7)

            for h in history:
                if h["calculation_uuid"] == selected_record["calculation_uuid"]:
                    h["hard_rules"]["startup"] = startup
                    h["hard_rules"]["negative_info"] = negative_info
                    h["validator_comment"] = comment
                    h["final_rating"] = final_rating
                    h["status"] = "VALIDATED"
                    h["validated_timestamp"] = datetime.now(timezone.utc).isoformat()

            save_history(history)

            st.success("Calculation validated successfully")

# ==================================================
# HISTORY VIEWER MODE
# ==================================================

elif role == "History Viewer":

    st.header("📜 Scoring History")

    history = load_history()

    if not history:
        st.info("History is empty")
    else:

        # -----------------------------
        # Миграция старых записей
        # -----------------------------
        data_changed = False
        for i, h in enumerate(history):

            if "calculation_no" not in h:
                h["calculation_no"] = i + 1
                data_changed = True

            if "calculation_uuid" not in h:
                h["calculation_uuid"] = str(uuid.uuid4())
                data_changed = True

        if data_changed:
            save_history(history)

        # -----------------------------
        # Превращаем в DataFrame
        # -----------------------------
        df = pd.DataFrame(history)

        df["timestamp_dt"] = pd.to_datetime(df["timestamp"], errors="coerce")

        if "validated_timestamp" in df.columns:
            df["validated_dt"] = pd.to_datetime(df["validated_timestamp"], errors="coerce")
        else:
            df["validated_dt"] = None

        # -----------------------------
        # KPI BLOCK
        # -----------------------------
        total = len(df)
        validated = (df["status"] == "VALIDATED").sum()
        avg_pd = df["model_output"].apply(lambda x: x["pd"]).mean()
        high_risk = (df["final_rating"] >= 8).mean() * 100

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total", total)
        col2.metric("Validated", validated)
        col3.metric("Avg PD", f"{avg_pd:.2%}")
        col4.metric("High Risk %", f"{high_risk:.1f}%")

        st.markdown("---")

        # -----------------------------
        # FILTERS
        # -----------------------------
        status_filter = st.selectbox(
            "Status",
            ["ALL", "PENDING", "VALIDATED"]
        )

        search_client = st.text_input("Search Client Name")

        date_from = st.date_input("Validated From", value=None)
        date_to = st.date_input("Validated To", value=None)

        filtered_df = df.copy()

        if status_filter != "ALL":
            filtered_df = filtered_df[filtered_df["status"] == status_filter]

        if search_client:
            filtered_df = filtered_df[
                filtered_df["client_name"].str.contains(search_client, case=False)
            ]

        if date_from:
            filtered_df = filtered_df[
                filtered_df["validated_dt"] >= pd.to_datetime(date_from)
            ]

        if date_to:
            filtered_df = filtered_df[
                filtered_df["validated_dt"] <= pd.to_datetime(date_to)
            ]

        if filtered_df.empty:
            st.info("No records for selected filters")
        else:

            # -----------------------------
            # Подготовка таблицы
            # -----------------------------
            table_df = pd.DataFrame({
                "UUID": filtered_df["calculation_uuid"],
                "Client ID": filtered_df.get("client_id"),
                "Client Name": filtered_df["client_name"],
                "Calculation No": filtered_df["calculation_no"].apply(
                    lambda x: f"CR-{int(x):05d}"
                ),
                "Timestamp": filtered_df["timestamp_dt"],
                "PD (%)": filtered_df["model_output"].apply(
                    lambda x: round(x["pd"] * 100, 2)
                ),
                "Model Rating": filtered_df["model_output"].apply(
                    lambda x: x["rating"]
                ),
                "Final Rating": filtered_df["final_rating"],
                "Status": filtered_df["status"]
            })

            table_df = table_df.sort_values(
                by="Timestamp",
                ascending=False
            )

            # -----------------------------
            # PAGINATION
            # -----------------------------
            page_size = st.selectbox("Rows per page", [5, 10, 20], index=1)

            total_pages = int((len(table_df) - 1) / page_size) + 1

            page = st.number_input(
                "Page",
                min_value=1,
                max_value=total_pages,
                value=1
            )

            start = (page - 1) * page_size
            end = start + page_size

            page_df = table_df.iloc[start:end]

            # -----------------------------
            # STYLE
            # -----------------------------
            def highlight_rating(val):
                if val >= 8:
                    return "background-color: #5c0000; color: white"
                elif val >= 6:
                    return "background-color: #664400; color: white"
                else:
                    return "background-color: #003300; color: white"

            display_df = page_df.drop(columns=["UUID"])

            styled_df = display_df.style.map(
                highlight_rating,
                subset=["Final Rating"]
            )

            st.dataframe(
                styled_df,
                use_container_width=True,
                hide_index=True
            )

            # -----------------------------
            # DETAIL VIEW
            # -----------------------------
            display_map = {
                f"{row['Calculation No']} | {row['Client Name']}":
                    row["UUID"]
                for _, row in page_df.iterrows()
            }

            selected_display = st.selectbox(
                "Select calculation for details",
                list(display_map.keys())
            )

            selected_uuid = display_map[selected_display]

            selected_record = next(
                h for h in history
                if h["calculation_uuid"] == selected_uuid
            )

            # ---------------------------------------------
            # STRUCTURED DETAILS VIEW
            # ---------------------------------------------

            st.markdown("## 📄 Calculation Details")

            # Header
            st.markdown(
                f"### {selected_record['client_name']}  |  "
                f"CR-{int(selected_record['calculation_no']):05d}"
            )

            st.markdown("---")

            # =========================
            # CLIENT INFO
            # =========================
            st.markdown("### 🧾 Client Information")

            col1, col2, col3 = st.columns(3)

            col1.metric("Client ID", selected_record.get("client_id", "Excel"))
            col2.metric("Status", selected_record.get("status"))
            col3.metric("Final Rating", selected_record.get("final_rating"))

            st.markdown("---")

            # =========================
            # MODEL RESULTS
            # =========================
            st.markdown("### 📊 Model Results")

            model_output = selected_record["model_output"]

            col1, col2, col3 = st.columns(3)

            col1.metric("PD", f"{model_output['pd']:.2%}")
            col2.metric("Model Rating", model_output["rating"])
            col3.metric("Logit Score", f"{model_output['logit_score']:.3f}")

            st.markdown("---")

            # =========================
            # INPUT DATA
            # =========================
            st.markdown("### 📥 Input Data")

            input_df = pd.DataFrame(
                selected_record["input_data"].items(),
                columns=["Variable", "Value"]
            )

            st.dataframe(input_df, use_container_width=True, hide_index=True)

            st.markdown("---")

            # =========================
            # FEATURE CONTRIBUTIONS
            # =========================
            st.markdown("### 📈 Feature Contributions")

            contrib_rows = []

            for var, data in model_output["contributions"].items():
                contrib_rows.append({
                    "Variable": var,
                    "Value": data["value"],
                    "WoE": data["woe"],
                    "Coefficient": data["coefficient"],
                    "Contribution": round(data["contribution"], 4)
                })

            contrib_df = pd.DataFrame(contrib_rows)

            # Подсветка влияния
            def highlight_contribution(val):
                if val > 0:
                    return "background-color: #5c0000; color: white"
                elif val < 0:
                    return "background-color: #003300; color: white"
                return ""

            styled_contrib = contrib_df.style.map(
                highlight_contribution,
                subset=["Contribution"]
            )

            st.dataframe(styled_contrib, use_container_width=True, hide_index=True)

            st.markdown("---")

            # =========================
            # HARD RULES
            # =========================
            st.markdown("### ⚖ Hard Rules")

            hard_rules = selected_record["hard_rules"]

            col1, col2 = st.columns(2)
            col1.metric("Startup", "Yes" if hard_rules["startup"] else "No")
            col2.metric("Negative Info", "Yes" if hard_rules["negative_info"] else "No")

            st.markdown("---")

            # =========================
            # VALIDATION INFO
            # =========================
            st.markdown("### 📝 Validation Info")

            st.write("Validator Comment:", selected_record.get("validator_comment", ""))

            if selected_record.get("validated_timestamp"):
                st.write("Validated At:", selected_record["validated_timestamp"][:16].replace("T", " "))

            st.write("Created At:", selected_record["timestamp"][:16].replace("T", " "))


# ==================================================
# ADMIN MODE
# ==================================================

elif role == "Admin":

    st.header("⚙ Model Administration")

    model_data = load_model()

    st.subheader("Global Model Parameters")

    intercept = st.number_input(
        "Intercept",
        value=float(model_data["model"]["intercept"])
    )

    cal_intercept = st.number_input(
        "Calibration Intercept",
        value=float(model_data["model"]["calibration_intercept"])
    )

    cal_slope = st.number_input(
        "Calibration Slope",
        value=float(model_data["model"]["calibration_slope"])
    )

    st.subheader("Variable Parameters")

    for var_name, var_data in model_data["variables"].items():

        st.markdown(f"### Variable: {var_name}")

        coef = st.number_input(
            f"Coefficient - {var_name}",
            value=float(var_data["coefficient"]),
            key=f"coef_{var_name}"
        )

        model_data["variables"][var_name]["coefficient"] = coef

        for i, bin_data in enumerate(var_data["bins"]):

            woe = st.number_input(
                f"WoE - {var_name} - Bin {i}",
                value=float(bin_data["woe"]),
                key=f"woe_{var_name}_{i}"
            )

            model_data["variables"][var_name]["bins"][i]["woe"] = woe

    st.subheader("Master Scale")

    for i, scale in enumerate(model_data["master_scale"]):

        rating = scale["rating"]

        min_pd = st.number_input(
            f"Min PD - Rating {rating}",
            value=float(scale["min_pd"]),
            key=f"min_pd_{i}"
        )

        max_pd = st.number_input(
            f"Max PD - Rating {rating}",
            value=float(scale["max_pd"]),
            key=f"max_pd_{i}"
        )

        model_data["master_scale"][i]["min_pd"] = min_pd
        model_data["master_scale"][i]["max_pd"] = max_pd

    # -----------------------------
    # SAVE CHANGES
    # -----------------------------

    if st.button("Save Model Changes"):

        old_model = load_model()

        model_data["model"]["intercept"] = intercept
        model_data["model"]["calibration_intercept"] = cal_intercept
        model_data["model"]["calibration_slope"] = cal_slope

        save_model(model_data)

        # Логирование изменений
        change_log = load_model_change_log()

        change_log.append({
            "timestamp": datetime.now().isoformat(),
            "old_model": old_model,
            "new_model": model_data
        })

        save_model_change_log(change_log)

        st.success("Model updated successfully. Restart app to apply changes.")