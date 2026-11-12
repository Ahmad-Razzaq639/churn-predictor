import joblib
import pandas as pd
import streamlit as st

import churn_utils  # noqa: F401  (pickle ko clean() chahiye)

st.set_page_config(page_title="Telco Churn Predictor", page_icon="📉", layout="centered")


@st.cache_resource
def load_model():
    return joblib.load("model.joblib")


model = load_model()

st.title("📉 Telco Customer Churn Predictor")
st.caption("Customer ki details bharo — model batayega ke wo company chhodne ka risk rakhta hai ya nahi.")

with st.form("customer"):
    st.subheader("👤 Customer info")
    c1, c2, c3 = st.columns(3)
    gender = c1.selectbox("Gender", ["Female", "Male"])
    senior = c2.selectbox("Senior Citizen", ["No", "Yes"])
    partner = c3.selectbox("Partner", ["No", "Yes"])
    dependents = c1.selectbox("Dependents", ["No", "Yes"])
    tenure = c2.number_input("Tenure (months)", 0, 100, 12)

    st.subheader("📞 Services")
    c1, c2 = st.columns(2)
    phone = c1.selectbox("Phone Service", ["Yes", "No"])
    lines = c2.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
    internet = c1.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    addon_opts = ["No", "Yes", "No internet service"]
    sec = c1.selectbox("Online Security", addon_opts)
    backup = c2.selectbox("Online Backup", addon_opts)
    device = c1.selectbox("Device Protection", addon_opts)
    tech = c2.selectbox("Tech Support", addon_opts)
    tv = c1.selectbox("Streaming TV", addon_opts)
    movies = c2.selectbox("Streaming Movies", addon_opts)

    st.subheader("💳 Billing")
    c1, c2 = st.columns(2)
    contract = c1.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    paperless = c2.selectbox("Paperless Billing", ["Yes", "No"])
    payment = st.selectbox("Payment Method", [
        "Electronic check", "Mailed check",
        "Bank transfer (automatic)", "Credit card (automatic)"])
    c1, c2 = st.columns(2)
    monthly = c1.number_input("Monthly Charges", 0.0, 200.0, 70.0, step=0.5)
    total = c2.number_input("Total Charges", 0.0, 10000.0, float(monthly * tenure), step=1.0)

    submit = st.form_submit_button("Predict", use_container_width=True)

if submit:
    row = pd.DataFrame([{
        "gender": gender, "SeniorCitizen": 1 if senior == "Yes" else 0,
        "Partner": partner, "Dependents": dependents, "tenure": tenure,
        "PhoneService": phone, "MultipleLines": lines, "InternetService": internet,
        "OnlineSecurity": sec, "OnlineBackup": backup, "DeviceProtection": device,
        "TechSupport": tech, "StreamingTV": tv, "StreamingMovies": movies,
        "Contract": contract, "PaperlessBilling": paperless, "PaymentMethod": payment,
        "MonthlyCharges": monthly, "TotalCharges": total,
    }])
    pred = int(model.predict(row)[0])
    prob = float(model.predict_proba(row)[0][1])

    st.divider()
    if pred == 1:
        st.error("⚠️ **Churn hone ka high risk** — ye customer chhod sakta hai.")
    else:
        st.success("✅ **Low risk** — ye customer rukne ki zyada sambhavna hai.")
    st.metric("Churn risk score", f"{prob:.0%}")
    st.progress(min(max(prob, 0.0), 1.0))
    st.caption("Note: model SMOTE-balanced data pe train hua hai, isliye score ko "
               "'exact probability' nahi, relative risk score samjho.")
