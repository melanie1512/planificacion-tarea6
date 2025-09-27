import streamlit as st
import requests

st.set_page_config(page_title="Heart Predictor", layout="centered")

st.title("Heart Failure Prediction")
st.caption("API FastAPI existente (endpoints: /health, /predict)")

api_url = st.text_input("API URL", "http://127.0.0.1:8000")

with st.expander("Probar conexión /health", expanded=False):
    if st.button("Ping /health"):
        try:
            r = requests.get(f"{api_url}/health", timeout=5)
            st.code(r.json(), language="json")
        except Exception as e:
            st.error(f"Error: {e}")

st.divider()
st.subheader("Predicción individual (/predict)")

presets = {
    "— seleccionar —": None,
    "NO_TIENE (ejemplo)": {
        "Age": 42, "Sex": "F", "ChestPainType": "ATA",
        "RestingBP": 118, "Cholesterol": 190, "FastingBS": 0,
        "RestingECG": "Normal", "MaxHR": 172, "ExerciseAngina": "N",
        "Oldpeak": 0.0, "ST_Slope": "Up"
    },
    "TIENE (ejemplo)": {
        "Age": 63, "Sex": "F", "ChestPainType": "ASY",
        "RestingBP": 148, "Cholesterol": 260, "FastingBS": 1,
        "RestingECG": "ST", "MaxHR": 118, "ExerciseAngina": "Y",
        "Oldpeak": 2.3, "ST_Slope": "Flat"
    },
}

choice = st.selectbox("Usar ejemplo:", list(presets.keys()), index=0)

defaults = presets.get(choice) or {
    "Age": 58, "Sex": "M", "ChestPainType": "NAP",
    "RestingBP": 124, "Cholesterol": 220, "FastingBS": 0,
    "RestingECG": "Normal", "MaxHR": 165, "ExerciseAngina": "N",
    "Oldpeak": 0.8, "ST_Slope": "Up"
}

with st.form("pred_form"):
    col1, col2 = st.columns(2)
    with col1:
        Age = st.number_input("Age", min_value=1, max_value=120, value=defaults["Age"])
        Sex = st.selectbox("Sex", ["M","F"], index=0 if defaults["Sex"]=="M" else 1)
        ChestPainType = st.selectbox("ChestPainType", ["TA","ATA","NAP","ASY"], index=["TA","ATA","NAP","ASY"].index(defaults["ChestPainType"]))
        RestingBP = st.number_input("RestingBP", min_value=0, max_value=300, value=defaults["RestingBP"])
        Cholesterol = st.number_input("Cholesterol", min_value=0, max_value=1000, value=defaults["Cholesterol"])
        FastingBS = st.selectbox("FastingBS (0/1)", [0,1], index=defaults["FastingBS"])
    with col2:
        RestingECG = st.selectbox("RestingECG", ["Normal","ST","LVH"], index=["Normal","ST","LVH"].index(defaults["RestingECG"]))
        MaxHR = st.number_input("MaxHR", min_value=0, max_value=250, value=defaults["MaxHR"])
        ExerciseAngina = st.selectbox("ExerciseAngina", ["Y","N"], index=0 if defaults["ExerciseAngina"]=="Y" else 1)
        Oldpeak = st.number_input("Oldpeak", min_value=-5.0, max_value=10.0, value=float(defaults["Oldpeak"]), step=0.1, format="%.1f")
        ST_Slope = st.selectbox("ST_Slope", ["Up","Flat","Down"], index=["Up","Flat","Down"].index(defaults["ST_Slope"]))
        threshold = st.slider("Threshold (no usado por tu backend actual)", 0.0, 1.0, 0.5, 0.01)

    submitted = st.form_submit_button("Predict")
    if submitted:
        payload = {
            "Age": int(Age), "Sex": Sex, "ChestPainType": ChestPainType,
            "RestingBP": int(RestingBP), "Cholesterol": int(Cholesterol), "FastingBS": int(FastingBS),
            "RestingECG": RestingECG, "MaxHR": int(MaxHR), "ExerciseAngina": ExerciseAngina,
            "Oldpeak": float(Oldpeak), "ST_Slope": ST_Slope
        }
        try:
            r = requests.post(f"{api_url}/predict", json=payload, params={"threshold": threshold}, timeout=10)
            r.raise_for_status()
            res = r.json()
            
            label = "tiene" if int(res.get("prediction", 0)) == 1 else "no_tiene"
            st.success(f"Predicción: {label}")
            st.code(res, language="json")
        except Exception as e:
            st.error(f"Error llamando a /predict: {e}")
            st.code(payload, language="json")
