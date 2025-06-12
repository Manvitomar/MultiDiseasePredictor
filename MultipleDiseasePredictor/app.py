import streamlit as st
import pickle
import numpy as np

# Load models
diabetes_model = pickle.load(open("models\diabetesModel.pkl", "rb"))
heart_model = pickle.load(open("models\heartModel.pkl", "rb"))
kidney_model = pickle.load(open("models\kidneyModel.pkl", "rb"))

# Define ranges
ranges = {
    "diabetes": {
        "Pregnancies": (0, 10), "Glucose": (70, 140), "BloodPressure": (60, 80),
        "SkinThickness": (10, 50), "Insulin": (16, 166), "BMI": (18.5, 24.9),
        "DiabetesPedigreeFunction": (0.0, 1.0), "Age": (20, 60)
    },
    "heart": {
        "Age": (30, 60), "RestingBP": (90, 120), "Cholesterol": (125, 200),
        "MaxHR": (100, 200), "Oldpeak": (0.0, 2.0)
    },
    "kidney": {
        "Age": (20, 60), "BloodPressure": (60, 80), "SpecificGravity": (1.005, 1.025),
        "Albumin": (0, 1), "Sugar": (0, 1), "BloodGlucoseRandom": (70, 140),
        "BloodUrea": (7, 20), "SerumCreatinine": (0.6, 1.2), "Sodium": (135, 145),
        "Potassium": (3.5, 5.0), "Hemoglobin": (12, 17.5)
    }
}

# Tips for abnormal values
tips = {
    "Glucose": {
        "low": "Include more healthy carbs like fruits and whole grains.",
        "high": "Avoid sugar, processed foods, and stay active."
    },
    "BloodPressure": {
        "low": "Increase fluids and salt slightly, stay hydrated.",
        "high": "Reduce sodium, manage stress, and exercise."
    },
    "BMI": {
        "low": "Increase calorie intake with nutritious food.",
        "high": "Limit carbs and increase physical activity."
    },
    "Insulin": {
        "low": "Include more whole grains and protein in your meals.",
        "high": "Consult a doctor to adjust insulin intake."
    },
    "Age": {
        "low": "Maintain healthy habits from a young age.",
        "high": "Regular check-ups and a balanced lifestyle are key."
    },
    "DiabetesPedigreeFunction": {
        "low": "Genetic risk is low. Keep healthy habits.",
        "high": "You may be genetically at risk. Stay active and eat well."
    },
    "Cholesterol": {
        "low": "Include healthy fats like nuts and seeds.",
        "high": "Reduce saturated fats and exercise regularly."
    },
    "MaxHR": {
        "low": "Do light cardio and build endurance slowly.",
        "high": "Stay hydrated and avoid overexertion."
    },
    "Oldpeak": {
        "low": "Stable. Just maintain your routine.",
        "high": "May indicate stress or heart risk. Get it checked."
    },
    "Hemoglobin": {
        "low": "Add iron-rich foods like spinach and lentils.",
        "high": "Avoid excess iron supplements unless prescribed."
    }
}

# Format and show insight
def show_insight(value, normal_range, label):
    if value < normal_range[0]:
        status = "Low"
        tip = tips.get(label, {}).get("low")
    elif value > normal_range[1]:
        status = "High"
        tip = tips.get(label, {}).get("high")
    else:
        status = "Normal"
        tip = None

    info = (
        f"**{label}:** {value} | **Status:** {status} "
        f"(Normal Range: {normal_range[0]} - {normal_range[1]})"
    )
    if tip:
        info += f"\n**Tip:** {tip}"
    return info

# App layout
st.set_page_config(page_title="Multi Disease Predictor", layout="centered")
st.title("✨ Multi Disease Predictor")
st.markdown("Predict **Diabetes**, **Heart**, or **Kidney Disease** with insights and health tips.")

# Sidebar option
option = st.sidebar.radio("Choose Prediction", ["Diabetes", "Heart Disease", "Kidney Disease"])

# Diabetes
if option == "Diabetes":
    st.header("🧬 Diabetes Prediction")
    labels = list(ranges["diabetes"].keys())
    inputs = [st.number_input(f"{label}") for label in labels]
    user_data = np.array([inputs])

    if st.button("Predict Diabetes"):
        result = diabetes_model.predict(user_data)
        prob = diabetes_model.predict_proba(user_data)[0][1] * 100

        st.subheader("Result")
        if result[0] == 0:
            st.success("No signs of Diabetes.")
        else:
            st.error(f"Risk: {round(prob, 2)}% for Diabetes")

        st.markdown("---")
        st.subheader("Parameter Insights")
        for i, label in enumerate(labels):
            st.markdown(show_insight(inputs[i], ranges['diabetes'][label], label))

# Heart Disease
elif option == "Heart Disease":
    st.header("❤️ Heart Disease Prediction")
    age = st.number_input("Age")
    sex = st.radio("Sex", [1, 0], format_func=lambda x: "Male" if x == 1 else "Female")
    cp = st.selectbox("Chest Pain Type (0–3)", [0, 1, 2, 3])
    trestbps = st.number_input("Resting BP")
    chol = st.number_input("Cholesterol")
    fbs = st.radio("Fasting Blood Sugar > 120", [1, 0])
    restecg = st.selectbox("Resting ECG (0–2)", [0, 1, 2])
    thalach = st.number_input("Max Heart Rate")
    exang = st.radio("Exercise Induced Angina", [1, 0])
    oldpeak = st.number_input("Oldpeak")
    slope = st.selectbox("Slope (0–2)", [0, 1, 2])
    ca = st.selectbox("Major Vessels Colored (0–3)", [0, 1, 2, 3])
    thal = st.selectbox("Thalassemia (1–3)", [1, 2, 3])

    features = [age, sex, cp, trestbps, chol, fbs, restecg, thalach,
                exang, oldpeak, slope, ca, thal]
    user_data = np.array([features])

    if st.button("Predict Heart Disease"):
        result = heart_model.predict(user_data)
        prob = heart_model.predict_proba(user_data)[0][1] * 100

        st.subheader("Result")
        if result[0] == 0:
            st.success("Heart is healthy.")
        else:
            st.error(f"Risk: {round(prob, 2)}% for Heart Disease")

        st.markdown("---")
        st.subheader("Parameter Insights")
        summary = {
            "Age": age, "RestingBP": trestbps,
            "Cholesterol": chol, "MaxHR": thalach, "Oldpeak": oldpeak
        }
        for label, val in summary.items():
            st.markdown(show_insight(val, ranges['heart'][label], label))

# Kidney Disease
elif option == "Kidney Disease":
    st.header("🧪 Kidney Disease Prediction")
    labels = list(ranges["kidney"].keys())
    inputs = [st.number_input(f"{label}") for label in labels]
    padded_data = np.array([inputs + [0] * (25 - len(inputs))])

    if st.button("Predict Kidney Disease"):
        result = kidney_model.predict(padded_data)
        prob = kidney_model.predict_proba(padded_data)[0][1] * 100

        st.subheader("Result")
        if result[0] == 0:
            st.success("No signs of Kidney Disease.")
        else:
            st.error(f"Risk: {round(prob, 2)}% for Kidney Disease")

        st.markdown("---")
        st.subheader("Parameter Insights")
        for i, label in enumerate(labels):
            st.markdown(show_insight(inputs[i], ranges['kidney'][label], label))
