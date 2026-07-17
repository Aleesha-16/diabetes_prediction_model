import os
import gradio as gr
import joblib
import numpy as np

# Load trained model
diabetes_model = joblib.load("diabetes_prediction_model.pkl")

# Debug info
try:
    print("Model loaded successfully")
    print("Expected features:", diabetes_model.n_features_in_)
except Exception:
    pass


def predict_diabetes(
    pregnancies,
    glucose,
    blood_pressure,
    skin_thickness,
    insulin,
    bmi,
    diabetes_pedigree,
    age,
):
    try:
        input_data = np.array([[
            float(pregnancies),
            float(glucose),
            float(blood_pressure),
            float(skin_thickness),
            float(insulin),
            float(bmi),
            float(diabetes_pedigree),
            float(age)
        ]])

        prediction = diabetes_model.predict(input_data)

        if prediction[0] == 1:
            return "⚠️ Positive for Diabetes\n\nPlease consult a healthcare professional."
        else:
            return "✅ No Diabetes Detected"

    except Exception as e:
        return f"Prediction Error: {str(e)}"


demo = gr.Interface(
    fn=predict_diabetes,
    inputs=[
        gr.Number(label="👶 Pregnancies", value=0),
        gr.Number(label="🩸 Glucose Level", value=120),
        gr.Number(label="💓 Blood Pressure", value=70),
        gr.Number(label="📏 Skin Thickness", value=20),
        gr.Number(label="💉 Insulin", value=79),
        gr.Number(label="⚖️ BMI", value=25.0),
        gr.Number(label="🧬 Diabetes Pedigree Function", value=0.5),
        gr.Number(label="🎂 Age", value=30),
    ],
    outputs=gr.Textbox(label="Prediction Result"),
    title="🩺 Diabetes Prediction System",
    description="Enter the patient's medical information below. The machine learning model will predict whether the patient is likely to have diabetes.",
    theme=gr.themes.Soft(),
)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))

    demo.launch(
        server_name="0.0.0.0",
        server_port=port
    )
