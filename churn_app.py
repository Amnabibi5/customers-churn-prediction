import gradio as gr
import numpy as np
import pickle
import pandas as pd

# Load the trained model
with open("/content/rf_model.pkl", "rb") as f:  # Update path as needed
    model = pickle.load(f)

# Define the prediction function with mapping
def predict_churn(gender, senior, partner, dependents, tenure, phoneservice, multiplelines, internetservice,
                  onlinesecurity, onlinebackup, deviceprotection, techsupport, streamingtv, streamingmovies,
                  contract, paperlessbilling, paymentmethod, monthly, total):

    # Map string inputs to numeric codes
    gender_map = {"Male": 1, "Female": 0}

    gender = gender_map[gender]

    # Prepare input data
    input_data = pd.DataFrame([[gender, senior, partner, dependents, tenure, phoneservice, multiplelines,
                                internetservice, onlinesecurity, onlinebackup, deviceprotection, techsupport,
                                streamingtv, streamingmovies, contract, paperlessbilling, paymentmethod,
                                monthly, total, 0]],  # Placeholder for 'tenure_group'
        columns=['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure', 'PhoneService',
                 'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                 'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling',
                 'PaymentMethod', 'MonthlyCharges', 'TotalCharges', 'tenure_group'])

    # Create tenure_group
    input_data['tenure_group'] = pd.cut(input_data['tenure'],
                                        bins=[0, 12, 24, 48, 60, 72],
                                        labels=['0-12', '13-24', '25-48', '49-60', '61-72']).cat.codes

    # Make prediction
    prediction = model.predict(input_data)[0]

    return "⚠️ Likely to Churn" if prediction == 1 else "✅ Not Likely to Churn"

# Build Gradio UI
interface = gr.Interface(
    fn=predict_churn,
    inputs=[
        gr.Dropdown(["Male", "Female"], label="Gender", value="Male"),
        gr.Radio([0, 1], label="Senior Citizen (0 = No, 1 = Yes)", value=0),
        gr.Radio([0, 1], label="Has Partner (0=No, 1=Yes)", value=0),
        gr.Radio([0, 1], label="Has Dependents (0=No, 1=Yes)", value=0),
        gr.Slider(0, 72, step=1, label="Tenure (months)", value=1),
        gr.Radio([0, 1], label="Phone Service (0=No, 1=Yes)", value=1),
        gr.Radio([0, 1, 2], label="Multiple Lines (0=No, 1=No phone service, 2=Yes)", value=0),
        gr.Radio([0, 1, 2], label="Internet Service (0=DSL, 1=Fiber optic, 2=No)", value=0),
        gr.Radio([0, 1, 2], label="Online Security (0=No, 1=No internet service, 2=Yes)", value=0),
        gr.Radio([0, 1, 2], label="Online Backup (0=No, 1=No internet service, 2=Yes)", value=0),
        gr.Radio([0, 1, 2], label="Device Protection (0=No, 1=No internet service, 2=Yes)", value=0),
        gr.Radio([0, 1, 2], label="Tech Support (0=No, 1=No internet service, 2=Yes)", value=0),
        gr.Radio([0, 1, 2], label="Streaming TV (0=No, 1=No internet service, 2=Yes)", value=0),
        gr.Radio([0, 1, 2], label="Streaming Movies (0=No, 1=No internet service, 2=Yes)", value=0),
        gr.Radio([0, 1, 2], label="Contract (0=Month-to-month, 1=One year, 2=Two year)", value=0),
        gr.Radio([0, 1], label="Paperless Billing (0=No, 1=Yes)", value=1),
        gr.Radio([0, 1, 2, 3], label="Payment Method (0=Bank transfer, 1=Credit card, 2=Electronic check, 3=Mailed check)", value=2),
        gr.Number(label="Monthly Charges", value=29.85),
        gr.Number(label="Total Charges", value=29.85)
    ],
    outputs="text",
    title="📊 Customer Churn Prediction",
    description="Enter customer info to check churn risk."
)

# Launch the app
interface.launch()
