import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt

# Page configuration
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .prediction-high-risk {
        background-color: #ffebee;
        border-left: 5px solid #f44336;
        padding: 1rem;
        border-radius: 0.5rem;
    }
    .prediction-low-risk {
        background-color: #e8f5e8;
        border-left: 5px solid #4caf50;
        padding: 1rem;
        border-radius: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Load the trained model and preprocessing objects
@st.cache_resource
def load_model():
    try:
        with open('churn_prediction_model.pkl', 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        st.error("Model file not found. Please ensure 'churn_prediction_model.pkl' is in the same directory.")
        return None

# Load model
model_objects = load_model()

if model_objects is None:
    st.stop()

# Extract objects
model = model_objects['best_model']
scaler = model_objects['scaler']
label_encoders = model_objects['label_encoders']
feature_names = model_objects['feature_names']
model_name = model_objects['model_name']
performance_metrics = model_objects['performance_metrics']

# Title and description
st.markdown('<h1 class="main-header">🔮 Customer Churn Prediction Dashboard</h1>', unsafe_allow_html=True)

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Choose a page:", ["🏠 Home", "🔮 Prediction", "📊 Analytics", "ℹ️ Model Info"])

if page == "🏠 Home":
    st.markdown("## Welcome to the Customer Churn Prediction System")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            label="Model Type",
            value=model_name,
            help="The best performing model selected during training"
        )
    
    with col2:
        st.metric(
            label="Accuracy",
            value=f"{performance_metrics['accuracy']:.1%}",
            help="Model accuracy on test data"
        )
    
    with col3:
        st.metric(
            label="ROC AUC Score",
            value=f"{performance_metrics['roc_auc']:.3f}",
            help="Area under the ROC curve"
        )
    
    st.markdown("---")
    
    st.markdown("""
    ### 🎯 Purpose
    This application helps predict whether a customer is likely to churn (cancel their subscription) 
    based on various customer attributes and usage patterns.
    
    ### 🔍 Features
    - **Individual Predictions**: Get churn probability for a specific customer
    - **Batch Predictions**: Upload a CSV file for multiple predictions
    - **Analytics Dashboard**: Explore model insights and feature importance
    - **Model Information**: Learn about the model's performance and methodology
    
    ### 🚀 How to Use
    1. Navigate to the **Prediction** page to make individual or batch predictions
    2. Use the **Analytics** page to explore model insights
    3. Check **Model Info** for technical details about the model
    """)

elif page == "🔮 Prediction":
    st.markdown("## 🔮 Make Predictions")
    
    # Prediction type selection
    prediction_type = st.radio("Choose prediction type:", ["Single Customer", "Batch Prediction"])
    
    if prediction_type == "Single Customer":
        st.markdown("### Enter Customer Information")
        
        # Create input form
        with st.form("customer_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                # Demographics
                st.subheader("Demographics")
                gender = st.selectbox("Gender", ["Female", "Male"])
                senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])
                partner = st.selectbox("Partner", ["No", "Yes"])
                dependents = st.selectbox("Dependents", ["No", "Yes"])
                
                # Account Information
                st.subheader("Account Information")
                tenure = st.slider("Tenure (months)", 0, 72, 24)
                contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
                paperless_billing = st.selectbox("Paperless Billing", ["No", "Yes"])
                payment_method = st.selectbox("Payment Method", 
                                            ["Electronic check", "Mailed check", 
                                             "Bank transfer (automatic)", "Credit card (automatic)"])
                
            with col2:
                # Services
                st.subheader("Services")
                phone_service = st.selectbox("Phone Service", ["No", "Yes"])
                multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
                internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
                online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
                online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
                device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
                tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
                streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
                streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])
                
                # Charges
                st.subheader("Charges")
                monthly_charges = st.number_input("Monthly Charges", 0.0, 200.0, 50.0)
                total_charges = st.number_input("Total Charges", 0.0, 10000.0, 1000.0)
            
            submitted = st.form_submit_button("Predict Churn")
            
            if submitted:
                # Create feature dictionary
                input_data = {
                    'gender': gender,
                    'SeniorCitizen': 1 if senior_citizen == "Yes" else 0,
                    'Partner': partner,
                    'Dependents': dependents,
                    'tenure': tenure,
                    'PhoneService': phone_service,
                    'MultipleLines': multiple_lines,
                    'InternetService': internet_service,
                    'OnlineSecurity': online_security,
                    'OnlineBackup': online_backup,
                    'DeviceProtection': device_protection,
                    'TechSupport': tech_support,
                    'StreamingTV': streaming_tv,
                    'StreamingMovies': streaming_movies,
                    'Contract': contract,
                    'PaperlessBilling': paperless_billing,
                    'PaymentMethod': payment_method,
                    'MonthlyCharges': monthly_charges,
                    'TotalCharges': total_charges
                }
                
                # Create feature engineering variables
                if tenure == 0:
                    tenure_group = 0  # 0-12 months
                elif tenure <= 12:
                    tenure_group = 0
                elif tenure <= 24:
                    tenure_group = 1
                elif tenure <= 48:
                    tenure_group = 2
                elif tenure <= 60:
                    tenure_group = 3
                else:
                    tenure_group = 4
                
                # Monthly charges group
                if monthly_charges <= 35:
                    monthly_charges_group = 0
                elif monthly_charges <= 65:
                    monthly_charges_group = 1
                elif monthly_charges <= 89:
                    monthly_charges_group = 2
                else:
                    monthly_charges_group = 3
                
                # Charges per tenure
                charges_per_tenure = total_charges / (tenure + 1)
                
                input_data['tenure_group'] = tenure_group
                input_data['monthly_charges_group'] = monthly_charges_group
                input_data['charges_per_tenure'] = charges_per_tenure
                
                # Convert to DataFrame
                input_df = pd.DataFrame([input_data])
                
                # Apply label encoding
                for col in input_df.columns:
                    if col in label_encoders and col not in ['SeniorCitizen', 'tenure', 'MonthlyCharges', 
                                                             'TotalCharges', 'tenure_group', 
                                                             'monthly_charges_group', 'charges_per_tenure']:
                        try:
                            input_df[col] = label_encoders[col].transform(input_df[col])
                        except ValueError:
                            # Handle unseen labels
                            input_df[col] = 0
                
                # Ensure all features are present and in correct order
                for feature in feature_names:
                    if feature not in input_df.columns:
                        input_df[feature] = 0
                
                input_df = input_df[feature_names]
                
                # Scale features
                input_scaled = scaler.transform(input_df)
                
                # Make prediction
                prediction = model.predict(input_scaled)[0]
                probability = model.predict_proba(input_scaled)[0][1]
                
                # Display results
                st.markdown("---")
                st.markdown("## 🎯 Prediction Results")
                
                if prediction == 1:
                    st.markdown(f"""
                    <div class="prediction-high-risk">
                        <h3>⚠️ HIGH RISK OF CHURN</h3>
                        <p><strong>Churn Probability:</strong> {probability:.1%}</p>
                        <p>This customer is likely to churn. Consider implementing retention strategies.</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="prediction-low-risk">
                        <h3>✅ LOW RISK OF CHURN</h3>
                        <p><strong>Churn Probability:</strong> {probability:.1%}</p>
                        <p>This customer is likely to stay. Continue providing excellent service.</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Risk gauge
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = probability * 100,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "Churn Risk (%)"},
                    gauge = {
                        'axis': {'range': [None, 100]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [0, 30], 'color': "lightgreen"},
                            {'range': [30, 70], 'color': "yellow"},
                            {'range': [70, 100], 'color': "red"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 90
                        }
                    }
                ))
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
    
    else:  # Batch Prediction
        st.markdown("### 📁 Upload CSV File for Batch Predictions")
        
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        
        if uploaded_file is not None:
            try:
                # Read the uploaded file
                batch_data = pd.read_csv(uploaded_file)
                
                st.write("📋 Data Preview:")
                st.dataframe(batch_data.head())
                
                if st.button("Generate Predictions"):
                    # Process batch data (simplified version)
                    # Note: In a real application, you'd need to apply all the same preprocessing
                    # as done in the training script
                    
                    st.info("Processing batch predictions... (This is a simplified version)")
                    st.write("For full batch processing, ensure your CSV has all required features and preprocessing.")
                    
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")

elif page == "📊 Analytics":
    st.markdown("## 📊 Model Analytics & Insights")
    
    # Feature importance (if Random Forest)
    if model_name == "Random Forest":
        st.subheader("🎯 Feature Importance")
        
        feature_importance = pd.DataFrame({
            'Feature': feature_names,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        fig = px.bar(
            feature_importance.head(15), 
            x='Importance', 
            y='Feature',
            orientation='h',
            title="Top 15 Most Important Features"
        )
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("### 📈 Key Insights:")
        top_features = feature_importance.head(5)['Feature'].tolist()
        st.write(f"**Most Important Features:** {', '.join(top_features)}")
    
    # Model performance metrics
    st.subheader("🎯 Model Performance")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Performance metrics
        st.markdown("#### Performance Metrics")
        metrics_df = pd.DataFrame({
            'Metric': ['Accuracy', 'ROC AUC'],
            'Score': [performance_metrics['accuracy'], performance_metrics['roc_auc']]
        })
        
        fig = px.bar(metrics_df, x='Metric', y='Score', 
                     title="Model Performance Scores",
                     color='Score', color_continuous_scale='viridis')
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Risk distribution simulation
        st.markdown("#### Risk Distribution Simulation")
        np.random.seed(42)
        simulated_risks = np.random.beta(2, 5, 1000)  # Simulate typical customer risk distribution
        
        fig = px.histogram(x=simulated_risks, nbins=30,
                          title="Simulated Customer Risk Distribution",
                          labels={'x': 'Churn Risk', 'y': 'Number of Customers'})
        fig.update_layout(height=400)
        st.plotly_
