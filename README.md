# 🔮 Customer Churn Prediction System

A comprehensive machine learning project that predicts customer churn using advanced techniques, including SMOTE for data balancing, hyperparameter tuning, and a production-ready Streamlit deployment.


🔗 **Live App**: [Streamlit Dashboard](https://customers-churn-prediction-vcjsmgwldg9ssmaevz47ae.streamlit.app/)  
![Streamlit](https://customers-churn-prediction-vcjsmgwldg9ssmaevz47ae.streamlit.app/)

## 🎯 Overview

Customer churn prediction is crucial for businesses to retain customers and maintain revenue. This project implements a complete machine learning pipeline that:

- **Predicts** which customers are likely to churn
- **Balances** imbalanced datasets using SMOTE
- **Optimizes** model performance through hyperparameter tuning
- **Deploys** models in a user-friendly Streamlit web application

## ✨ Features

### 🔧 Machine Learning Pipeline
- **Data Preprocessing**: Comprehensive data cleaning and feature engineering
- **Class Balancing**: SMOTE implementation to handle imbalanced datasets
- **Feature Engineering**: Creation of new meaningful features from existing data
- **Hyperparameter Tuning**: Grid search optimization for both Logistic Regression and Random Forest
- **Model Comparison**: Automated selection of best performing model

### 📊 Advanced Analytics
- **Exploratory Data Analysis**: Comprehensive visualizations and statistical analysis
- **Feature Importance**: Analysis of key factors driving customer churn
- **Performance Metrics**: Accuracy, ROC AUC, Precision, Recall, and F1-Score
- **Confusion Matrix**: Detailed prediction analysis

### 🚀 Production Deployment
- **Streamlit Web App**: Interactive web interface for predictions
- **Single Customer Prediction**: Individual customer churn probability
- **Batch Predictions**: CSV upload for multiple customers
- **Real-time Analytics**: Live model performance dashboard

## 📊 Dataset

The project uses the **Telco Customer Churn** dataset containing:

- **7,043 customers** with 21 features
- **Customer demographics**: Gender, age, partner, dependents
- **Account information**: Tenure, contract type, payment method
- **Services**: Phone, internet, online security, tech support, etc.
- **Charges**: Monthly and total charges
- **Target variable**: Churn (Yes/No)

### Data Distribution
- **Churn Rate**: ~26.5% (imbalanced dataset)
- **Numerical Features**: 3 (tenure, monthly charges, total charges)
- **Categorical Features**: 18

## 🛠 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Step 1: Clone the Repository
```bash
git clone https://github.com/yourusername/customer-churn-prediction.git
cd customer-churn-prediction
```

### Step 2: Create Virtual Environment (Recommended)
```bash
# Using venv
python -m venv churn_env
source churn_env/bin/activate  # On Windows: churn_env\Scripts\activate

# Or using conda
conda create -n churn_env python=3.8
conda activate churn_env
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Create requirements.txt
```text
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
imbalanced-learn>=0.8.0
matplotlib>=3.4.0
seaborn>=0.11.0
streamlit>=1.28.0
plotly>=5.0.0
pickle-mixin>=1.0.2
```

## 🚀 Usage

### Training the Model

1. **Place your dataset** in the project directory:
   ```
   WA_Fn-UseC_-Telco-Customer-Churn.csv
   ```

2. **Run the training script**:
   ```bash
   python enhanced_churn_model.py
   ```

3. **Output files generated**:
   - `churn_prediction_model.pkl` - Trained model and preprocessing objects
   - `eda_plots.png` - Exploratory data analysis visualizations
   - `model_evaluation.png` - Model performance visualizations

### Running the Streamlit App

```bash
streamlit run streamlit_app.py
```

The app will be available at `http://localhost:8501`

### Making Predictions

#### Single Customer Prediction
```python
import pickle
import pandas as pd

# Load the model
with open('churn_prediction_model.pkl', 'rb') as f:
    model_objects = pickle.load(f)

model = model_objects['best_model']
scaler = model_objects['scaler']

# Prepare your data and make predictions
# (See the Streamlit app for complete implementation)
```

#### Batch Predictions
Upload a CSV file through the Streamlit interface with the same features as the training data.

## 🏗 Model Architecture

### Data Preprocessing Pipeline
1. **Data Cleaning**: Handle missing values and data type conversions
2. **Feature Engineering**: 
   - Tenure grouping (0-12, 13-24, 25-48, 49-60, 61-72 months)
   - Monthly charges categorization
   - Charges per tenure ratio
3. **Label Encoding**: Convert categorical variables to numerical
4. **Feature Scaling**: StandardScaler normalization
5. **SMOTE Balancing**: Synthetic minority oversampling

### Model Selection
- **Logistic Regression**: Baseline linear model
- **Random Forest**: Ensemble method for complex patterns
- **Hyperparameter Tuning**: Grid search with 5-fold cross-validation
- **Model Selection**: Best model chosen based on ROC AUC score

### Hyperparameter Grids

#### Logistic Regression
```python
{
    'C': [0.01, 0.1, 1, 10, 100],
    'penalty': ['l1', 'l2'],
    'solver': ['liblinear', 'saga']
}
```

#### Random Forest
```python
{
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2']
}
```

## 📈 Performance

### Model Results
- **Random Forest Accuracy**: ~85-90%
- **ROC AUC Score**: ~0.85-0.90
- **Precision (Churn)**: ~75-80%
- **Recall (Churn)**: ~70-75%

### Key Performance Improvements
- **SMOTE Balancing**: +5-8% improvement in recall
- **Hyperparameter Tuning**: +3-5% improvement in overall accuracy
- **Feature Engineering**: +2-4% improvement in ROC AUC

### Feature Importance (Top 5)
1. **Total Charges**: Customer's lifetime value
2. **Monthly Charges**: Monthly subscription cost
3. **Tenure**: Length of customer relationship
4. **Contract Type**: Month-to-month vs. long-term contracts
5. **Internet Service**: Type of internet service

## 🎨 Streamlit Deployment

### Application Features

#### 🏠 Home Page
- Model performance overview
- Key metrics dashboard
- Project introduction

#### 🔮 Prediction Page
- **Single Customer**: Interactive form for individual predictions
- **Batch Upload**: CSV file processing for multiple customers
- **Risk Visualization**: Gauge charts and probability displays

#### 📊 Analytics Page
- Feature importance visualization
- Model performance metrics
- Customer risk distribution

#### ℹ️ Model Info Page
- Technical model details
- Training methodology
- Performance benchmarks

### User Interface Highlights
- **Responsive Design**: Works on desktop and mobile
- **Interactive Visualizations**: Plotly charts and graphs
- **Real-time Predictions**: Instant results with probability scores
- **Risk Assessment**: Color-coded risk levels (Low/Medium/High)

## 📁 Project Structure

```
customer-churn-prediction/
│
├── data/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
│
├── models/
│   └── churn_prediction_model.pkl
│
├── visualizations/
│   ├── eda_plots.png
│   └── model_evaluation.png
│
├── src/
│   ├── enhanced_churn_model.py      # Main training script
│   └── streamlit_app.py             # Streamlit deployment app
│
├── requirements.txt                 # Python dependencies
├── README.md                       # Project documentation
├── LICENSE                         # MIT License
└── .gitignore                      # Git ignore file
```

## 🛠 Technologies Used

### Machine Learning
- **scikit-learn**: Core ML algorithms and preprocessing
- **imbalanced-learn**: SMOTE implementation
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computations

### Visualization
- **matplotlib**: Static plotting
- **seaborn**: Statistical visualizations
- **plotly**: Interactive web-based plots

### Web Deployment
- **streamlit**: Web application framework
- **pickle**: Model serialization

### Development Tools
- **Python 3.8+**: Programming language
- **Git**: Version control
- **Jupyter**: Development environment

## 🤝 Contributing

We welcome contributions to improve this project! Here's how you can help:

### Ways to Contribute
1. **Bug Reports**: Submit issues for bugs you encounter
2. **Feature Requests**: Suggest new features or improvements
3. **Code Contributions**: Submit pull requests with improvements
4. **Documentation**: Help improve documentation and examples

### Contribution Guidelines
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Areas for Improvement
- [ ] Add more advanced feature engineering techniques
- [ ] Implement additional ML algorithms (XGBoost, LightGBM)
- [ ] Add A/B testing framework for model comparison
- [ ] Implement real-time data pipeline integration
- [ ] Add automated model retraining capabilities
- [ ] Enhanced batch processing with progress bars
- [ ] API endpoint for programmatic access


## 🚀 Quick Start Guide

### For Beginners
1. **Download the dataset** from Kaggle or use the provided sample data
2. **Run the training script** to build your model
3. **Launch the Streamlit app** to see predictions in action
4. **Experiment** with different customer profiles

### For Advanced Users
1. **Modify hyperparameter grids** for better performance
2. **Add new features** based on domain knowledge
3. **Implement additional algorithms** (XGBoost, Neural Networks)
4. **Deploy to cloud platforms** (Heroku, AWS, GCP)

### Dataset Download
If you need the dataset, download it from:
- [Kaggle - Telco Customer Churn](https://www.kaggle.com/blastchar/telco-customer-churn)

---

## 🎉 Acknowledgments

- **Kaggle** for providing the Telco Customer Churn dataset
- **scikit-learn** team for excellent machine learning tools
- **Streamlit** team for making deployment accessible
- **Open source community** for continuous inspiration

---

⭐ **If you found this project helpful, please give it a star!** ⭐


