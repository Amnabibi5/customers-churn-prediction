# Customer Churn Prediction - Full Project

# Step 1: Import libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Step 2: Load data
data = pd.read_csv(r"c:\Users\khan\Desktop\final_project\WA_Fn-UseC_-Telco-Customer-Churn.csv")
print(data.head())

# Step 3: Data inspection and cleaning
print(data.info())
print(data.isnull().sum())
data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')
data['TotalCharges'].fillna(data['TotalCharges'].median(), inplace=True)
data.drop('customerID', axis=1, inplace=True)
data['Churn'] = data['Churn'].apply(lambda x: 1 if x == 'Yes' else 0)

# Step 4: Encoding categorical variables
cat_cols = data.select_dtypes(include=['object']).columns
for col in cat_cols:
    data[col] = LabelEncoder().fit_transform(data[col])
print(data.head())

# Step 5: EDA - Visualizations
sns.countplot(x='Churn', data=data)
plt.title("Churn Distribution")
plt.show()

plt.figure(figsize=(15,10))
sns.heatmap(data.corr(), annot=True, fmt='.2f', cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

sns.boxplot(x='Churn', y='tenure', data=data)
plt.title('Tenure by Churn')
plt.show()

# Step 6: Feature engineering (optional)
data['tenure_group'] = pd.cut(data['tenure'], bins=[0,12,24,48,60,72],
                              labels=['0-12','13-24','25-48','49-60','61-72'])
data['tenure_group'] = LabelEncoder().fit_transform(data['tenure_group'])

# Step 7: Prepare data for modeling
X = data.drop('Churn', axis=1)
y = data['Churn']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 8: Model training - Logistic Regression
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train, y_train)
y_pred_lr = lr_model.predict(X_test)
print("Logistic Regression Accuracy:", accuracy_score(y_test, y_pred_lr))
print(classification_report(y_test, y_pred_lr))

# Step 9: Model training - Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)
print("Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf))

# Step 10: Confusion Matrix for Random Forest
cm = confusion_matrix(y_test, y_pred_rf)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Random Forest Confusion Matrix')
plt.show()

# Step 11: Save model for deployment (optional)
import pickle
with open('rf_model.pkl', 'wb') as f:
    pickle.dump(rf_model, f)



