# Enhanced Customer Churn Prediction - Full Project with SMOTE and Hyperparameter Tuning
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
from imblearn.over_sampling import SMOTE
import pickle
import warnings
warnings.filterwarnings('ignore')

print("Starting Enhanced Customer Churn Prediction Model Training...")

# Step 1: Load and inspect data
data = pd.read_csv(r"c:data\WA_Fn-UseC_-Telco-Customer-Churn.csv")
print("Dataset shape:", data.shape)
print("\nFirst 5 rows:")
print(data.head())

# Step 2: Data cleaning and preprocessing
print("\n=== DATA CLEANING ===")
print("Missing values:")
print(data.isnull().sum())

# Handle TotalCharges column
data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')
print(f"TotalCharges missing values: {data['TotalCharges'].isnull().sum()}")
data['TotalCharges'].fillna(data['TotalCharges'].median(), inplace=True)

# Drop customerID as it's not useful for prediction
data.drop('customerID', axis=1, inplace=True)

# Convert target variable to binary
data['Churn'] = data['Churn'].apply(lambda x: 1 if x == 'Yes' else 0)

print(f"Churn distribution:\n{data['Churn'].value_counts()}")
print(f"Churn rate: {data['Churn'].mean():.2%}")

# Step 3: Feature Engineering
print("\n=== FEATURE ENGINEERING ===")

# Create tenure groups
data['tenure_group'] = pd.cut(data['tenure'], 
                              bins=[0,12,24,48,60,72],
                              labels=['0-12','13-24','25-48','49-60','61-72'])

# Create monthly charges groups
data['monthly_charges_group'] = pd.cut(data['MonthlyCharges'], 
                                       bins=4, 
                                       labels=['Low','Medium-Low','Medium-High','High'])

# Create total charges per tenure ratio
data['charges_per_tenure'] = data['TotalCharges'] / (data['tenure'] + 1)  # +1 to avoid division by zero

# Encode categorical variables
cat_cols = data.select_dtypes(include=['object', 'category']).columns
le_dict = {}

for col in cat_cols:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    le_dict[col] = le

print(f"Categorical columns encoded: {list(cat_cols)}")
print("Dataset shape after preprocessing:", data.shape)

# Step 4: EDA and Visualizations
print("\n=== EXPLORATORY DATA ANALYSIS ===")

plt.figure(figsize=(15, 12))

# Churn distribution
plt.subplot(2, 3, 1)
sns.countplot(data=data, x='Churn')
plt.title("Churn Distribution")

# Correlation heatmap (top features)
plt.subplot(2, 3, 2)
corr_with_target = data.corr()['Churn'].abs().sort_values(ascending=False)[1:11]
top_features = corr_with_target.index
corr_matrix = data[list(top_features) + ['Churn']].corr()
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm')
plt.title("Top 10 Features Correlation with Churn")

# Tenure distribution by churn
plt.subplot(2, 3, 3)
sns.boxplot(data=data, x='Churn', y='tenure')
plt.title('Tenure by Churn')

# Monthly charges by churn
plt.subplot(2, 3, 4)
sns.boxplot(data=data, x='Churn', y='MonthlyCharges')
plt.title('Monthly Charges by Churn')

# Contract type by churn
plt.subplot(2, 3, 5)
contract_churn = pd.crosstab(data['Contract'], data['Churn'], normalize='index')
contract_churn.plot(kind='bar', ax=plt.gca())
plt.title('Churn Rate by Contract Type')
plt.legend(['No Churn', 'Churn'])

plt.tight_layout()
plt.savefig('eda_plots.png', dpi=300, bbox_inches='tight')
plt.show()

# Step 5: Prepare data for modeling
print("\n=== DATA PREPARATION ===")
X = data.drop('Churn', axis=1)
y = data['Churn']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
                                                    random_state=42, stratify=y)

print(f"Training set shape: {X_train.shape}")
print(f"Test set shape: {X_test.shape}")
print(f"Training set churn rate: {y_train.mean():.2%}")

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 6: Apply SMOTE for balancing
print("\n=== APPLYING SMOTE ===")
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)

print(f"Before SMOTE - Class distribution: {np.bincount(y_train)}")
print(f"After SMOTE - Class distribution: {np.bincount(y_train_balanced)}")

# Step 7: Hyperparameter Tuning
print("\n=== HYPERPARAMETER TUNING ===")

# Logistic Regression hyperparameter tuning
print("Tuning Logistic Regression...")
lr_params = {
    'C': [0.01, 0.1, 1, 10, 100],
    'penalty': ['l1', 'l2'],
    'solver': ['liblinear', 'saga']
}

lr_grid = GridSearchCV(LogisticRegression(random_state=42, max_iter=1000),
                       lr_params, cv=5, scoring='roc_auc', n_jobs=-1)
lr_grid.fit(X_train_balanced, y_train_balanced)

print(f"Best LR parameters: {lr_grid.best_params_}")
print(f"Best LR CV score: {lr_grid.best_score_:.4f}")

# Random Forest hyperparameter tuning
print("Tuning Random Forest...")
rf_params = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2']
}

rf_grid = GridSearchCV(RandomForestClassifier(random_state=42),
                       rf_params, cv=5, scoring='roc_auc', n_jobs=-1)
rf_grid.fit(X_train_balanced, y_train_balanced)

print(f"Best RF parameters: {rf_grid.best_params_}")
print(f"Best RF CV score: {rf_grid.best_score_:.4f}")

# Step 8: Train final models
print("\n=== TRAINING FINAL MODELS ===")

# Best Logistic Regression
best_lr = lr_grid.best_estimator_
y_pred_lr = best_lr.predict(X_test_scaled)
y_pred_lr_proba = best_lr.predict_proba(X_test_scaled)[:, 1]

# Best Random Forest
best_rf = rf_grid.best_estimator_
y_pred_rf = best_rf.predict(X_test_scaled)
y_pred_rf_proba = best_rf.predict_proba(X_test_scaled)[:, 1]

# Step 9: Model Evaluation
print("\n=== MODEL EVALUATION ===")

def evaluate_model(y_true, y_pred, y_pred_proba, model_name):
    print(f"\n{model_name} Results:")
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
    print(f"ROC AUC: {roc_auc_score(y_true, y_pred_proba):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))
    return accuracy_score(y_true, y_pred), roc_auc_score(y_true, y_pred_proba)

lr_acc, lr_auc = evaluate_model(y_test, y_pred_lr, y_pred_lr_proba, "Logistic Regression")
rf_acc, rf_auc = evaluate_model(y_test, y_pred_rf, y_pred_rf_proba, "Random Forest")

# Step 10: Visualizations
plt.figure(figsize=(15, 10))

# Confusion matrices
plt.subplot(2, 3, 1)
cm_lr = confusion_matrix(y_test, y_pred_lr)
sns.heatmap(cm_lr, annot=True, fmt='d', cmap='Blues')
plt.title('Logistic Regression Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')

plt.subplot(2, 3, 2)
cm_rf = confusion_matrix(y_test, y_pred_rf)
sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Blues')
plt.title('Random Forest Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')

# ROC curves
plt.subplot(2, 3, 3)
fpr_lr, tpr_lr, _ = roc_curve(y_test, y_pred_lr_proba)
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_pred_rf_proba)

plt.plot(fpr_lr, tpr_lr, label=f'Logistic Regression (AUC = {lr_auc:.3f})')
plt.plot(fpr_rf, tpr_rf, label=f'Random Forest (AUC = {rf_auc:.3f})')
plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves')
plt.legend()

# Feature importance for Random Forest
plt.subplot(2, 3, 4)
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': best_rf.feature_importances_
}).sort_values('importance', ascending=False).head(10)

sns.barplot(data=feature_importance, y='feature', x='importance')
plt.title('Top 10 Feature Importances (Random Forest)')

# Model comparison
plt.subplot(2, 3, 5)
models = ['Logistic Regression', 'Random Forest']
accuracies = [lr_acc, rf_acc]
aucs = [lr_auc, rf_auc]

x = np.arange(len(models))
width = 0.35

plt.bar(x - width/2, accuracies, width, label='Accuracy', alpha=0.8)
plt.bar(x + width/2, aucs, width, label='ROC AUC', alpha=0.8)

plt.xlabel('Models')
plt.ylabel('Score')
plt.title('Model Performance Comparison')
plt.xticks(x, models)
plt.legend()
plt.ylim(0, 1)

for i, (acc, auc) in enumerate(zip(accuracies, aucs)):
    plt.text(i - width/2, acc + 0.01, f'{acc:.3f}', ha='center')
    plt.text(i + width/2, auc + 0.01, f'{auc:.3f}', ha='center')

plt.tight_layout()
plt.savefig('model_evaluation.png', dpi=300, bbox_inches='tight')
plt.show()

# Step 11: Save models and preprocessing objects
print("\n=== SAVING MODELS ===")

# Choose the best model (highest ROC AUC)
if rf_auc > lr_auc:
    best_model = best_rf
    best_model_name = "Random Forest"
    print(f"Random Forest selected as best model (ROC AUC: {rf_auc:.4f})")
else:
    best_model = best_lr
    best_model_name = "Logistic Regression"
    print(f"Logistic Regression selected as best model (ROC AUC: {lr_auc:.4f})")

# Save all necessary objects for deployment
models_dict = {
    'best_model': best_model,
    'scaler': scaler,
    'label_encoders': le_dict,
    'feature_names': list(X.columns),
    'model_name': best_model_name,
    'performance_metrics': {
        'accuracy': rf_acc if rf_auc > lr_auc else lr_acc,
        'roc_auc': rf_auc if rf_auc > lr_auc else lr_auc
    }
}

with open('churn_prediction_model.pkl', 'wb') as f:
    pickle.dump(models_dict, f)

print("Model and preprocessing objects saved successfully!")
print("\nFeature names for deployment:")
print(list(X.columns))

# Print final summary
print(f"\n{'='*50}")
print("TRAINING SUMMARY")
print(f"{'='*50}")
print(f"Dataset size: {data.shape[0]} customers")
print(f"Number of features: {X.shape[1]}")
print(f"Churn rate: {y.mean():.2%}")
print(f"Best model: {best_model_name}")
print(f"Best model accuracy: {models_dict['performance_metrics']['accuracy']:.4f}")
print(f"Best model ROC AUC: {models_dict['performance_metrics']['roc_auc']:.4f}")
print("Model ready for deployment!")
