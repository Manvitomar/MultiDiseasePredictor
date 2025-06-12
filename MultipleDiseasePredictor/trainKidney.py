import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
import pickle

# Load dataset
data = pd.read_csv("MultipleDiseasePredictor/datasets/kidneyDisease.csv")

# Encode categorical columns
le = LabelEncoder()
for column in data.columns:
    if data[column].dtype == 'object':
        data[column] = le.fit_transform(data[column].astype(str))

# Prepare features and labels
X = data.drop('classification', axis=1)
y = data['classification']

# Map classification labels to 0 and 1 if necessary
if y.dtype == 'object':
    y = y.map({'ckd': 1, 'notckd': 0})

# Handle missing values using mean imputation
imputer = SimpleImputer(strategy='mean')
X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define base models
rf = RandomForestClassifier()
lr = LogisticRegression(max_iter=1000)
knn = KNeighborsClassifier()

# Combine using soft voting
voting_model = VotingClassifier(estimators=[
    ('rf', rf),
    ('lr', lr),
    ('knn', knn)
], voting='soft')  # ✅ soft for confidence %

# Train model
voting_model.fit(X_train, y_train)

# Evaluate accuracy and print in percentage
y_pred = voting_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred) * 100
print(f"Ensemble Kidney Model Accuracy: {accuracy:.2f}%")

# Save model
pickle.dump(voting_model, open("MultipleDiseasePredictor/models/kidney.pkl", "wb"))
