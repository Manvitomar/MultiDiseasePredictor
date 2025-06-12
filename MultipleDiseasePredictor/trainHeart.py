import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import pickle

# Load the dataset
data = pd.read_csv("MultipleDiseasePredictor/datasets/heartDisease.csv")

# Separate features and label
X = data.drop('target', axis=1)
y = data['target']

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define individual models
rf = RandomForestClassifier()
lr = LogisticRegression(max_iter=1000)
knn = KNeighborsClassifier()

# Combine models using soft voting for confidence output
voting_model = VotingClassifier(estimators=[
    ('rf', rf),
    ('lr', lr),
    ('knn', knn)
], voting='soft')  # ✅ Soft voting for predict_proba

# Train the ensemble model
voting_model.fit(X_train, y_train)

# Evaluate model and print accuracy in percentage
y_pred = voting_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred) * 100
print(f"Ensemble Heart Model Accuracy: {accuracy:.2f}%")

# Save model using pickle
pickle.dump(voting_model, open("MultipleDiseasePredictor/models/heart.pkl", "wb"))  # ✅ Save updated model
