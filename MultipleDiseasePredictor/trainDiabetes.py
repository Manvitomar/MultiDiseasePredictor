import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import pickle

# Load dataset
data = pd.read_csv("MultipleDiseasePredictor/datasets/diabetesDisease.csv")

# Features and target
X = data.drop('Outcome', axis=1)
y = data['Outcome']

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create individual models (all support predict_proba)
model1 = RandomForestClassifier()
model2 = LogisticRegression(max_iter=1000)
model3 = KNeighborsClassifier()

# Combine models using VotingClassifier with soft voting (for predict_proba)
voting_model = VotingClassifier(estimators=[
    ('rf', model1),
    ('lr', model2),
    ('knn', model3)
], voting='soft')  # ✅ changed to 'soft'

# Train the combined model
voting_model.fit(X_train, y_train)

# Test Accuracy (in %)
y_pred = voting_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred) * 100
print(f"Ensemble Diabetes Model Accuracy: {accuracy:.2f}%")

# Save the model
pickle.dump(voting_model, open("MultipleDiseasePredictor/models/diabetes.pkl", "wb"))  # ✅ saved updated model
