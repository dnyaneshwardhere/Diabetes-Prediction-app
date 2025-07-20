# model_rf_cleaned.py

# Heading: Import libraries
import pandas as pd
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Heading: Load cleaned dataset
df = pd.read_csv("diabetes_clean_rf.csv")

# Heading: Features and target
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

# Heading: Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Heading: Train Random Forest model
model = RandomForestClassifier(n_estimators=200, max_depth=8, class_weight="balanced", random_state=42)
model.fit(X_train, y_train)

# Heading: Evaluate the model
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Heading: Confusion matrix plot
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()

# Heading: Feature importance plot
importances = pd.DataFrame({
    'Feature': X.columns,
    'Importance': model.feature_importances_
}).sort_values(by='Importance', ascending=False)

sns.barplot(data=importances, x="Importance", y="Feature", palette="viridis")
plt.title("Feature Importance")
plt.tight_layout()
plt.show()

# Heading: Save the model
joblib.dump(model, "rf_model.pkl")
print("Random Forest model saved as 'rf_model.pkl'")
