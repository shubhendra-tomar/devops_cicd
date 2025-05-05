import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib

# Load dataset
data = pd.read_csv('Train.csv')

# Feature and Target Split
X = data.drop(columns=["ID", "Reached.on.Time_Y.N"])
y = data['Reached.on.Time_Y.N']

# Encode categorical features
categorical_columns = ['Warehouse_block', 'Mode_of_Shipment', 'Product_importance', 'Gender']
label_encoders = {}

for column in categorical_columns:
    le = LabelEncoder()
    X[column] = le.fit_transform(X[column])
    label_encoders[column] = le  # Save encoder for future use

# Handle missing values if any (optional: fill with mean)
X.fillna(X.mean(), inplace=True)

# Scale numerical features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# Train a Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=40)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Save the model, scaler, and label encoders
joblib.dump(model, 'model.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(label_encoders, 'label_encoders.pkl')

print("Model, scaler, and label encoders saved as 'model.pkl', 'scaler.pkl', and 'label_encoders.pkl'")
