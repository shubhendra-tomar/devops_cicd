from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np

# Load the saved model, scaler, and encoders
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
label_encoders = joblib.load("label_encoders.pkl")

# Define the FastAPI app
app = FastAPI()

# Define a request model using Pydantic
class ShipmentFeatures(BaseModel):
    Warehouse_block: str
    Mode_of_Shipment: str
    Customer_care_calls: int
    Customer_rating: int
    Cost_of_the_Product: float
    Prior_purchases: int
    Product_importance: str
    Gender: str
    Discount_offered: float
    Weight_in_gms: float

@app.post("/predict")
def predict(features: ShipmentFeatures):
    try:
        # Encode categorical features
        categorical_features = [
            features.Warehouse_block,
            features.Mode_of_Shipment,
            features.Product_importance,
            features.Gender,
        ]
        encoded_features = [
            label_encoders["Warehouse_block"].transform([categorical_features[0]])[0],
            label_encoders["Mode_of_Shipment"].transform([categorical_features[1]])[0],
            label_encoders["Product_importance"].transform([categorical_features[2]])[0],
            label_encoders["Gender"].transform([categorical_features[3]])[0],
        ]

        # Combine all features in the correct order
        numerical_features = [
            features.Customer_care_calls,
            features.Customer_rating,
            features.Cost_of_the_Product,
            features.Prior_purchases,
            features.Discount_offered,
            features.Weight_in_gms,
        ]
        input_data = np.array([encoded_features + numerical_features])

        # Verify feature shape matches training data
        if input_data.shape[1] != scaler.n_features_in_:
            raise ValueError(f"Expected {scaler.n_features_in_} features, but got {input_data.shape[1]}.")

        # Scale the input data
        scaled_data = scaler.transform(input_data)

        # Predict using the loaded model
        prediction = model.predict(scaled_data)
        prediction_proba = model.predict_proba(scaled_data)

        # Return the prediction and probability
        return {
            "prediction": int(prediction[0]),
            "probability": prediction_proba[0].tolist()
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error in prediction: {str(e)}")
