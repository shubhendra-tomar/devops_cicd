from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_predict_endpoint():
    payload = {
        "Warehouse_block": "A",
        "Mode_of_Shipment": "Flight",
        "Customer_care_calls": 4,
        "Customer_rating": 3,
        "Cost_of_the_Product": 200,
        "Prior_purchases": 3,
        "Product_importance": "low",
        "Gender": "M",
        "Discount_offered": 20,
        "Weight_in_gms": 2000,
        "Competitors_Entered": 5,
        "Marketing_Spend": 1000.50,
        "Price": 250.75,
        "Inventory_Level": 2
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    assert "prediction" in response.json()
    assert "probability" in response.json()
