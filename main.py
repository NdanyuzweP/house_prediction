from fastapi import FastAPI
from pydantic import BaseModel
import tensorflow as tf
import numpy as np

# Load the model
model = tf.keras.models.load_model("models/house_model.h5")

# Initialize FastAPI app
app = FastAPI()

# Define the input data model
class HouseData(BaseModel):
    squareMeters: float
    numberOfRooms: int
    hasYard: int
    hasPool: int
    floors: int
    numPrevOwners: int
    made: float
    isNewBuilt: int
    hasStormProtector: int
    basement: float
    attic: float
    garage: float
    hasStorageRoom: int
    hasGuestRoom: int
    price: float

# Define prediction endpoint
@app.post("/predict")
def predict(data: HouseData):
    # Convert input data to numpy array
    input_data = np.array([[ 
        data.squareMeters, data.numberOfRooms, data.hasYard, 
        data.hasPool, data.floors, data.numPrevOwners, 
        data.made, data.isNewBuilt, data.hasStormProtector, 
        data.basement, data.attic, data.garage, 
        data.hasStorageRoom, data.hasGuestRoom, data.price
    ]])
    
    # Make prediction
    prediction = model.predict(input_data)
    
    # Convert prediction to class label (0 or 1)
    category = "Luxury" if prediction[0][0] > 1.2 else "Basic"
    
    return {"category": category, "probability": float(prediction[0][0])}
