import pickle
import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Load the saved ML model
with open("model.pkl", "rb") as file:
    model = pickle.load(file)

# Initialize FastAPI app
app = FastAPI()

# Enable CORS (Allow frontend & external access)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all domains (change for production)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define request model (data input format)
class InputData(BaseModel):
    patient_id: int
    age: float
    fsh: float
    lh: float
    glucose_fasting: float
    glucose_pp: float
    insulin_fasting: float
    insulin_pp: float
    prolactin: float
    testosterone: float
    t3: float
    t4: float
    tsh: float

# Root endpoint (to check if API is running)
@app.get("/")
def home():
    return {"message": "FastAPI is running!"}

# POST endpoint to make predictions
@app.post("/predict")
def predict(data: InputData):
    # Convert input data to numpy array
    input_array = np.array([[
        data.age, data.fsh, data.lh, data.glucose_fasting, 
        data.glucose_pp, data.insulin_fasting, data.insulin_pp, 
        data.prolactin, data.testosterone, data.t3, data.t4, data.tsh
    ]])

    # Make prediction
    prediction = model.predict(input_array)
    
    return {"patient_id": data.patient_id, "prediction": prediction.tolist()}

# Run app locally (for Render, Railway, etc.)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
