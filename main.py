import pickle
import numpy as np
import google.generativeai as genai
import base64
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Load the saved ML model
with open("model.pkl", "rb") as file:
    model = pickle.load(file)

# Configure Gemini API
GOOGLE_API_KEY = "AIzaSyDDyCLAu9ypq7f2p2vEtG0RwfvVFg8fLOc"
genai.configure(api_key=GOOGLE_API_KEY)

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

# Define the structure of the extracted data
class ExtractedData(BaseModel):
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

# Function to extract text data from image using Gemini API
import json

import re

def extract_data_from_image(image_bytes: bytes):
    # Convert image to base64
    encoded_image = base64.b64encode(image_bytes).decode("utf-8")

    # Initialize Gemini model
    model = genai.GenerativeModel(model_name="gemini-1.5-pro")

    # Structured prompt to guide Gemini
    prompt = (
        "Extract the following medical parameters from the image and return them as text:\n"
        "patient_id, age, fsh, lh, glucose_fasting, glucose_pp, insulin_fasting, "
        "insulin_pp, prolactin, testosterone, t3, t4, tsh.\n"
        "Each value should be in the format: 'key: value'.\n"
        "For example:\n"
        "patient_id: 12345\n"
        "age: 45.0\n"
        "fsh: 6.5\n"
        "..."
    )

    # Send request to Gemini
    response = model.generate_content(
        [
            {"mime_type": "image/jpeg", "data": encoded_image},
            prompt,
        ]
    )

    # Extract raw text response
    raw_text = response.text.strip()
    
    # Print response to debug (remove in production)
    print("Gemini Response:\n", raw_text)

    try:
        # Use regex to extract values
        pattern = re.compile(r'(\w+):\s*([\d\.]+)')
        extracted_values = dict(pattern.findall(raw_text))

        # Convert extracted values to proper types
        extracted_data = ExtractedData(
            patient_id=int(extracted_values.get("patient_id", 0)),
            age=float(extracted_values.get("age", 0)),
            fsh=float(extracted_values.get("fsh", 0)),
            lh=float(extracted_values.get("lh", 0)),
            glucose_fasting=float(extracted_values.get("glucose_fasting", 0)),
            glucose_pp=float(extracted_values.get("glucose_pp", 0)),
            insulin_fasting=float(extracted_values.get("insulin_fasting", 0)),
            insulin_pp=float(extracted_values.get("insulin_pp", 0)),
            prolactin=float(extracted_values.get("prolactin", 0)),
            testosterone=float(extracted_values.get("testosterone", 0)),
            t3=float(extracted_values.get("t3", 0)),
            t4=float(extracted_values.get("t4", 0)),
            tsh=float(extracted_values.get("tsh", 0)),
        )

        return extracted_data

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to extract data: {str(e)}")


# POST endpoint to make predictions using an image
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Read image file
    image_bytes = await file.read()

    # Extract data from image using Gemini API
    extracted_data = extract_data_from_image(image_bytes)

    # Convert extracted data to numpy array
    input_array = np.array([[
        extracted_data.age, extracted_data.fsh, extracted_data.lh, extracted_data.glucose_fasting, 
        extracted_data.glucose_pp, extracted_data.insulin_fasting, extracted_data.insulin_pp, 
        extracted_data.prolactin, extracted_data.testosterone, extracted_data.t3, extracted_data.t4, extracted_data.tsh
    ]])

    # Make prediction
    prediction = model.predict(input_array)

    return {"patient_id": extracted_data.patient_id, "prediction": prediction.tolist()}

# Run app locally (for Render, Railway, etc.)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
