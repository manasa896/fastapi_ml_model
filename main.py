from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
# Load trained model
model = joblib.load("model.pkl")
# Create FastAPI app
app = FastAPI(title="Student Result Prediction API")
# Input schema
class StudentInput(BaseModel):
    Math: float
    Science: float
    English: float

@app.post("/predict")
def predict_result(student: StudentInput):
    data = pd.DataFrame([{
    "Math": student.Math,
    "Science": student.Science,
    "English": student.English
    }])
    prediction = model.predict(data)[0]
    result = "Pass" if prediction == 1 else "Fail"
    return {
        "prediction": result,
        "code": int(prediction)
    }
