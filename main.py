import os
import io
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image
from pypdf import PdfReader
from google import genai

app = FastAPI(title="AI Nutrition API")

# Enable CORS so Netlify frontend can talk to backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust to your Netlify domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

API_KEY = os.environ.get("GEMINI_API_KEY")
if API_KEY:
    client = genai.Client(api_key=API_KEY)
else:
    client = None

def calculate_bmr_and_tdee(gender: str, weight: float, height: float, age: int, activity: str):
    if gender.lower() == "male":
        bmr = (10 * weight) + (6.25 * height) - (5 * age) + 5
    else:
        bmr = (10 * weight) + (6.25 * height) - (5 * age) - 161

    multipliers = {
        "sedentary": 1.2,
        "lightly_active": 1.375,
        "moderately_active": 1.55,
        "very_active": 1.725,
        "extra_active": 1.9,
    }
    tdee = bmr * multipliers.get(activity, 1.2)
    return round(bmr), round(tdee)

class BMRRequest(BaseModel):
    gender: str
    weight: float
    height: float
    age: int
    activity: str

@app.post("/api/bmr")
def get_bmr(data: BMRRequest):
    bmr, tdee = calculate_bmr_and_tdee(data.gender, data.weight, data.height, data.age, data.activity)
    return {"bmr": bmr, "tdee": tdee}

@app.post("/api/analyze-lab")
async def analyze_lab(file: UploadFile = File(...)):
    if not client:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY not set on server.")

    contents = []
    file_bytes = await file.read()

    if file.filename.endswith(".pdf"):
        pdf_reader = PdfReader(io.BytesIO(file_bytes))
        extracted_text = ""
        for page in pdf_reader.pages:
            text = page.extract_text()
            if text:
                extracted_text += text
        contents.append(f"Analyze this pathology lab report text and summarize nutrient status:\n\n{extracted_text}")
    else:
        try:
            img = Image.open(io.BytesIO(file_bytes))
            contents = ["Analyze this path lab report image and summarize nutrient status.", img]
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid image file: {str(e)}")

    prompt = """
    You are a clinical nutrition specialist reviewing a medical blood/lab report.
    Extract and summarize:
    1. Key Deficiencies & Low Markers (Vitamin D, Vitamin B12, Iron, Hb, Calcium, etc.)
    2. Elevated Markers (Cholesterol, Glucose, Uric Acid, etc.)
    3. Clinical Interpretation
    4. Key Nutrients/Minerals needing dietary attention.
    Keep the response concise and clearly structured in markdown format.
    """
    contents.insert(0, prompt)

    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=contents
    )
    return {"analysis": response.text}

class PlanRequest(BaseModel):
    age: int
    gender: str
    height: float
    weight: float
    activity: str
    bmr: float
    tdee: float
    goal: str
    diet_pref: str
    lab_summary: str = ""
    notes: str = ""

@app.post("/api/generate-plan")
def generate_plan(data: PlanRequest):
    if not client:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY not set on server.")

    prompt = f"""
    Act as an expert Clinical Nutritionist and Dietitian.
    Design a personalized 1-day nutrition and lifestyle plan based on the following patient profile:

    **Biometric Profile:**
    - Age: {data.age} | Gender: {data.gender}
    - Height: {data.height} cm | Weight: {data.weight} kg
    - Activity Level: {data.activity}
    - BMR: {data.bmr} kcal | TDEE: {data.tdee} kcal
    - Goal: {data.goal}
    - Dietary Preference: {data.diet_pref}
    - Notes/Allergies: {data.notes}

    **Path Lab Report Summary:**
    {data.lab_summary if data.lab_summary else 'No lab report provided.'}

    **Deliverables Required:**
    1. Daily Calorie & Macro Distribution (Calories, Protein, Carbs, Fats)
    2. Targeted Nutrient Foods matching deficiencies
    3. Daily Sample Meal Plan (Breakfast, Lunch, Evening Snack, Dinner)
    4. Lifestyle & Supplementation Guidance
    """

    response = client.models.generate_content(
        model="gemini-3.6-flash",
        contents=prompt
    )
    return {"plan": response.text}
