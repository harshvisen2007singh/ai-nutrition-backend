import os
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict
from google import genai
from google.genai import types

app = FastAPI(title="AI Personalised Nutrition Suite API")

# Enable CORS for Netlify Frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize GenAI Client using GEMINI_API_KEY environment variable
api_key = os.environ.get("GEMINI_API_KEY")
client = genai.Client(api_key=api_key) if api_key else None


class BiometricsInput(BaseModel):
    model_config = ConfigDict(extra="ignore")  # Ignores extra fields from frontend without throwing 422 error
    age: float
    gender: str
    height: float
    weight: float
    activity: str


@app.get("/")
def read_root():
    return {"status": "ok", "message": "Backend API is live!"}


@app.post("/analyze_lab_report")
@app.post("/analyze-report")
async def analyze_lab_report(file: UploadFile = File(...)):
    if not client:
        raise HTTPException(
            status_code=500, detail="GEMINI_API_KEY environment variable is not set on Render."
        )

    try:
        contents = await file.read()

        # Determine MIME type
        mime_type = file.content_type
        if not mime_type or mime_type == "application/octet-stream":
            filename = file.filename.lower() if file.filename else ""
            if filename.endswith(".pdf"):
                mime_type = "application/pdf"
            elif filename.endswith((".jpg", ".jpeg")):
                mime_type = "image/jpeg"
            elif filename.endswith(".png"):
                mime_type = "image/png"
            else:
                mime_type = "image/jpeg"

        prompt = """
        You are an expert clinical nutrition AI assistant. Perform a comprehensive analysis of the attached path lab blood report.

        Please render your response in clear, beautifully formatted Markdown with the following sections:

        ### 🩺 1. Extracted Lab Parameters & Biomarkers
        - List all key tested parameters detected in the document (e.g., Hemoglobin, Vitamin D3, Vitamin B12, TSH/Thyroid, Blood Glucose, Lipid Profile, Liver/Kidney Markers).
        - Specify the patient's observed test value vs standard reference ranges.

        ### ⚠️ 2. Identified Deficiencies & Abnormal Findings
        - Clearly highlight any values that are Low, High, or Borderline.
        - Explain what each deficiency or abnormality indicates in simple, actionable terms.

        ### 🥗 3. Targeted Dietary & Nutritional Corrections
        - **Foods to Consume:** List specific whole foods, vegetables, and nutrient sources that directly help restore deficient markers.
        - **Foods to Avoid:** List dietary items to restrict based on elevated/abnormal values.

        ### 💡 4. Lifestyle & Clinical Recommendations
        - Provide actionable habits (hydration, sunlight exposure, physical activity, sleep hygiene).
        - State clearly when to consult a medical specialist for potential therapeutic supplementation.
        """

        response = client.models.generate_content(
            model="gemini-3.5-flash-lite",
            contents=[
                types.Part.from_bytes(data=contents, mime_type=mime_type),
                prompt,
            ],
        )

        return {"analysis": response.text}

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Lab report analysis failed: {str(e)}"
        )


@app.post("/generate_diet_plan")
@app.post("/generate-plan")
async def generate_diet_plan(data: BiometricsInput):
    if not client:
        raise HTTPException(
            status_code=500, detail="GEMINI_API_KEY environment variable is not set on Render."
        )

    try:
        bmr = (
            (10 * data.weight)
            + (6.25 * data.height)
            - (5 * data.age)
            + (5 if data.gender.lower() == "male" else -161)
        )
        multipliers = {
            "sedentary": 1.2,
            "light": 1.375,
            "moderate": 1.55,
            "active": 1.725,
            "extra": 1.9,
        }
        tdee = bmr * multipliers.get(data.activity.lower(), 1.55)

        prompt = f"""
        Act as a professional nutritionist. Generate a 1-day sample meal plan for a {data.age}-year-old {data.gender} 
        with Height: {data.height} cm, Weight: {data.weight} kg, BMR: {round(bmr)} kcal, and TDEE: {round(tdee)} kcal.

        Structure in Markdown:
        - **Breakfast**
        - **Lunch**
        - **Snack**
        - **Dinner**
        Include estimated calories and macro breakdowns (Protein, Carbs, Fats) for each meal.
        """

        response = client.models.generate_content(
            model="gemini-3.5-flash-lite",
            contents=prompt,
        )

        return {"plan": response.text, "bmr": round(bmr), "tdee": round(tdee)}

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Diet plan generation failed: {str(e)}"
        )
