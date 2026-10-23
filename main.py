import os
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import google.generativeai as genai

# Configure Gemini API Key from environment variables
GENAI_API_KEY = os.getenv("GEMINI_API_KEY")
if GENAI_API_KEY:
    genai.configure(api_key=GENAI_API_KEY)

app = FastAPI(title="AI Personalised Nutrition Suite Backend")

# Enable CORS for Netlify Frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class BiometricsInput(BaseModel):
    age: float
    gender: str
    height: float
    weight: float
    activity: str

@app.get("/")
def read_root():
    return {"status": "ok", "message": "AI Nutrition Suite API is live!"}

@app.post("/analyze_lab_report")
@app.post("/analyze-report")
async def analyze_lab_report(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        
        # Determine media type dynamically for images & PDFs
        mime_type = file.content_type
        if not mime_type or mime_type == "application/octet-stream":
            filename = file.filename.lower()
            if filename.endswith(".pdf"):
                mime_type = "application/pdf"
            elif filename.endswith((".jpg", ".jpeg")):
                mime_type = "image/jpeg"
            elif filename.endswith(".png"):
                mime_type = "image/png"
            else:
                mime_type = "image/jpeg"

        file_part = {
            "mime_type": mime_type,
            "data": contents
        }

        # Comprehensive Medical & Deficiency Extraction Prompt
        prompt = """
        You are an expert clinical nutrition AI assistant. Perform an in-depth, structured analysis of the attached path lab blood report.

        Please render your analysis in clear Markdown format using the following structure:

        ### 🩺 1. Extracted Lab Parameters & Biomarkers
        - Create a list/table of all key tested parameters detected in the document (e.g., Hemoglobin, RBC, Vitamin D3, Vitamin B12, TSH/Thyroid, Fasting Blood Glucose, Lipid Profile, Liver/Kidney Markers).
        - Specify the patient's observed test value vs standard reference ranges.

        ### ⚠️ 2. Identified Deficiencies & Abnormal Findings
        - Clearly highlight any values that are Low, High, or Borderline.
        - Explain what each deficiency or abnormality indicates in simple, actionable terms (e.g., "Low Vitamin D3 impairs bone density and immunity").

        ### 🥗 3. Targeted Nutritional & Dietary Corrections
        - **Foods to Consume:** List specific whole foods, vegetables, and nutrient sources that directly help restore deficient markers.
        - **Foods to Avoid:** List dietary items to restrict based on elevated/abnormal values (e.g., high sodium, refined sugars, saturated fats).

        ### 💡 4. Lifestyle & Clinical Recommendations
        - Provide actionable habits (hydration, sunlight exposure, physical activity, sleep hygiene).
        - State clearly when to consult a medical specialist for potential therapeutic supplementation.
        """

        model = genai.GenerativeModel('gemini-2.5-flash')
        response = model.generate_content([prompt, file_part])

        return {"analysis": response.text}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lab report analysis failed: {str(e)}")

@app.post("/generate_diet_plan")
@app.post("/generate-plan")
async def generate_diet_plan(data: BiometricsInput):
    try:
        # Mifflin-St Jeor Equation
        bmr = (10 * data.weight) + (6.25 * data.height) - (5 * data.age) + (5 if data.gender.lower() == "male" else -161)
        multipliers = {"sedentary": 1.2, "light": 1.375, "moderate": 1.55, "active": 1.725}
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

        model = genai.GenerativeModel('gemini-3.5-flash-lite')
        response = model.generate_content(prompt)

        return {"plan": response.text, "bmr": round(bmr), "tdee": round(tdee)}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Diet plan generation failed: {str(e)}")
