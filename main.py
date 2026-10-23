import os
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import google.generativeai as genai

# Configure Gemini API Key
GENAI_API_KEY = os.environ.get("GEMINI_API_KEY")
if GENAI_API_KEY:
    genai.configure(api_key=GENAI_API_KEY)

app = FastAPI(title="AI Personalised Nutrition Suite API")

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
    return {"status": "ok", "message": "Backend API is live!"}

@app.post("/analyze_lab_report")
@app.post("/analyze-report")
async def analyze_lab_report(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        
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

        prompt = """
        You are an expert clinical nutrition AI assistant. Perform a detailed analysis of the attached path lab report.

        Render your response in clear Markdown format:

        ### 🩺 1. Extracted Lab Parameters & Biomarkers
        - List all tested parameters detected (e.g., Hemoglobin, Vitamin D3, B12, TSH, Glucose, Lipid Profile).
        - State patient observed values vs reference ranges.

        ### ⚠️ 2. Identified Deficiencies & Abnormal Findings
        - Highlight values that are Low, High, or Borderline.
        - Explain what each abnormality means in simple terms.

        ### 🥗 3. Targeted Nutritional & Dietary Corrections
        - **Foods to Consume:** Whole foods to help restore deficient levels.
        - **Foods to Avoid:** Items to restrict based on elevated markers.

        ### 💡 4. Lifestyle & Clinical Recommendations
        - Actionable lifestyle tips (hydration, exercise, sleep, sunlight).
        - Direct advice on when to consult a physician for therapeutic supplements.
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
