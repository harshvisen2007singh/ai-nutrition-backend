import os
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict
from google import genai
from google.genai import types

app = FastAPI(title="AI Personalised Nutrition Suite API")

# Allow CORS requests from Netlify and local development clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

api_key = os.environ.get("GEMINI_API_KEY")
client = genai.Client(api_key=api_key) if api_key else None


class BiometricsInput(BaseModel):
    model_config = ConfigDict(extra="ignore")
    age: float
    gender: str
    height: float
    weight: float
    activity: str
    diet_type: str = "Vegetarian"
    allergies: str = "None"


@app.get("/")
def read_root():
    return {"status": "ok", "message": "Backend API is live!"}


@app.post("/analyze_lab_report")
@app.post("/analyze-report")
async def analyze_lab_report(file: UploadFile = File(...)):
    if not client:
        raise HTTPException(
            status_code=500, detail="GEMINI_API_KEY is not configured on Render environment variables."
        )

    try:
        contents = await file.read()
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
        You are an expert clinical nutrition AI assistant. Perform a detailed analysis of the attached path lab report.

        Render your response in clear Markdown format:

        ### 🩺 1. Extracted Lab Parameters & Biomarkers
        - List all tested parameters detected (e.g., Hemoglobin, Vitamin D3, B12, TSH, Glucose, Lipid Profile).
        - State patient observed values vs reference ranges.

        ### ⚠️ 2. Identified Deficiencies & Abnormal Findings
        - Highlight values that are Low, High, or Borderline.
        - Explain what each abnormality means in simple, patient-friendly terms.

        ### 👨‍⚕️ 3. Recommended Medical Specialist Referral
        - Based strictly on the abnormal findings, explicitly name the specific medical specialist(s) the patient should consult (e.g., Endocrinologist, Hematologist, Cardiologist, or General Physician).
        - Provide a clear, comforting explanation of why consulting this specialist is recommended.

        ### 🥗 4. Targeted Nutritional & Dietary Corrections
        - **Foods to Consume:** Whole foods to help restore deficient levels.
        - **Foods to Avoid:** Items to restrict based on elevated markers.

        ### 💡 5. Lifestyle & Clinical Recommendations
        - Actionable lifestyle tips (hydration, exercise, sleep, sunlight).
        - Direct advice on when to consult a physician for therapeutic supplements.

        ### 🔗 6. Clinical & Research References
        Include clickable direct Markdown web links for medical verification:
        - [World Health Organization (WHO) Guidelines](https://www.who.int/health-topics/nutrition)
        - [ICMR - National Institute of Nutrition (NIN) Dietary Guidelines](https://www.nin.res.in/)
        - [NIH National Library of Medicine (PubMed)](https://pubmed.ncbi.nlm.nih.gov/)
        """

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[
                types.Part.from_bytes(data=contents, mime_type=mime_type),
                prompt,
            ],
        )

        return {"analysis": response.text}

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Analysis engine error: {str(e)}"
        )


@app.post("/generate_diet_plan")
@app.post("/generate-plan")
async def generate_diet_plan(data: BiometricsInput):
    if not client:
        raise HTTPException(
            status_code=500, detail="GEMINI_API_KEY is not configured on Render environment variables."
        )

    try:
        bmr = (10 * data.weight) + (6.25 * data.height) - (5 * data.age) + (5 if data.gender.lower() == "male" else -161)
        multipliers = {"sedentary": 1.2, "light": 1.375, "moderate": 1.55, "active": 1.725, "extra": 1.9}
        tdee = bmr * multipliers.get(data.activity.lower(), 1.55)

        prompt = f"""
        Act as a professional clinical nutritionist. Generate a 1-day sample meal plan for a {data.age}-year-old {data.gender} 
        with Height: {data.height} cm, Weight: {data.weight} kg, BMR: {round(bmr)} kcal, and TDEE: {round(tdee)} kcal.

        CRITICAL DIETARY CONSTRAINTS:
        - **Diet Type:** STRICTLY adhere to a **{data.diet_type}** diet.
        - **Allergies & Sensitivities:** STRICTLY EXCLUDE any foods containing: **{data.allergies}**.

        Structure in clear Markdown:
        ### 🥗 Personalised {data.diet_type} Meal Plan
        - **Breakfast**
        - **Lunch**
        - **Snack**
        - **Dinner**
        Include estimated calories and macro breakdowns (Protein, Carbs, Fats) for each meal.

        ### ⚠️ Allergy & Safety Verification
        - Explicitly confirm that the meal plan excludes **{data.allergies}** and strictly follows **{data.diet_type}** guidelines.

        ### 📚 Scientific References & Health Guidelines
        Provide direct, clickable Markdown hyperlinked sources for the patient to learn more:
        - [WHO Healthy Diet Guidelines](https://www.who.int/news-room/fact-sheets/detail/healthy-diet)
        - [ICMR-NIN Dietary Guidelines for Indians](https://www.nin.res.in/dietaryguidelines/index.html)
        - [Harvard T.H. Chan School of Public Health - The Nutrition Source](https://www.hsph.harvard.edu/nutritionsource/)
        - [USDA FoodData Central](https://fdc.nal.usda.gov/)
        """

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
        )

        return {"plan": response.text, "bmr": round(bmr), "tdee": round(tdee)}

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Diet generation error: {str(e)}"
        )
