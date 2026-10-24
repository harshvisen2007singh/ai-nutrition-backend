import os
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict
from google import genai
from google.genai import types

app = FastAPI(title="AI Personalised Nutrition Suite API")

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
    disease: str = "None / General Wellness"  # Can now accept comma-separated multiple diseases
    dosha: str = "N/A (Not Applicable / Skip Tridosha)"


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
        You are an expert integrative clinical nutritionist proficient in Allopathy, Homeopathy, and Ayurveda. Perform a detailed multimodal analysis of the attached path lab report.

        Render your response in clear Markdown format:

        ### 🩺 1. Extracted Lab Parameters & Biomarkers (Allopathic View)
        - List tested parameters detected (e.g., Hemoglobin, Vitamin D3, B12, TSH, Glucose, Lipid Profile).
        - State patient observed values vs standard reference ranges.

        ### ⚠️ 2. Identified Deficiencies & Abnormal Findings
        - Highlight values that are Low, High, or Borderline.
        - Explain clinical significance in simple terms.

        ### 👨‍⚕️ 3. Integrative Medical Specialist Referral
        - Explicitly name recommended specialists across treatment streams (e.g., Endocrinologist, Cardiologist, Homeopathic Practitioner, or Ayurvedic Vaidya).

        ### 🌿 4. Tridosha & Functional Health Correlation
        - Correlate detected deficiencies with Ayurvedic Tridosha imbalances (Vata, Pitta, Kapha) if applicable.

        ### 🥗 5. Targeted Precision Diet Plan
        - **Foods to Consume:** Whole foods restoring biomarker levels & balancing health.
        - **Foods to Avoid:** Items aggravating current clinical markers.

        ### 🔗 6. Clinical & Research References
        - [World Health Organization (WHO) Guidelines](https://www.who.int/health-topics/nutrition)
        - [ICMR - National Institute of Nutrition (NIN)](https://www.nin.res.in/)
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
        Act as an integrative clinical nutritionist expert in Modern Allopathic Nutrition, Homeopathy, and Ayurvedic Principles.
        Generate a 1-day integrative meal plan for a {data.age}-year-old {data.gender} (Height: {data.height} cm, Weight: {data.weight} kg, BMR: {round(bmr)} kcal, TDEE: {round(tdee)} kcal).

        CRITICAL MULTI-CONDITION & DIETARY CONSTRAINTS:
        - **Selected Clinical Conditions / Diseases:** STRICTLY TAILOR FOR ALL SELECTED CONDITIONS: **{data.disease}**.
          * You MUST harmonize dietary requirements across ALL selected conditions simultaneously (e.g., if both Diabetes AND Hypertension are selected, combine low Glycemic Index foods with Low Sodium/DASH guidelines; if Gout is selected, strictly restrict purines).
        - **Ayurvedic Prakriti / Dosha Target:** **{data.dosha}** (Ignore if N/A).
        - **Dietary Preference:** STRICTLY **{data.diet_type}**.
        - **Allergies & Sensitivities:** STRICTLY EXCLUDE **{data.allergies}**.

        Structure in clear Markdown:
        ### 🩺 Multi-Condition Therapeutic Meal Plan
        **Selected Conditions:** {data.disease} | **Diet Pattern:** {data.diet_type} | **Exclusions:** {data.allergies}

        - **Breakfast**
        - **Lunch**
        - **Snack**
        - **Dinner**
        Include estimated calories and macro breakdowns (Protein, Carbs, Fats) for each meal.

        ### ⚠️ Clinical Harmony & Safety Verification
        - Explicitly detail how this meal plan addresses EACH of the selected conditions (**{data.disease}**) safely without cross-interfering.
        - Confirm strict exclusion of **{data.allergies}**.

        ### 📚 Scientific & Reference Links
        Provide direct, clickable Markdown hyperlinked sources:
        - [WHO Healthy Diet Guidelines](https://www.who.int/news-room/fact-sheets/detail/healthy-diet)
        - [ICMR-NIN Dietary Guidelines for Indians](https://www.nin.res.in/dietaryguidelines/index.html)
        - [Harvard T.H. Chan School of Public Health - Nutrition Source](https://www.hsph.harvard.edu/nutritionsource/)
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
