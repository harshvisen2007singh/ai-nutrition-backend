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
    disease: str = "None / Healthy"
    dosha: str = "Tridosha Balance (General)"


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
        - Explicitly name recommended specialists across treatment streams (e.g., Allopathic Specialist like Endocrinologist/Cardiologist, Homeopathic Practitioner, or Ayurvedic Vaidya).

        ### 🌿 4. Tridosha & Functional Health Correlation
        - Correlate detected deficiencies with Ayurvedic Tridosha imbalances (Vata: dry/nervous, Pitta: metabolic/inflammatory, Kapha: sluggish/mucus).

        ### 🥗 5. Targeted Precision Diet Plan
        - **Foods to Consume:** Whole foods restoring biomarker levels & balancing Doshas.
        - **Foods to Avoid:** Items aggravating current clinical markers or Doshas.

        ### 🔗 6. Clinical & Research References
        - [World Health Organization (WHO) Guidelines](https://www.who.int/health-topics/nutrition)
        - [ICMR - National Institute of Nutrition (NIN)](https://www.nin.res.in/)
        - [AYUSH Ministry - Ayurvedic Dietary Principles](https://www.ayush.gov.in/)
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
        Act as an integrative clinical nutritionist expert in Modern Allopathic Nutrition, Homeopathy, and Ayurvedic Tridosha Balance.
        Generate a 1-day integrative meal plan for a {data.age}-year-old {data.gender} (Height: {data.height} cm, Weight: {data.weight} kg, BMR: {round(bmr)} kcal, TDEE: {round(tdee)} kcal).

        INTEGRATIVE CLINICAL CONSTRAINTS:
        - **Medical Condition:** Tailor strictly for **{data.disease}**.
        - **Ayurvedic Prakriti / Dosha Target:** Align with **{data.dosha}**. 
          * Vata Dominant: Warm, grounding, cooked, nutrient-dense foods with healthy fats.
          * Pitta Dominant: Cooling, non-acidic, moderate spice, hydrating foods.
          * Kapha Dominant: Light, warm, pungent/bitter spices, low-oil, low-mucus forming foods.
        - **Dietary Preference:** STRICTLY **{data.diet_type}**.
        - **Allergies & Sensitivities:** STRICTLY EXCLUDE **{data.allergies}**.

        Structure in clear Markdown:
        ### 🧬 Integrative Precision Meal Plan
        **Condition:** {data.disease} | **Dosha Focus:** {data.dosha} | **Diet:** {data.diet_type}

        - **Breakfast**
        - **Lunch**
        - **Snack**
        - **Dinner**
        Include calories, macronutrient breakdown, and Ayurvedic therapeutic properties (e.g., Rasa, Virya) for key ingredients.

        ### 🌿 Homeopathic & Ayurvedic Complementary Notes
        - Gentle lifestyle, herbal infusion, or Homeopathic bio-chemic cell salt advice complementing the diet.

        ### 📚 Scientific & Traditional References
        - [WHO Healthy Diet Guidelines](https://www.who.int/news-room/fact-sheets/detail/healthy-diet)
        - [ICMR-NIN Dietary Guidelines for Indians](https://www.nin.res.in/dietaryguidelines/index.html)
        - [Ministry of AYUSH - Traditional Knowledge Digital Library](https://www.ayush.gov.in/)
        """

        response = client.models.generate_content(
            model="gemini-3.5-flash-lite",
            contents=prompt,
        )

        return {"plan": response.text, "bmr": round(bmr), "tdee": round(tdee)}

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Diet generation error: {str(e)}"
        )
