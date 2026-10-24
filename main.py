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
    disease: str = "None / General Wellness"
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
        You are an expert integrative clinical oncologist and clinical nutritionist. Perform a detailed multimodal analysis of the attached path lab report.

        Render your response in clear Markdown format:

        ### 🩺 1. Extracted Lab Parameters & Biomarkers (Clinical View)
        - List tested parameters detected (e.g., Hemoglobin, Vitamin D3, B12, TSH, Glucose, Lipid Profile, Liver/Kidney Function).
        - State patient observed values vs standard reference ranges.

        ### ⚠️ 2. Identified Deficiencies & Abnormal Findings
        - Highlight values that are Low, High, or Borderline.
        - Explain clinical significance in simple, comforting terms.

        ### 👨‍⚕️ 3. Integrative Medical Specialist Referral
        - Explicitly name recommended specialists (e.g., Medical Oncologist, Clinical Dietitian, Endocrinologist, Gastroenterologist, Ayurvedic Vaidya).

        ### 🌿 4. Functional & Constitutional Health Correlation
        - Correlate detected deficiencies with body constitution or symptom severity if applicable.

        ### 🥗 5. Targeted Precision Diet & Symptom Management
        - **Foods to Consume:** Nutrient-dense whole foods restoring biomarker levels.
        - **Foods to Avoid:** Items aggravating current symptoms or clinical markers.

        ### 🔗 6. Clinical & Research References
        - [World Health Organization (WHO) Guidelines](https://www.who.int/health-topics/nutrition)
        - [ICMR - National Institute of Nutrition (NIN)](https://www.nin.res.in/)
        - [Indian Council of Medical Research (ICMR) Oncology Guidelines](https://main.icmr.nic.in/)
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
        Act as an expert clinical oncology nutritionist specializing in Indian dietary patterns (Veg, Non-Veg, Vegan, Eggetarian).
        Generate a 1-day therapeutic, symptom-management meal plan for a {data.age}-year-old {data.gender} (Height: {data.height} cm, Weight: {data.weight} kg, BMR: {round(bmr)} kcal, TDEE: {round(tdee)} kcal).

        CRITICAL CLINICAL & ONCOLOGY CONSTRAINTS:
        - **Selected Conditions / Cancers:** **{data.disease}**.
          * Apply specific clinical nutrition rules for selected cancer types using Indian foods tailored to diet preference ({data.diet_type}):
            - **Breast Cancer:** Maintain lean muscle mass & manage appetite/treatment weight changes (e.g., Dal, Paneer, Tofu, Eggs, Fish/Chicken if Non-Veg, Soft Vegetables, Roti).
            - **Lung Cancer:** Address appetite loss, fatigue, & weight loss (e.g., Moong Dal Khichdi, Curd/Yogurt, Eggs, Soft Paneer, Protein Soups).
            - **Colorectal Cancer:** Adapt fiber & texture to bowel symptoms (e.g., White Rice, Soft Dal, Steamed Vegetables, Ripe Bananas; modify insoluble vs soluble fiber).
            - **Stomach Cancer:** Manage early fullness & small stomach capacity (e.g., Frequent small portions of Khichdi, Soft Idli, Curd, Protein Broths/Soups).
            - **Oral & Throat Cancer:** Manage chewing, swallowing difficulties, dry mouth, & sores (e.g., Blended Khichdi, Smooth Dal, Mashed Veggies, Nutrient Smoothies, Scrambled Eggs).
            - **Pancreatic Cancer:** Support poor digestion, weight loss, & fat intolerance (e.g., Small high-protein meals, low-to-moderate adjusted fats, cooked easy-to-digest foods).
            - **Liver Cancer:** Consider liver function, fluid retention, & protein/sodium balance (e.g., Individually balanced protein, controlled sodium khichdi/soup).
            - **Prostate Cancer:** Support muscle & bone health during hormone therapy (e.g., Dal, Soy/Tofu, Beans, Dairy if tolerated, Cruciferous Vegetables, Whole Grains).
            - **Ovarian Cancer:** Manage abdominal bloating, early fullness, & nausea (e.g., Small frequent meals, Soft Rice, Eggs/Tofu, Clear Warm Soups).
            - **Leukaemia / Lymphoma:** Prevent neutropenic infection risks & manage fatigue (e.g., Freshly prepared, thoroughly cooked hygiene-safe meals, well-cooked Dal, Eggs, Meat if Non-Veg).

        - **Dietary Preference:** STRICTLY **{data.diet_type}**.
          * Veg: Dal, Paneer, Soy, Tofu, Dairy, Legumes.
          * Non-Veg: Well-cooked Fish, Chicken, Eggs, Dal, Paneer.
          * Vegan: Tofu, Soy, Dal, Legumes, Plant Milk, Nuts/Seeds (No Dairy/Eggs/Meat).
          * Eggetarian: Eggs, Dal, Paneer, Tofu, Legumes (No Meat/Fish).
        - **Allergies & Restrictions:** EXCLUDE **{data.allergies}**.
        - **Ayurvedic Prakriti:** **{data.dosha}** (Ignore if N/A).

        Structure in clear Markdown:
        ### 🩺 Clinical Oncology & Therapeutic Meal Plan
        **Selected Conditions/Cancers:** {data.disease} | **Diet Pattern:** {data.diet_type} | **Exclusions:** {data.allergies}

        - **Breakfast**
        - **Mid-Morning Snack**
        - **Lunch**
        - **Evening Snack**
        - **Dinner**
        Include estimated calories, macronutrient breakdown (Protein, Carbs, Fats), and food texture/consistency notes.

        ### ⚠️ Symptom & Food Safety Guidelines
        - Provide practical Indian meal prep tips for managing nausea, taste changes, early fullness, or neutropenic food safety.
        - Confirm strict exclusion of **{data.allergies}**.

        ### 📚 Clinical References
        - [ICMR Guidelines for Cancer Management](https://main.icmr.nic.in/)
        - [WHO Healthy Diet & Cancer Prevention](https://www.who.int/news-room/fact-sheets/detail/healthy-diet)
        - [ICMR-NIN Dietary Guidelines for Indians](https://www.nin.res.in/dietaryguidelines/index.html)
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
