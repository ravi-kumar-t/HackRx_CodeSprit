import os
from dotenv import load_dotenv
import google.generativeai as genai
import json

# Load environment variables
load_dotenv()
gemini_api_key = os.getenv("GOOGLE_API_KEY")
if not gemini_api_key:
    raise ValueError("GOOGLE_API_KEY not found.")

genai.configure(api_key=gemini_api_key)

GEMINI_MODEL_NAME = "models/gemini-1.5-flash-002"
model = genai.GenerativeModel(GEMINI_MODEL_NAME)

def extract_query_info(user_query: str) -> dict:
    """
    Extract structured details (age, gender, procedure, location, policy_duration) from user query.
    """
    extraction_prompt = f"""
    Extract the following from the insurance query:
    - age (integer)
    - gender (male/female/not specified)
    - procedure (string)
    - location (string)
    - policy_duration (string)

    If missing, return "not specified".
    Output only JSON.

    Query: "{user_query}"
    JSON Output:
    """
    try:
        response = model.generate_content(
            extraction_prompt,
            generation_config={"response_mime_type": "application/json"}
        )
        raw_json_string = response.text.strip()
        if raw_json_string.startswith("```json") and raw_json_string.endswith("```"):
            raw_json_string = raw_json_string[7:-3].strip()
        return json.loads(raw_json_string)
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    test_query = "46-year-old male, knee surgery in Pune, 3-month-old insurance policy"
    print(json.dumps(extract_query_info(test_query), indent=2))
