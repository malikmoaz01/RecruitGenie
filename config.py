import os
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("API_KEY")

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_API_KEY = API_KEY

if API_KEY:
    os.environ["OPENAI_API_KEY"] = API_KEY
