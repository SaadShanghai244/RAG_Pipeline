import os

# Load all environment variables with optional default values and type casting where needed
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "default_openai_api_key")  
MODEL_NAME = os.getenv("MODEL", "gpt-4o-mini")  
DIR_PATH = os.getenv("DIR_PATH", "./data/txt/") 

TEMPERATURE = float(os.getenv("TEMPERATURE", 0.1))  
MAX_TOKENS = int(os.getenv("MAX_TOKENS", 50))  
EMBEDDED_MODEL = os.getenv("EMBEDDED_MODEL", "text-embedding-3-small")
