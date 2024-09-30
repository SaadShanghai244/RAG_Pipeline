# It will load all the environment variables
import os

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL_NAME = os.getenv("MODEL")
DIR_PATH = os.getenv("DIR_PATH")
TEMPERATURE = os.getenv("TEMPERATURE")
MAX_TOKENS = os.getenv("MAX_TOKENS")
EMBEDDED_MODEL = os.getenv("EMBEDDED_MODEL")