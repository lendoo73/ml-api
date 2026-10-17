import os

from dotenv import load_dotenv

load_dotenv()

ENVIRONMENT = os.getenv("ENVIRONMENT")

CORS_ALLOWED_ORIGINS = os.getenv("CORS_ALLOWED_ORIGINS")

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT = os.getenv("LOG_FORMAT")
LOG_TO_STDOUT = os.getenv("LOG_TO_STDOUT", "False").lower() in ("1", "true", "yes")