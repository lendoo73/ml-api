import os

from dotenv import load_dotenv

load_dotenv()

ENVIRONMENT = os.getenv("ENVIRONMENT")

CORS_ALLOWED_ORIGINS = os.getenv("CORS_ALLOWED_ORIGINS")

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT = os.getenv("LOG_FORMAT")
LOG_TO_STDOUT = os.getenv("LOG_TO_STDOUT", "False").lower() in ("1", "true", "yes")

MAIL_SMTP_SERVER = os.getenv("MAIL_SMTP_SERVER")
MAIL_FROM_ADDRESS = os.getenv("MAIL_FROM_ADDRESS")
MAIL_SMTP_PORT = int(os.getenv("MAIL_SMTP_PORT", "587"))
MAIL_SMTP_USERNAME = os.getenv("MAIL_SMTP_USERNAME")
MAIL_SMTP_PASSWORD = os.getenv("MAIL_SMTP_PASSWORD")
MAIL_SMTP_SECURITY = os.getenv("MAIL_SMTP_SECURITY", "STARTTLS")
MAIL_SMTP_TIMEOUT = int(os.getenv("MAIL_SMTP_TIMEOUT", "15"))