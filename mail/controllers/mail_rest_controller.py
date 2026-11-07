from pathlib import Path

from cscs_tools.mail.exceptions.exceptions import MailAuthenticationException, MailConnectionException
from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from mail.models.dtos.mail_send_response_dto import MailSendResponseDto
from mail.services.mail_service import MailService

MAX_IMAGE_SIZE = 10 * 1024 * 1024
ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/gif", "image/webp"}
EMAIL_PATTERN = r"^[^@\s]+@[^@\s]+\.[^@\s]+$"

router = APIRouter(prefix="/mail", tags=["mail"])


@router.post("/send", response_model=MailSendResponseDto)
def send_mail(
    to: str = Form(..., min_length=3, max_length=254, pattern=EMAIL_PATTERN),
    text: str = Form(..., min_length=1, max_length=100_000),
    image: UploadFile = File(...),
) -> MailSendResponseDto:
    if image.content_type not in ALLOWED_IMAGE_TYPES:
        raise HTTPException(status_code=415, detail="Only JPEG, PNG, GIF, and WebP images are supported.")

    image_bytes = image.file.read(MAX_IMAGE_SIZE + 1)
    if len(image_bytes) > MAX_IMAGE_SIZE:
        raise HTTPException(status_code=413, detail="Image must be 10 MB or smaller.")

    try:
        MailService().send_message(
            to=to,
            text=text,
            image=image_bytes,
            image_name=Path(image.filename or "image").name,
        )
    except MailAuthenticationException as error:
        raise HTTPException(status_code=503, detail="Mail service authentication failed.") from error
    except MailConnectionException as error:
        raise HTTPException(status_code=503, detail="Mail service is unavailable.") from error

    return MailSendResponseDto(sent=True, message="Mail sent successfully.")
