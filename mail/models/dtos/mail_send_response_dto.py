from pydantic import BaseModel


class MailSendResponseDto(BaseModel):
    sent: bool
    message: str
