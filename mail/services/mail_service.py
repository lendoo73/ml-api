from pathlib import Path
from tempfile import TemporaryDirectory

from cscs_tools.mail.enums.security_mode import SecurityMode
from cscs_tools.mail.services.mail_service import MailService as CscsMailService

from config.config import MAIL_FROM_ADDRESS, MAIL_SMTP_PASSWORD, MAIL_SMTP_PORT, MAIL_SMTP_SECURITY
from config.config import MAIL_SMTP_SERVER, MAIL_SMTP_TIMEOUT, MAIL_SMTP_USERNAME


class MailService(CscsMailService):
    def __init__(self) -> None:
        if not MAIL_SMTP_SERVER or not MAIL_FROM_ADDRESS:
            raise RuntimeError("MAIL_SMTP_SERVER and MAIL_FROM_ADDRESS must be configured.")

        try:
            security = SecurityMode[MAIL_SMTP_SECURITY.upper()]
        except KeyError as error:
            raise RuntimeError("MAIL_SMTP_SECURITY must be STARTTLS, SSL, or PLAIN.") from error

        super().__init__(
            smtp_server=MAIL_SMTP_SERVER,
            from_address=MAIL_FROM_ADDRESS,
            smtp_port=MAIL_SMTP_PORT,
            smtp_username=MAIL_SMTP_USERNAME,
            smtp_password=MAIL_SMTP_PASSWORD,
            security=security,
            timeout=MAIL_SMTP_TIMEOUT,
        )

    def send_message(self, to: str, text: str, image: bytes, image_name: str) -> None:
        safe_name = Path(image_name).name or "image"
        with TemporaryDirectory(prefix="ml_api_mail_") as temporary_directory:
            html_path = Path(temporary_directory) / "message.html"
            image_path = Path(temporary_directory) / safe_name
            html_path.write_text(text, encoding="utf-8")
            image_path.write_bytes(image)
            self.send_html(
                to=to,
                subject="Message",
                html_file_path=str(html_path),
                attachments=[str(image_path)],
            )
