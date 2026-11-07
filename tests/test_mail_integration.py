import pytest
from fastapi.testclient import TestClient

from main import app
from mail.controllers.mail_rest_controller import MAX_IMAGE_SIZE


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_send_mail_accepts_html_and_image(client, monkeypatch):
    sent = {}

    class FakeMailService:
        def send_message(self, **kwargs):
            sent.update(kwargs)

    monkeypatch.setattr("mail.controllers.mail_rest_controller.MailService", FakeMailService)
    response = client.post(
        "/api/mail/send",
        data={"to": "person@example.com", "text": "<h1>Hello</h1>"},
        files={"image": ("photo.png", b"image bytes", "image/png")},
    )

    assert response.status_code == 200
    assert response.json() == {"sent": True, "message": "Mail sent successfully."}
    assert sent == {
        "to": "person@example.com",
        "text": "<h1>Hello</h1>",
        "image": b"image bytes",
        "image_name": "photo.png",
    }


@pytest.mark.parametrize(
    ("data", "image", "expected_status"),
    [
        ({"to": "not-an-email", "text": "Hi"}, ("photo.png", b"image", "image/png"), 422),
        ({"to": "person@example.com", "text": "Hi"}, ("photo.txt", b"file", "text/plain"), 415),
        (
            {"to": "person@example.com", "text": "Hi"},
            ("photo.png", b"x" * (MAX_IMAGE_SIZE + 1), "image/png"),
            413,
        ),
    ],
)
def test_send_mail_rejects_invalid_input(client, monkeypatch, data, image, expected_status):
    def fail_if_called():
        pytest.fail("Mail service must not be called for invalid input.")

    monkeypatch.setattr("mail.controllers.mail_rest_controller.MailService", fail_if_called)
    response = client.post("/api/mail/send", data=data, files={"image": image})

    assert response.status_code == expected_status
