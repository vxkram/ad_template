import io
import os
import sys
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import main


@pytest.fixture
def client():
    return TestClient(main.app)


def make_image_bytes(size=(64, 64), color=(200, 30, 30)):
    buf = io.BytesIO()
    Image.new("RGB", size, color=color).save(buf, format="PNG")
    buf.seek(0)
    return buf


def test_hex_to_rgb():
    assert main.hex_to_rgb("#FF0000") == (255, 0, 0)
    assert main.hex_to_rgb("00ff00") == (0, 255, 0)


def test_prompt_with_colorname_exact_csv_match():
    result = main.prompt_with_colorname("disney pixar", "#FF0000")
    assert result.startswith("disney pixar, ")


def test_prompt_with_colorname_falls_back_to_model():
    # An arbitrary hex unlikely to be an exact named color in the CSV.
    result = main.prompt_with_colorname("disney pixar", "#123456")
    assert result.startswith("disney pixar, ")


def test_index_page(client):
    response = client.get("/")
    assert response.status_code == 200
    assert b"AD TEMPLATE" in response.content


def test_get_output_template_form(client):
    response = client.get("/get_output_template")
    assert response.status_code == 200
    assert b"AD TEMPLATE" in response.content


def test_color_name_endpoint(client):
    response = client.post("/color_name", data={"prompt": "disney pixar", "hex": "#FF0000"})
    assert response.status_code == 200
    assert "disney pixar" in response.json()["message"]


def test_getimg2img_uses_mocked_pipeline_and_redirects(client, monkeypatch, tmp_path):
    fake_output = Image.new("RGB", (512, 512), color=(10, 20, 30))
    fake_pipeline_result = MagicMock()
    fake_pipeline_result.images = [fake_output]
    fake_pipeline = MagicMock(return_value=fake_pipeline_result)

    monkeypatch.setattr(main, "get_pipeline", lambda: fake_pipeline)

    saved_path = tmp_path / "image_createdx.jpg"
    monkeypatch.setattr(main, "GENERATED_IMAGE_PATH", str(saved_path))

    files = {"image": ("product.png", make_image_bytes(), "image/png")}
    data = {"prompt": "disney pixar", "hex": "#FF0000"}
    response = client.post("/getimg2img", files=files, data=data, follow_redirects=False)

    assert response.status_code == 303
    assert response.headers["location"].endswith("/get_output_template")
    fake_pipeline.assert_called_once()
    assert saved_path.exists()


def test_output_template_without_generated_image_returns_400(client, monkeypatch, tmp_path):
    monkeypatch.setattr(main, "GENERATED_IMAGE_PATH", str(tmp_path / "does_not_exist.jpg"))

    files = {"logo": ("logo.png", make_image_bytes(), "image/png")}
    response = client.post("/output_template", files=files, data={"hex": "#FF0000"})
    assert response.status_code == 400
    assert "generate one on the home page first" in response.json()["error"]


def test_output_template_happy_path(client, monkeypatch, tmp_path):
    created_path = tmp_path / "image_createdx.jpg"
    Image.new("RGB", (512, 512), color=(50, 60, 70)).save(created_path)
    monkeypatch.setattr(main, "GENERATED_IMAGE_PATH", str(created_path))

    files = {"logo": ("logo.png", make_image_bytes(), "image/png")}
    data = {"hex": "#FF0000", "punchline_text": "Great product", "button_text": "Buy now"}
    response = client.post("/output_template", files=files, data=data)

    assert response.status_code == 200
    assert b"data:image/jpeg;base64," in response.content
