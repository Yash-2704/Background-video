"""
tests/test_upload.py
─────────────────────
Tests for POST /api/v1/upload-image.

All tests use a minimal FastAPI TestClient that includes only the upload
router. The _UPLOADS_DIR is patched to tmp_path so no files land in the
real output/uploads/ directory.
"""

import io
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image

from api.routes.upload import router

# Minimal app — isolates upload router from other routes
_app = FastAPI()
_app.include_router(router, prefix="/api/v1")
client = TestClient(_app)


def _make_jpeg_bytes(width: int = 10, height: int = 10) -> bytes:
    img = Image.new("RGB", (width, height), color=(100, 150, 200))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


# ── TEST A ─────────────────────────────────────────────────────────────────────

def test_valid_jpeg_upload_returns_200_and_image_id(tmp_path):
    """TEST A: valid JPEG upload returns 200 and response contains image_id and preview_url."""
    with patch("api.routes.upload._UPLOADS_DIR", tmp_path):
        response = client.post(
            "/api/v1/upload-image",
            files={"file": ("photo.jpg", _make_jpeg_bytes(), "image/jpeg")},
        )
    assert response.status_code == 200
    data = response.json()
    assert "image_id" in data
    assert "preview_url" in data
    assert data["image_id"] != ""
    assert data["preview_url"].startswith("/api/v1/media/")


# ── TEST B ─────────────────────────────────────────────────────────────────────

def test_non_image_file_rejected_with_422(tmp_path):
    """TEST B: posting a text/plain file is rejected with 422."""
    with patch("api.routes.upload._UPLOADS_DIR", tmp_path):
        response = client.post(
            "/api/v1/upload-image",
            files={"file": ("notes.txt", b"hello world", "text/plain")},
        )
    assert response.status_code == 422


# ── TEST C ─────────────────────────────────────────────────────────────────────

def test_file_over_10mb_rejected_with_422(tmp_path):
    """TEST C: file larger than 10 MB is rejected with 422."""
    big_data = b"x" * (10 * 1024 * 1024 + 1)
    with patch("api.routes.upload._UPLOADS_DIR", tmp_path):
        response = client.post(
            "/api/v1/upload-image",
            files={"file": ("big.jpg", big_data, "image/jpeg")},
        )
    assert response.status_code == 422


# ── TEST D ─────────────────────────────────────────────────────────────────────

def test_uploaded_file_exists_on_disk(tmp_path):
    """TEST D: after a valid upload the file exists at uploads_dir/{image_id}.jpg."""
    with patch("api.routes.upload._UPLOADS_DIR", tmp_path):
        response = client.post(
            "/api/v1/upload-image",
            files={"file": ("scene.jpg", _make_jpeg_bytes(), "image/jpeg")},
        )
    assert response.status_code == 200
    image_id = response.json()["image_id"]
    assert (tmp_path / f"{image_id}.jpg").exists()
