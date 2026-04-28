"""
api/routes/upload.py
─────────────────────
POST /upload-image

Accepts an image file (JPEG, PNG, WebP), validates MIME type and size,
saves as JPEG to output/uploads/{uuid}.jpg, and returns an UploadResponse.
"""

import io
import uuid
from pathlib import Path

from fastapi import APIRouter, HTTPException, UploadFile
from PIL import Image

from api.models import UploadResponse

router = APIRouter()

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_UPLOADS_DIR  = _PROJECT_ROOT / "output" / "uploads"

_ALLOWED_MIME_TYPES = {"image/jpeg", "image/png", "image/webp"}
_MAX_FILE_SIZE      = 10 * 1024 * 1024  # 10 MB


@router.post("/upload-image", response_model=UploadResponse)
async def upload_image(file: UploadFile) -> UploadResponse:
    """
    Accept an image upload and save it to output/uploads/{uuid}.jpg.

    Validation:
      - MIME type must be image/jpeg, image/png, or image/webp (422 otherwise)
      - File size must be ≤ 10 MB (422 otherwise)

    Returns image_id, preview_url, original dimensions, and original filename.
    """
    if file.content_type not in _ALLOWED_MIME_TYPES:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Unsupported file type: {file.content_type!r}. "
                "Allowed: image/jpeg, image/png, image/webp."
            ),
        )

    content = await file.read()

    if len(content) > _MAX_FILE_SIZE:
        raise HTTPException(
            status_code=422,
            detail=f"File too large: {len(content)} bytes. Maximum is 10 MB.",
        )

    img = Image.open(io.BytesIO(content)).convert("RGB")
    orig_w, orig_h = img.size

    image_id  = str(uuid.uuid4())
    save_path = _UPLOADS_DIR / f"{image_id}.jpg"
    img.save(str(save_path), format="JPEG", quality=95)

    return UploadResponse(
        image_id=image_id,
        preview_url=f"/api/v1/media/{image_id}/{image_id}.jpg",
        width=orig_w,
        height=orig_h,
        original_filename=file.filename or "",
    )
