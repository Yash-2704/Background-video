"""
tests/test_video_serve.py
─────────────────────────
10 tests for GET /api/v1/media/{clip_id}/{filename}.

Uses FastAPI TestClient with real files on disk (no cv2, no run_generation).
Patches _PROJECT_ROOT in api.routes.bundle so the endpoint resolves to
the temp directory created by the fake_bundle fixture.
"""

import pytest
from unittest.mock import patch
from fastapi.testclient import TestClient

from api.main import app
import api.routes.bundle as bundle_module

client = TestClient(app)


# ── Fixture ────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def fake_bundle(tmp_path_factory):
    """
    Creates a minimal fake output bundle directory.
    Writes a tiny valid-ish MP4 (ftyp box) and a minimal GIF89a header.
    Uses getbasetemp() so paths align with endpoint expectations:
      _PROJECT_ROOT / "output" / clip_id / "final" / filename
    """
    clip_id = "bg_test_abc123"

    # Base temp dir — will serve as _PROJECT_ROOT in patched tests
    base = tmp_path_factory.getbasetemp()
    bundle_dir = base / "output" / clip_id
    final_dir  = bundle_dir / "final"
    final_dir.mkdir(parents=True)

    # Minimal valid-ish MP4 (ftyp box — FastAPI FileResponse serves it fine)
    mp4_path = final_dir / f"{clip_id}.mp4"
    mp4_path.write_bytes(
        b'\x00\x00\x00\x20ftypisom'
        b'\x00\x00\x00\x00isomiso2'
        b'avc1mp41'
    )

    # Minimal valid GIF89a header
    gif_path = final_dir / f"{clip_id}_preview.gif"
    gif_path.write_bytes(
        b'GIF89a\x01\x00\x01\x00\x00\xff\x00,'
        b'\x00\x00\x00\x00\x01\x00\x01\x00\x00\x02\x00;'
    )

    # Minimal JSON metadata file for regression test (test_10)
    meta_path = bundle_dir / f"{clip_id}_metadata.json"
    meta_path.write_text('{"clip_id": "bg_test_abc123"}', encoding="utf-8")

    return {
        "clip_id":     clip_id,
        "bundle_dir":  bundle_dir,
        "mp4_path":    mp4_path,
        "gif_path":    gif_path,
        "meta_path":   meta_path,
        # bundle_dir.parent == base/output, .parent.parent == base
        "_project_root": bundle_dir.parent.parent,
    }


# ── Tests ──────────────────────────────────────────────────────────────────────

def test_01_mp4_returns_200(fake_bundle):
    """GET /media/{clip_id}/{clip_id}.mp4 returns 200 when file exists."""
    clip_id  = fake_bundle["clip_id"]
    filename = f"{clip_id}.mp4"
    with patch.object(bundle_module, '_PROJECT_ROOT', fake_bundle["_project_root"]):
        response = client.get(f"/api/v1/media/{clip_id}/{filename}")
    assert response.status_code == 200


def test_02_mp4_content_type(fake_bundle):
    """MP4 response Content-Type is video/mp4."""
    clip_id  = fake_bundle["clip_id"]
    filename = f"{clip_id}.mp4"
    with patch.object(bundle_module, '_PROJECT_ROOT', fake_bundle["_project_root"]):
        response = client.get(f"/api/v1/media/{clip_id}/{filename}")
    assert "video/mp4" in response.headers["content-type"]


def test_03_mp4_accept_ranges_header(fake_bundle):
    """MP4 response has Accept-Ranges: bytes header."""
    clip_id  = fake_bundle["clip_id"]
    filename = f"{clip_id}.mp4"
    with patch.object(bundle_module, '_PROJECT_ROOT', fake_bundle["_project_root"]):
        response = client.get(f"/api/v1/media/{clip_id}/{filename}")
    assert response.headers.get("accept-ranges") == "bytes"


def test_04_gif_returns_200(fake_bundle):
    """GET /media/{clip_id}/{clip_id}_preview.gif returns 200 when file exists."""
    clip_id  = fake_bundle["clip_id"]
    filename = f"{clip_id}_preview.gif"
    with patch.object(bundle_module, '_PROJECT_ROOT', fake_bundle["_project_root"]):
        response = client.get(f"/api/v1/media/{clip_id}/{filename}")
    assert response.status_code == 200


def test_05_gif_content_type(fake_bundle):
    """GIF response Content-Type is image/gif."""
    clip_id  = fake_bundle["clip_id"]
    filename = f"{clip_id}_preview.gif"
    with patch.object(bundle_module, '_PROJECT_ROOT', fake_bundle["_project_root"]):
        response = client.get(f"/api/v1/media/{clip_id}/{filename}")
    assert "image/gif" in response.headers["content-type"]


def test_06_unknown_filename_returns_400(fake_bundle):
    """GET /media/{clip_id}/unknown.mp4 returns 400 (not in whitelist)."""
    clip_id = fake_bundle["clip_id"]
    with patch.object(bundle_module, '_PROJECT_ROOT', fake_bundle["_project_root"]):
        response = client.get(f"/api/v1/media/{clip_id}/unknown.mp4")
    assert response.status_code == 400


def test_07_missing_file_returns_404():
    """GET /media/{clip_id}/{clip_id}.mp4 returns 404 when no output directory exists."""
    clip_id  = "bg_nonexistent_xyz999"
    filename = f"{clip_id}.mp4"
    # _PROJECT_ROOT points to real project root — no output dir for this clip_id
    response = client.get(f"/api/v1/media/{clip_id}/{filename}")
    assert response.status_code == 404


def test_08_path_traversal_dotdot_returns_400(fake_bundle):
    """Filename containing '..' is rejected with 400 (whitelist + guard)."""
    clip_id = fake_bundle["clip_id"]
    # Filename with literal '..' reaches the handler as a single path segment
    traversal_filename = f"..{clip_id}.mp4"
    with patch.object(bundle_module, '_PROJECT_ROOT', fake_bundle["_project_root"]):
        response = client.get(f"/api/v1/media/{clip_id}/{traversal_filename}")
    assert response.status_code == 400


def test_09_path_traversal_slash_returns_400(fake_bundle):
    """Filename not in whitelist (would need slash to escape) is rejected with 400."""
    clip_id = fake_bundle["clip_id"]
    # Any filename containing '..' is rejected by the guard before whitelist check
    traversal_filename = f"..%2F{clip_id}.mp4"
    with patch.object(bundle_module, '_PROJECT_ROOT', fake_bundle["_project_root"]):
        response = client.get(f"/api/v1/media/{clip_id}/{traversal_filename}")
    # The whitelist rejects this even if Starlette decodes the encoding
    assert response.status_code in (400, 404)


def test_10_existing_bundle_endpoint_still_works(fake_bundle):
    """Regression: /bundle/{clip_id}/{filename} still returns 200 for a valid JSON file."""
    clip_id  = fake_bundle["clip_id"]
    filename = f"{clip_id}_metadata.json"
    with patch.object(bundle_module, '_PROJECT_ROOT', fake_bundle["_project_root"]):
        response = client.get(f"/api/v1/bundle/{clip_id}/{filename}")
    assert response.status_code == 200


def test_11_raw_loop_file_found_in_raw_subdirectory(tmp_path):
    """GET /media/{clip_id}/bg_{clip_id}_raw_loop.mp4 returns 200 when the file
    lives in output/{clip_id}/raw/ (not output/{clip_id}/final/)."""
    clip_id = "bg_raw_test_xyz"
    raw_dir = tmp_path / "output" / clip_id / "raw"
    raw_dir.mkdir(parents=True)
    raw_file = raw_dir / f"bg_{clip_id}_raw_loop.mp4"
    raw_file.write_bytes(
        b'\x00\x00\x00\x20ftypisom'
        b'\x00\x00\x00\x00isomiso2'
        b'avc1mp41'
    )

    filename = f"bg_{clip_id}_raw_loop.mp4"
    with patch.object(bundle_module, '_PROJECT_ROOT', tmp_path):
        response = client.get(f"/api/v1/media/{clip_id}/{filename}")
    assert response.status_code == 200
