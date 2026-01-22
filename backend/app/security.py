"""
Security utilities for input validation and sanitization.
"""

import os
import re
from pathlib import Path
from typing import List, Set
from fastapi import UploadFile, HTTPException
from loguru import logger

# File upload limits
MAX_IMAGE_SIZE = 10 * 1024 * 1024  # 10MB
MAX_VIDEO_SIZE = 500 * 1024 * 1024  # 500MB
MAX_JSON_SIZE = 5 * 1024 * 1024  # 5MB

# Allowed MIME types
ALLOWED_IMAGE_TYPES = {
    "image/png",
    "image/jpeg",
    "image/jpg",
    "image/webp"
}

ALLOWED_VIDEO_TYPES = {
    "video/mp4",
    "video/quicktime",
    "video/x-msvideo",
    "video/webm"
}

ALLOWED_JSON_TYPES = {
    "application/json",
    "text/json"
}


def sanitize_filename(filename: str) -> str:
    """
    Sanitize filename to prevent path traversal attacks.

    Removes directory components and dangerous characters.

    Args:
        filename: Original filename

    Returns:
        Sanitized filename safe for use

    Example:
        >>> sanitize_filename("../../etc/passwd")
        'passwd'
        >>> sanitize_filename("file<>name.txt")
        'filename.txt'
    """
    # Remove any path components (directory traversal attempt)
    filename = os.path.basename(filename)

    # Remove or replace dangerous characters
    # Allow: alphanumeric, dash, underscore, period
    filename = re.sub(r'[^a-zA-Z0-9._-]', '', filename)

    # Prevent hidden files
    if filename.startswith('.'):
        filename = filename[1:]

    # Ensure filename is not empty
    if not filename:
        filename = "unnamed_file"

    # Limit length
    if len(filename) > 255:
        name, ext = os.path.splitext(filename)
        filename = name[:250] + ext

    return filename


def validate_path_traversal(path: str, base_dir: str) -> Path:
    """
    Validate that a path doesn't escape the base directory.

    Args:
        path: Path to validate
        base_dir: Base directory that path should stay within

    Returns:
        Resolved Path object

    Raises:
        HTTPException: If path attempts to escape base directory
    """
    base = Path(base_dir).resolve()
    target = (base / path).resolve()

    # Check if target is within base directory
    if not str(target).startswith(str(base)):
        logger.warning(f"Path traversal attempt detected: {path} escapes {base_dir}")
        raise HTTPException(
            status_code=400,
            detail="Invalid path: directory traversal detected"
        )

    return target


async def validate_image_upload(file: UploadFile) -> bool:
    """
    Validate uploaded image file.

    Checks:
    - File size
    - Content type
    - File extension

    Args:
        file: Uploaded file

    Returns:
        True if valid

    Raises:
        HTTPException: If validation fails
    """
    # Check file size
    content = await file.read()
    await file.seek(0)  # Reset file pointer

    if len(content) > MAX_IMAGE_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"Image too large. Maximum size: {MAX_IMAGE_SIZE / (1024*1024):.1f}MB"
        )

    # Check content type
    if file.content_type not in ALLOWED_IMAGE_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid image type: {file.content_type}. Allowed: {', '.join(ALLOWED_IMAGE_TYPES)}"
        )

    # Validate file has content
    if len(content) == 0:
        raise HTTPException(
            status_code=400,
            detail="Empty file uploaded"
        )

    # Check for valid image format by reading header
    # PNG magic number: \x89PNG
    # JPEG magic number: \xff\xd8\xff
    # WebP magic number: RIFF....WEBP
    is_png = content[:4] == b'\x89PNG'
    is_jpeg = content[:3] == b'\xff\xd8\xff'
    is_webp = content[:4] == b'RIFF' and content[8:12] == b'WEBP'

    if not (is_png or is_jpeg or is_webp):
        raise HTTPException(
            status_code=400,
            detail="File does not appear to be a valid image (magic bytes check failed)"
        )

    logger.info(f"Image validated: {file.filename} ({len(content)} bytes)")
    return True


async def validate_video_upload(file: UploadFile) -> bool:
    """
    Validate uploaded video file.

    Args:
        file: Uploaded file

    Returns:
        True if valid

    Raises:
        HTTPException: If validation fails
    """
    # Check file size
    content = await file.read()
    await file.seek(0)

    if len(content) > MAX_VIDEO_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"Video too large. Maximum size: {MAX_VIDEO_SIZE / (1024*1024):.0f}MB"
        )

    # Check content type
    if file.content_type not in ALLOWED_VIDEO_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid video type: {file.content_type}. Allowed: {', '.join(ALLOWED_VIDEO_TYPES)}"
        )

    # Validate file has content
    if len(content) == 0:
        raise HTTPException(
            status_code=400,
            detail="Empty file uploaded"
        )

    # Check for valid video format by reading header
    # MP4: starts with ftyp
    # MOV: starts with ftyp qt
    # AVI: starts with RIFF....AVI
    # WebM: starts with \x1A\x45\xDF\xA3
    is_mp4_mov = b'ftyp' in content[:20]
    is_avi = content[:4] == b'RIFF' and content[8:11] == b'AVI'
    is_webm = content[:4] == b'\x1A\x45\xDF\xA3'

    if not (is_mp4_mov or is_avi or is_webm):
        raise HTTPException(
            status_code=400,
            detail="File does not appear to be a valid video (magic bytes check failed)"
        )

    logger.info(f"Video validated: {file.filename} ({len(content)} bytes)")
    return True


async def validate_json_upload(file: UploadFile) -> bool:
    """
    Validate uploaded JSON file.

    Args:
        file: Uploaded file

    Returns:
        True if valid

    Raises:
        HTTPException: If validation fails
    """
    import json

    # Check file size
    content = await file.read()
    await file.seek(0)

    if len(content) > MAX_JSON_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"JSON file too large. Maximum size: {MAX_JSON_SIZE / (1024*1024):.0f}MB"
        )

    # Validate JSON structure
    try:
        json.loads(content)
    except json.JSONDecodeError as e:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid JSON file: {str(e)}"
        )

    logger.info(f"JSON validated: {file.filename} ({len(content)} bytes)")
    return True


def sanitize_session_id(session_id: str) -> str:
    """
    Sanitize session ID to prevent injection attacks.

    Args:
        session_id: Session identifier

    Returns:
        Sanitized session ID

    Raises:
        HTTPException: If session ID is invalid
    """
    # Only allow alphanumeric and hyphens
    if not re.match(r'^[a-zA-Z0-9-]{8,64}$', session_id):
        raise HTTPException(
            status_code=400,
            detail="Invalid session ID format"
        )
    return session_id


def sanitize_manual_id(manual_id: str) -> str:
    """
    Sanitize manual ID to prevent injection attacks.

    Args:
        manual_id: Manual identifier

    Returns:
        Sanitized manual ID

    Raises:
        HTTPException: If manual ID is invalid
    """
    # Only allow alphanumeric, hyphens, and underscores
    if not re.match(r'^[a-zA-Z0-9_-]{1,100}$', manual_id):
        raise HTTPException(
            status_code=400,
            detail="Invalid manual ID format"
        )
    return manual_id
