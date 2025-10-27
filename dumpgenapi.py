#!/usr/bin/env python

import pycameleon
import re
from pathlib import Path


def sanitize_filename(name: str) -> str:
    """Convert any string into a safe filename."""
    # Replace invalid filename characters with underscores
    safe = re.sub(r'[<>:"/\\|?*\x00-\x1F]', "_", name)
    # Optional: trim spaces and limit length
    safe = safe.strip().replace(" ", "_")[:255]
    return safe or "untitled"


def write_string_to_file(name: str, content: str, directory: str = ".") -> Path:
    """Sanitize filename and write content safely."""
    safe_name = sanitize_filename(name)
    path = Path(directory) / f"{safe_name}.xml"
    path.write_text(content, encoding="utf-8")
    return path


if __name__ == "__main__":
    for cam in pycameleon.enumerate_cameras():
        try:
            cam.open()
            genapi = cam.load_context()
            filename = sanitize_filename(str(cam))
            write_string_to_file(filename, genapi)
        finally:
            cam.close()
