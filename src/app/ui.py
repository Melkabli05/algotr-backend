from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, RedirectResponse

BUILD = Path("frontend/dist")

router = APIRouter(include_in_schema=False)


@router.get("/ui")
def ui_root() -> RedirectResponse:
    return RedirectResponse("/ui/")


@router.get("/ui/{path:path}")
def ui(path: str) -> FileResponse:
    """A built file of the terminal, or its page for any route the browser asks for."""
    root = BUILD.resolve()
    if not (root / "index.html").exists():
        raise HTTPException(status_code=404, detail="Build the terminal: pnpm --dir frontend build")
    found = (root / path).resolve()
    if path and found.is_file() and found.is_relative_to(root):
        return FileResponse(found)
    return FileResponse(root / "index.html")
