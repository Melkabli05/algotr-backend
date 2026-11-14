from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware


def allow_origins(app: FastAPI, origins: list[str]) -> None:
    """Lets a terminal served from another origin call the API; with no origins listed, none may."""
    if origins:
        app.add_middleware(
            CORSMiddleware, allow_origins=origins, allow_methods=["*"], allow_headers=["*"]
        )
