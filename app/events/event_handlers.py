"""Event module to configure startup and shutdown process"""
from typing import Callable
from fastapi import FastAPI
from decouple import config

from core import Model


def _startup_model(app: FastAPI) -> None:
    """Initialise API status objects

    Args:
        app (FastAPI): FastAPI object
    """
    app.state.model = Model(
        filename=config("MODEL_FILENAME", cast=str, default="weights/doubleit_model.pt")
    )


def start_app_handler(app: FastAPI) -> Callable:
    """Start app event

    Args:
        app (FastAPI): FastAPI object

    Returns:
        Callable: Start app function
    """

    def startup() -> None:
        _startup_model(app)

    return startup


def _shutdown_model(app: FastAPI) -> None:
    """Delete API status objects

    Args:
        app (FastAPI): FastAPI object
    """
    app.state.model = None


def stop_app_handler(app: FastAPI) -> Callable:
    """Stop app event

    Args:
        app (FastAPI): FastAPI object

    Returns:
        Callable: Stop app function
    """

    def shutdown() -> None:
        _shutdown_model(app)

    return shutdown
