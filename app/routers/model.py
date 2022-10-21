"""Hello name module"""
from typing import List
from fastapi import APIRouter, Body
from starlette.requests import Request
from pydantic import BaseModel, Field
import torch

from core import Model

router = APIRouter(prefix="", tags=["model"])


class PredictionInput(BaseModel):
    """Prediction Input Schema"""

    values: List[int] = Field(min_items=4, max_items=4)


class PredictionOutput(BaseModel):
    """Prediction Output Schema"""

    values: List[int] = Field(min_items=4, max_items=4)


@router.post("/prediction", response_model=PredictionOutput)
async def post_prediction(
    request: Request, features: PredictionInput = Body()
) -> PredictionOutput:
    """Endpoint to return prediction result

    Args:
        features (PredictionInput): input features

    Returns:
        PredictionOutput: prediction
    """
    model: Model = request.app.state.model
    return {"values": model.predict(torch.Tensor(features.values)).tolist()}
