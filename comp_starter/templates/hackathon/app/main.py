"""FastAPI application for {{ project_name }}."""

from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pathlib import Path

app = FastAPI(
    title="{{ project_name }} API",
    description="Hackathon API for model serving",
    version="0.1.0",
)


class PredictRequest(BaseModel):
    """Request schema for predictions."""
    features: list[dict]


class PredictResponse(BaseModel):
    """Response schema for predictions."""
    predictions: list
    model_version: str = "0.1.0"


@app.get("/")
def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "{{ project_name }}"}


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    """Make predictions using the trained model."""
    # TODO: Load and use your trained model
    # model = load_model("best")
    # predictions = model.predict(features)
    return PredictResponse(
        predictions=[0] * len(request.features),
        model_version="0.1.0",
    )


@app.get("/docs")
def api_docs():
    """Redirect to API docs."""
    return {"message": "Visit /redoc or /docs for interactive API documentation"}
