"""Model inventory response schemas."""

from __future__ import annotations

from pydantic import BaseModel


class LocalArtifact(BaseModel):
    name: str
    path: str
    exists: bool
    size_bytes: int | None = None


class RegisteredModelVersion(BaseModel):
    name: str
    version: str
    current_stage: str | None = None
    run_id: str | None = None
    source: str | None = None


class DatabaseModelVersion(BaseModel):
    version: str
    accuracy: float | None = None
    created_at: str


class ModelsResponse(BaseModel):
    supported_models: list[str]
    local_artifacts: list[LocalArtifact]
    registered_models: list[RegisteredModelVersion]
    database_models: list[DatabaseModelVersion]
