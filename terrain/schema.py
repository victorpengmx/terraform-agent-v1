"""Base for models whose JSON schema goes to a model (structured outputs, tool parameters)."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class Strict(BaseModel):
    """Unknown fields are an error, so the schema says additionalProperties: false (structured outputs need it)."""

    model_config = ConfigDict(extra="forbid")
