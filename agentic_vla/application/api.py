"""Authenticated local HTTP access to an explicitly supplied tool service."""

import secrets
from typing import Annotated, Any

from fastapi import Depends, FastAPI, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, ConfigDict, Field, StrictInt, StrictStr

from .control_plane import (
    ExecutionDisabled,
    LangGraphToolService,
    RequestConflict,
    RequestInvalid,
)


class ToolRequestBody(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    request_id: StrictStr = Field(
        min_length=1, max_length=128, pattern=r"^[A-Za-z0-9_.:-]+$"
    )
    name: StrictStr = Field(min_length=1, max_length=64)
    arguments: dict[str, Any]
    expected_episode_id: StrictStr | StrictInt
    expected_timestep: StrictInt = Field(ge=0)
    schema_version: StrictInt = 1


def create_app(service: LangGraphToolService, *, api_token: str) -> FastAPI:
    """Use one worker and localhost; no arbitrary model/process/task launching.

    The owner must initialize the session and coordinate environment stepping
    at safe boundaries. This API is not a replacement for the simulator bridge.
    """

    if not api_token.isascii() or len(api_token) < 32:
        raise ValueError("use an ASCII API token of at least 32 characters")
    bearer = HTTPBearer(auto_error=False)

    def authenticate(
        credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(bearer)],
    ) -> None:
        if credentials is None or not secrets.compare_digest(
            credentials.credentials.encode(), api_token.encode()
        ):
            raise HTTPException(
                status_code=401,
                detail="invalid credentials",
                headers={"WWW-Authenticate": "Bearer"},
            )

    app = FastAPI(
        title="Agentic VLA Tool Service",
        version="1.0",
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {
            "service": "agentic-vla-tools",
            "status": "ready",
            "robot_health_checked": False,
        }

    @app.get("/openapi.json", dependencies=[Depends(authenticate)])
    def schema() -> dict[str, Any]:
        return app.openapi()

    @app.get("/tools", dependencies=[Depends(authenticate)])
    def tools() -> dict[str, Any]:
        return service.catalog()

    @app.post("/requests", dependencies=[Depends(authenticate)])
    def invoke(body: ToolRequestBody) -> dict[str, Any]:
        try:
            return service.invoke(body.model_dump())
        except ExecutionDisabled as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc
        except RequestInvalid as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except RequestConflict as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.get("/requests/{request_id}", dependencies=[Depends(authenticate)])
    def request_status(request_id: str) -> dict[str, Any]:
        try:
            return service.ledger.get(request_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="unknown request") from exc

    return app
