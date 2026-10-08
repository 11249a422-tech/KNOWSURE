"""FastAPI app. Run with:  uvicorn app.main:app --reload

The API is served at both / and /api, so the same build of the website works behind the Vite dev proxy (/api)
and when this server also serves the website itself (KNOWSURE_STATIC_DIR, e.g. on Hugging Face Spaces).
"""
from __future__ import annotations

import os
import sys
import threading
from pathlib import Path

from fastapi import APIRouter, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from .config import Settings
from .generator import SLMError, SLMNotConfigured
from .pipeline import KnowSure, build_pipeline
from .ratelimit import RateLimited, RateLimiter
from .schemas import AskRequest, AskResponse, HealthResponse


def visitor_id(request: Request) -> str:
    """The visitor's IP. Behind a proxy (Hugging Face Spaces) the original address is in X-Forwarded-For."""
    forwarded = request.headers.get("x-forwarded-for", "")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def create_app(pipeline: KnowSure | None = None, warm_up: bool = False) -> FastAPI:
    ks = pipeline or build_pipeline(Settings.from_env())
    s = ks.settings
    limiter = RateLimiter(s.rate_limit_per_minute, s.rate_limit_per_day, s.global_daily_limit)
    if warm_up:
        # Load the local models in the background at startup, so the first real question isn't slow.
        threading.Thread(target=ks.warmup, daemon=True, name="knowsure-warmup").start()

    app = FastAPI(title="KnowSure API", version="0.3.0",
                  description="Reliability layer for a pretrained Small Language Model, verified against live "
                              "Wikipedia evidence: answer, verify, or abstain.")
    # Website (Vite dev server), desktop app (app://knowsure) and Android app (Capacitor's local origins).
    origins = os.getenv("KNOWSURE_CORS_ORIGINS", "http://localhost:5173,http://localhost:3000,app://knowsure,"
                                                 "http://localhost,https://localhost,capacitor://localhost").split(",")
    app.add_middleware(CORSMiddleware, allow_origins=[o.strip() for o in origins], allow_methods=["*"],
                       allow_headers=["*"])

    api = APIRouter()

    @api.get("/health", response_model=HealthResponse)
    def health():
        # The Gemini key is only needed when the SLM or a verifier runs on the Gemini API.
        key_needed = s.slm_provider == "gemini" or any(v.startswith("llm:") for v in s.verifier_specs)
        return HealthResponse(
            status="ok", slm_provider=s.slm_provider, slm_model=s.active_slm,
            api_key_configured=bool(s.gemini_api_key) or not key_needed,
            evidence_source=f"wikipedia ({s.wikipedia_lang})", wikipedia_contact_configured=bool(s.wiki_contact),
            verifier=", ".join(ks.verifier.names), verifiers=ks.verifier.names,
            features={"claim_level": s.claim_level, "paraphrases": s.paraphrases,
                      "adaptive_retrieval": s.adaptive_retrieval, "multiple_verifiers": len(ks.verifier.names) > 1,
                      "on_device_slm": s.slm_provider != "gemini"},
            confidence_calibrated=ks.calibrator.calibrated, limits=limiter.status(),
            embed_model=s.embed_model, nli_model=s.nli_model)

    @api.post("/ask", response_model=AskResponse)
    def ask(body: AskRequest, request: Request):
        try:
            limiter.check(visitor_id(request))
        except RateLimited as exc:
            raise HTTPException(429, str(exc), headers={"Retry-After": str(exc.retry_after)}) from exc
        try:
            return ks.ask(body.question.strip(), body.mode)
        except SLMNotConfigured as exc:
            raise HTTPException(503, str(exc)) from exc
        except SLMError as exc:
            raise HTTPException(502, str(exc)) from exc

    @api.post("/warmup")
    def warmup() -> dict[str, str]:
        ks.warmup()
        return {"status": "local models loaded"}

    app.include_router(api)
    app.include_router(api, prefix="/api")

    # Single-URL deployment: serve the built website too (registered last, so API routes win).
    if s.static_dir and (Path(s.static_dir) / "index.html").is_file():
        app.mount("/", StaticFiles(directory=s.static_dir, html=True), name="website")

    return app


app = create_app(warm_up="pytest" not in sys.modules)
