"""Runtime settings. Override any field with KNOWSURE_<FIELD_NAME>; the API key comes from GEMINI_API_KEY."""
from __future__ import annotations

import os
from dataclasses import dataclass, field, fields
from pathlib import Path

SLM_PROVIDERS = ("gemini", "ollama", "hf")


def load_dotenv(path: Path = Path(".env")) -> None:
    """Minimal .env reader (KEY=VALUE lines). Variables already set in the environment win."""
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def _parse(raw: str, default):
    if isinstance(default, bool):  # bool("false") is True, so parse booleans explicitly
        if raw.strip().lower() in ("1", "true", "yes", "on"):
            return True
        if raw.strip().lower() in ("0", "false", "no", "off"):
            return False
        raise ValueError(f"expected true/false, got {raw!r}")
    return type(default)(raw)


@dataclass(frozen=True)
class Settings:
    # ------------------------------------------------------------ SLM
    # Where the SLM runs: "gemini" (Gemini API), "ollama" (on-device via Ollama) or "hf" (on-device via transformers).
    slm_provider: str = "gemini"
    # Gemini API model. gemma-4-26b-a4b-it is an open Gemma model that activates ~4B parameters per token.
    # Run scripts/list_models.py to see what your key can use.
    gemini_api_key: str = field(default="", repr=False)  # never printed or returned by the API
    slm_model: str = "gemma-4-26b-a4b-it"
    gemini_base_url: str = "https://generativelanguage.googleapis.com/v1beta"
    thinking_level: str = "minimal"  # Gemma 4 / Gemini 3 reasoning effort; "" = don't send thinkingConfig
    # On-device SLMs
    ollama_base_url: str = "http://localhost:11434/v1"
    ollama_model: str = "gemma3:1b"
    hf_model: str = "Qwen/Qwen2.5-0.5B-Instruct"
    max_output_tokens: int = 128
    request_timeout: float = 30.0

    # ------------------------------------------------------------ evidence (live Wikipedia, nothing stored)
    wikipedia_lang: str = "en"
    wiki_contact: str = ""  # your email or project URL; Wikipedia rejects requests without one
    wiki_search_results: int = 5  # articles per search; lead sections only in the first round
    wiki_max_chars: int = 6000    # per article lead
    # Adaptive retrieval: if a claim isn't settled by article leads, search for the claim itself and read full articles.
    adaptive_retrieval: bool = True
    adaptive_full_articles: int = 2
    wiki_full_max_chars: int = 20000

    # ------------------------------------------------------------ verification
    # Passage ranking: "sentence-transformers" (neural, best quality), "fastembed" (same model, no PyTorch) or
    # "lexical" (word matching: no model, ~20x faster; for tiny servers such as Render's free plan).
    embed_backend: str = "sentence-transformers"
    embed_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    # Lexical scores run lower than neural ones, so the lexical backend uses its own thresholds and passes a few
    # more passages to the verifier.
    lexical_min_retrieval_score: float = 0.10
    lexical_topic_similarity: float = 0.15
    lexical_top_k: int = 6
    # Comma-separated verifier ensemble. "llm" = the SLM labels evidence sentences; "llm:<gemini model>" = another
    # Gemini API model does it (independent of the SLM); "nli" = the local NLI model below. Scores are averaged.
    # Examples: "llm,llm:gemini-3.6-flash" (independent judge; the free tier allows only ~20 requests/day) or
    # "llm,nli" (local, no quota; downloads ~370 MB on first use). Default: one judge, which fits the free tier.
    verifiers: str = "llm"
    nli_model: str = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"  # ~370 MB, robust to adversarial claims
    verifier_agreement_threshold: float = 0.5  # below this, verifiers disagree and KNOWN becomes UNCERTAIN
    verifier_timeout: float = 12.0  # seconds; a slower verifier is skipped for that check and reported

    # Claim-level detection: split the answer into atomic claims and verify each one.
    claim_level: bool = True
    max_claims: int = 4

    # Paraphrase consistency: re-ask the question in N other wordings and check the answers agree.
    paraphrases: int = 2                 # 0 disables
    consistency_threshold: float = 0.67  # below this, KNOWN becomes UNCERTAIN

    # Calibration: maps the signals to a confidence. Fit it with scripts/calibrate.py; until then it's uncalibrated.
    calibration_file: str = "data/calibration.json"

    # Public deployment: usage limits protect the shared Gemini key (0 = off). Visitors are told when to retry.
    rate_limit_per_minute: int = 0
    rate_limit_per_day: int = 0
    global_daily_limit: int = 0
    log_runs: bool = True  # write questions/answers to data/runs/runs.jsonl (turned off on the public Space)
    static_dir: str = ""   # if set, also serve the built website from this folder (single-URL deployment)

    data_dir: str = "data"  # run logs are written here
    chunk_words: int = 80
    chunk_overlap: int = 20
    top_k: int = 4

    # Decision thresholds. Starting points only: tune them on the benchmark with scripts/evaluate.py.
    min_retrieval_score: float = 0.30     # below this, a passage is treated as irrelevant
    topic_similarity: float = 0.50        # a sentence must be this similar to the claim for its contradiction to count
    max_verify_sentences: int = 10        # only the most on-topic sentences go to the verifier (speed)
    known_support: float = 0.70           # entailment needed for KNOWN
    uncertain_support: float = 0.35       # entailment needed for UNCERTAIN
    conflict_contradiction: float = 0.60  # contradiction that counts as a conflict
    conflict_decision: str = "UNCERTAIN"  # what a conflict becomes: "UNCERTAIN" or "UNKNOWN"

    @classmethod
    def from_env(cls, dotenv: Path = Path(".env")) -> "Settings":
        load_dotenv(dotenv)
        values = {}
        for f in fields(cls):
            raw = os.getenv(f"KNOWSURE_{f.name.upper()}")
            if raw not in (None, ""):
                values[f.name] = _parse(raw, f.default)
        values.setdefault("gemini_api_key", os.getenv("GEMINI_API_KEY", "").strip())
        settings = cls(**values)
        settings.validate()
        return settings

    def validate(self) -> None:
        if self.embed_backend not in ("sentence-transformers", "fastembed", "lexical"):
            raise ValueError("KNOWSURE_EMBED_BACKEND must be sentence-transformers, fastembed or lexical, "
                             f"got {self.embed_backend!r}")
        if self.conflict_decision not in ("UNCERTAIN", "UNKNOWN"):
            raise ValueError("KNOWSURE_CONFLICT_DECISION must be UNCERTAIN or UNKNOWN")
        if self.slm_provider not in SLM_PROVIDERS:
            raise ValueError(f"KNOWSURE_SLM_PROVIDER must be one of {SLM_PROVIDERS}")
        for spec in self.verifier_specs:
            if spec != "nli" and spec != "llm" and not spec.startswith("llm:"):
                raise ValueError(f"Unknown verifier {spec!r} in KNOWSURE_VERIFIERS (use llm, llm:<model> or nli)")
        if not self.verifier_specs:
            raise ValueError("KNOWSURE_VERIFIERS must name at least one verifier")

    @property
    def verifier_specs(self) -> list[str]:
        return [v.strip() for v in self.verifiers.split(",") if v.strip()]

    @property
    def active_slm(self) -> str:
        return {"gemini": self.slm_model, "ollama": self.ollama_model, "hf": self.hf_model}[self.slm_provider]

    @property
    def runs_log(self) -> Path:
        return Path(self.data_dir) / "runs" / "runs.jsonl"
