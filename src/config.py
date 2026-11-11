from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import find_dotenv, load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
APP_DB = DATA_DIR / "customer_service.db"
BENCHMARK_WORKLOAD = DATA_DIR / "optimization_workload.json"
MODEL_RATES = DATA_DIR / "model_rates.json"
MODEL_RATES_EXAMPLE = DATA_DIR / "model_rates.example.json"


@dataclass(frozen=True)
class LabSettings:
    api_key: str
    fast_model: str
    capable_model: str
    base_url: str | None = None


@dataclass(frozen=True)
class ModelRate:
    input_per_1m: float
    cached_input_per_1m: float
    output_per_1m: float


@dataclass(frozen=True)
class RateCard:
    currency: str
    as_of: str
    source_url: str
    models: dict[str, ModelRate]

    def estimate(
        self,
        model: str,
        input_tokens: int,
        cached_input_tokens: int,
        output_tokens: int,
    ) -> float:
        rate = self.models[model]
        cached = min(input_tokens, max(0, cached_input_tokens))
        uncached = max(0, input_tokens - cached)
        value = (
            uncached * rate.input_per_1m
            + cached * rate.cached_input_per_1m
            + output_tokens * rate.output_per_1m
        ) / 1_000_000
        return round(value, 8)


def load_settings() -> LabSettings:
    dotenv_path = find_dotenv(usecwd=True)
    if dotenv_path:
        load_dotenv(dotenv_path, override=False)

    api_key = os.getenv("API_KEY") or os.getenv("OPENAI_API_KEY")
    fast_model = os.getenv("FAST_MODEL")
    capable_model = os.getenv("CAPABLE_MODEL")
    missing = [
        name
        for name, value in (
            ("API_KEY", api_key),
            ("FAST_MODEL", fast_model),
            ("CAPABLE_MODEL", capable_model),
        )
        if not value
    ]
    if missing:
        raise RuntimeError(
            "Missing lab configuration: " + ", ".join(missing) + ". "
            "Copy .env.example to .env and fill in the live-lab values."
        )
    if fast_model == capable_model:
        raise RuntimeError(
            "FAST_MODEL and CAPABLE_MODEL must use different model IDs so the "
            "routing experiment has a measurable model choice."
        )
    return LabSettings(
        api_key=str(api_key),
        fast_model=str(fast_model),
        capable_model=str(capable_model),
        base_url=os.getenv("API_BASE_URL") or os.getenv("OPENAI_BASE_URL"),
    )


def load_rate_card(path: Path = MODEL_RATES) -> RateCard:
    if not path.exists():
        raise RuntimeError(
            f"Missing {path.name}. Copy {MODEL_RATES_EXAMPLE.name}, then record "
            "the provider source URL, effective date, and current rates."
        )
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not raw.get("currency") or not raw.get("as_of") or not raw.get("source_url"):
        raise ValueError("Rate card requires currency, as_of, and source_url")
    if str(raw["as_of"]).upper() == "YYYY-MM-DD" or "replace" in str(
        raw["source_url"]
    ).casefold():
        raise ValueError("Replace the example rate-card date and source URL before live calls")

    models: dict[str, ModelRate] = {}
    for model, values in raw.get("models", {}).items():
        rates = {
            "input_per_1m": float(values["input_per_1m"]),
            "cached_input_per_1m": float(
                values.get("cached_input_per_1m", values["input_per_1m"])
            ),
            "output_per_1m": float(values["output_per_1m"]),
        }
        if any(value < 0 for value in rates.values()):
            raise ValueError(f"Rates must be non-negative for {model}")
        models[str(model)] = ModelRate(**rates)
    if not models:
        raise ValueError("Rate card must contain at least one model")
    return RateCard(
        currency=str(raw["currency"]),
        as_of=str(raw["as_of"]),
        source_url=str(raw["source_url"]),
        models=models,
    )


def validate_rate_models(settings: LabSettings, rates: RateCard) -> None:
    required = {settings.fast_model, settings.capable_model}
    missing = sorted(required - set(rates.models))
    if missing:
        raise RuntimeError("Rate card is missing configured models: " + ", ".join(missing))
    placeholders = sorted(
        model
        for model in required
        if rates.models[model].input_per_1m == 0
        and rates.models[model].output_per_1m == 0
    )
    if placeholders:
        raise RuntimeError(
            "Configured live models require non-placeholder input or output rates: "
            + ", ".join(placeholders)
        )
