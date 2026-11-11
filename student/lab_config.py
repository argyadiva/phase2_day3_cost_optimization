"""The only file participants edit during the Module 13 hands-on.

Keep the three exercises cumulative. Start from the runnable baseline below, then:
1. compact the prompt,
2. implement model routing,
3. reduce retrieval top-k.
"""

from __future__ import annotations

from src.config import LabSettings
from src.models import AgentRequest
from src.prompt_contract import REFERENCE_INSTRUCTIONS


# STEP 1 — PROMPT COMPACTION
# Replace ACTIVE_INSTRUCTIONS with a prompt at least 25% shorter than the
# reference while preserving every safety contract listed in README.md.
PROMPT_VERSION = "manual-baseline-v1"
ACTIVE_INSTRUCTIONS = REFERENCE_INSTRUCTIONS


# STEP 3 — RETRIEVAL TOP-K
# Keep 3 for the baseline and first two experiments. Change it only during the
# final top-k exercise.
RETRIEVAL_TOP_K = 3


PUBLIC_FAST_CATEGORIES = {"returns", "delivery", "warranty", "support_hours"}


def select_model(request: AgentRequest, settings: LabSettings) -> str:
    """Return the model for one request.

    STEP 2 — MODEL ROUTING
    The baseline deliberately sends every request to the capable model. Replace
    the return statement with an explicit safe routing rule. Only anonymous,
    public, allow-listed requests may use the fast model. There is no automatic
    fallback in this lab, so routing failures remain visible in the evidence.
    """

    return settings.capable_model
