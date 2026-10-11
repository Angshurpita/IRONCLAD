"""Answer generation orchestrator with robust JSON extraction and citation validation."""

import json
import logging
import re
import time
from typing import Any, Dict, List

from ironclad.errors import InferenceError
from ironclad.generation.citations import validate_citations
from ironclad.generation.prompts import build_rag_user_prompt, build_system_prompt
from ironclad.generation.provider import BaseInferenceProvider
from ironclad.models import RAGResponse, SearchResult

logger = logging.getLogger("ironclad.gemma")

JSON_FENCE_REGEX = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)


def extract_json_payload(raw_text: str) -> Dict[str, Any]:
    """
    Robustly extract JSON object from LLM output, handling markdown fences or leading commentary.
    """
    cleaned = raw_text.strip()

    # 1. Direct parse attempt
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # 2. Markdown fence extraction
    match = JSON_FENCE_REGEX.search(cleaned)
    if match:
        try:
            return json.loads(match.group(1).strip())
        except json.JSONDecodeError:
            pass

    # 3. First opening curly brace to last closing curly brace
    first_brace = cleaned.find("{")
    last_brace = cleaned.rfind("}")
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        candidate = cleaned[first_brace : last_brace + 1]
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

    raise InferenceError(
        f"Model did not return a valid structured JSON object. Raw response:\n{raw_text[:300]}",
        hint="The model output was malformed. Please try re-running the query.",
    )


def generate_grounded_answer(
    query: str,
    evidence: List[SearchResult],
    provider: BaseInferenceProvider,
) -> RAGResponse:
    """
    Generate an evidence-grounded answer using Gemma with independent citation validation.
    
    If no evidence is supplied, immediately abstains without calling the provider.
    """
    # Immediate abstention if evidence is completely empty
    if not evidence:
        return RAGResponse(
            query=query,
            answer="I am unable to answer this question because no relevant evidence was found in the indexed knowledge base.",
            citations=[],
            insufficient_evidence=True,
            retrieved_chunks_count=0,
            model_used=provider.model_name,
            latency_ms=0.0,
        )

    system_prompt = build_system_prompt()
    user_prompt = build_rag_user_prompt(query, evidence)

    start_time = time.time()
    raw_response = provider.generate(prompt=user_prompt, system_prompt=system_prompt)
    latency_ms = round((time.time() - start_time) * 1000.0, 2)

    parsed_json = extract_json_payload(raw_response)

    answer_text = str(parsed_json.get("answer", "")).strip()
    raw_citations = parsed_json.get("citations", [])
    model_claimed_insufficient = bool(parsed_json.get("insufficient_evidence", False))

    # Independent verification of citations against retrieved evidence
    report = validate_citations(
        raw_citations=raw_citations,
        evidence=evidence,
        model_claimed_insufficient=model_claimed_insufficient,
    )

    is_insufficient = report.insufficient_evidence or model_claimed_insufficient

    # If evidence is insufficient, ensure answer reflects that
    if is_insufficient and not answer_text:
        answer_text = "The provided knowledge base context does not contain sufficient evidence to answer this question."

    return RAGResponse(
        query=query,
        answer=answer_text,
        citations=report.verified_citations,
        insufficient_evidence=is_insufficient,
        retrieved_chunks_count=len(evidence),
        model_used=provider.model_name,
        latency_ms=latency_ms,
    )
