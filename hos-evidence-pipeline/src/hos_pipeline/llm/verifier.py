"""Stage 5: strict second reader (grade A).

For each coded decision, every hypothesis coded "yes" is re-tested against
only its cited paragraphs (with their section headings) and the codebook
definition. All of a decision's claims go in one request to share context,
but each is judged on its own citations. Grade "A" iff the verdict is
"verified"; everything else stays grade "B" and is still written out.

Deterministic checks: a claim with no citation to an existing paragraph is
"insufficient_citation" without an API call; ``supporting_pos`` is limited to
the claim's own citations, and a "verified" verdict with no supporting
paragraph among them is downgraded to "not_verified". A claim the model
returns no verdict for is logged and retried on the next run.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationError

from ..config import Settings
from ..io_utils import append_jsonl, read_jsonl
from ..models import ClaimVerification, CodedDecision, Decision, HypothesisCode
from .client import (
    LLMRequest,
    LLMResult,
    LLMRunner,
    build_params,
    cache_ttl_for,
    estimate_cost,
    log,
    make_custom_ids,
    write_preview,
)
from .codebook import Codebook, load_codebook
from .coder import load_decisions, load_prompt, render_header, render_paragraphs

STAGE = "verification"
OUTPUT_TOKENS_ASSUMED = 3000  # per decision, thinking included; for the cost estimate only

Verdict = Literal["verified", "not_verified", "insufficient_citation"]


class ClaimVerdictOut(BaseModel):
    claim_id: str = Field(description="The claim's hypothesis id, e.g. H3")
    verdict: Verdict
    supporting_pos: list[int] = Field(description="Positions among this claim's citations that carry the finding")
    reason: str = Field(description="One or two sentences")


class VerificationOutput(BaseModel):
    verdicts: list[ClaimVerdictOut]


def verifier_system_prompt(codebook: Codebook) -> str:
    return load_prompt("verifier_system.md") + "\n" + codebook.render_markdown(["rules", "hypotheses"])


def cited_positions(claim: HypothesisCode, decision: Decision) -> list[int]:
    """The claim's cited positions that exist in the decision, in order."""
    existing = {p.pos for p in decision.paragraphs}
    return sorted({c.pos for c in claim.citations if c.pos in existing})


def verification_user_message(decision: Decision, claims: list[HypothesisCode], codebook: Codebook) -> str:
    blocks = []
    wanted: set[int] = set()
    for h in claims:
        item = codebook.item(h.id)
        pos = cited_positions(h, decision)
        wanted.update(pos)
        blocks.append(
            f"### {h.id}: {item.label}\nDefinition: {' '.join(item.definition.split())}\n"
            f"Cited paragraphs: {', '.join(f'[p{p}]' for p in pos)}"
        )
    paragraphs = [p for p in decision.paragraphs if p.pos in wanted]
    return (
        f"{render_header(decision)}\n\n<claims>\n" + "\n\n".join(blocks) + "\n</claims>\n\n"
        f"<cited_paragraphs>\n{render_paragraphs(paragraphs)}\n</cited_paragraphs>\n\n"
        "Test each claim against only the paragraphs cited for it and return one verdict per claim."
    )


def grade_verdict(
    case_id: str, claim_id: str, verdict: str, supporting_pos: list[int], cited: list[int], reason: str, verifier: str
) -> ClaimVerification:
    """Apply the deterministic grade rules to one verdict."""
    support = sorted({p for p in supporting_pos if p in set(cited)})
    reason = reason.strip()
    if verdict == "verified" and not support:
        verdict = "not_verified"
        reason += " [check] verified without a supporting paragraph among the claim's citations; downgraded."
    if verdict != "verified":
        support = []
    return ClaimVerification(
        case_id=case_id,
        claim_id=claim_id,
        verdict=verdict,  # type: ignore[arg-type]
        grade="A" if verdict == "verified" else "B",
        supporting_pos=support,
        reason=reason.strip(),
        verifier=verifier,
    )


def _latest_coded(path: Path) -> dict[str, CodedDecision]:
    """Last record per case wins (a re-coded decision replaces the earlier one)."""
    return {c.case_id: c for c in read_jsonl(path, CodedDecision)} if path.exists() else {}


def run_verification(
    settings: Settings,
    case_ids: list[str] | None = None,
    limit: int | None = None,
    dry_run: bool = False,
    mode: str = "auto",
    *,
    client: Any = None,
    async_client: Any = None,
    poll_interval: float = 60.0,
    max_wait: float | None = None,
) -> Path:
    """Second-read every "yes" hypothesis in coded.jsonl into verified.jsonl (resumable)."""
    codebook = load_codebook(settings.paths.codebook)
    out_path = settings.work("verified.jsonl")
    runner = LLMRunner(
        settings, STAGE, client=client, async_client=async_client, poll_interval=poll_interval, max_wait=max_wait
    )
    mode = runner.resolve_mode(mode)
    coded = _latest_coded(settings.work("coded.jsonl"))
    if not coded:
        raise FileNotFoundError(f"{settings.work('coded.jsonl')} is missing or empty; run the code stage first")
    done: set[tuple[str, str]] = (
        {(r["case_id"], r["claim_id"]) for r in read_jsonl(out_path)} if out_path.exists() else set()
    )

    def open_claims(case_id: str) -> list[HypothesisCode]:
        return [h for h in coded[case_id].hypotheses if h.present == "yes" and (case_id, h.id) not in done]

    wanted = set(case_ids) if case_ids else None
    todo = [cid for cid in coded if (wanted is None or cid in wanted) and open_claims(cid)]
    if limit is not None:
        todo = todo[:limit]
    pending = runner.pending_case_ids() if mode == "batch" and not dry_run else set()
    decisions = load_decisions(settings.work("decisions.jsonl"), set(todo) | pending)

    requests: list[LLMRequest] = []
    system = verifier_system_prompt(codebook)
    custom = make_custom_ids(todo)
    n_uncited = 0
    for cid in todo:
        decision = decisions.get(cid)
        if decision is None:
            if not dry_run:
                runner.log_error(cid, "decision_not_found", "not in decisions.jsonl")
            continue
        testable = []
        for h in open_claims(cid):
            if cited_positions(h, decision):
                testable.append(h)
            elif not dry_run:
                n_uncited += 1
                append_jsonl(
                    out_path,
                    grade_verdict(cid, h.id, "insufficient_citation", [], [], "No citation to an existing paragraph.", "rule"),
                )
                done.add((cid, h.id))
        if testable:
            requests.append(
                LLMRequest(
                    custom_id=custom[cid],
                    case_id=cid,
                    params=build_params(
                        model=settings.llm.verifier_model,
                        system=system,
                        user=verification_user_message(decision, testable, codebook),
                        max_tokens=settings.llm.max_tokens,
                        effort=settings.llm.effort_verify,
                        output_model=VerificationOutput,
                        cache_ttl=cache_ttl_for(mode),
                    ),
                )
            )

    if dry_run:
        preview = write_preview(settings.work("verification_requests_preview.jsonl"), requests)
        est = estimate_cost(requests, output_tokens_per_request=OUTPUT_TOKENS_ASSUMED, batch=mode == "batch")
        log(f"[{STAGE}] dry run, no API call. {len(requests)} decisions with claims to verify.")
        log(f"[{STAGE}] {est.summary()}")
        return preview

    def handle(res: LLMResult) -> str | None:
        decision = decisions.get(res.case_id)
        if decision is None or res.case_id not in coded:
            return "decision or coding not loaded for this result"
        expected = {h.id: cited_positions(h, decision) for h in open_claims(res.case_id)}
        expected = {k: v for k, v in expected.items() if v}
        try:
            output = VerificationOutput.model_validate_json(res.text)
        except ValidationError as exc:
            return f"output does not match the schema: {exc}"
        seen: set[str] = set()
        for v in output.verdicts:
            claim_id = v.claim_id.strip().upper()
            if claim_id not in expected or claim_id in seen:
                continue
            seen.add(claim_id)
            append_jsonl(
                out_path,
                grade_verdict(
                    res.case_id, claim_id, v.verdict, v.supporting_pos, expected[claim_id], v.reason,
                    res.model or settings.llm.verifier_model,
                ),
            )
            done.add((res.case_id, claim_id))
        missing = sorted(set(expected) - seen)
        return f"no verdict returned for {missing}; retried on the next run" if missing else None

    log(f"[{STAGE}] {len(requests)} decisions to verify ({mode}); {n_uncited} uncited claims graded without a call")
    runner.run(requests, handle, mode)
    return out_path
