"""Thin layer over the Anthropic SDK shared by the coding, verification and
outcome stages.

Request shape (Claude Opus 5.5, see the bundled claude-api docs):
- ``system`` is one text block (stage prompt + codebook) with ``cache_control``,
  so thousands of requests reuse the same cached prefix. The per-decision text
  goes in the user turn, after the breakpoint.
- ``output_config`` carries ``effort`` (always explicit: the model's default is
  medium) and ``format`` (structured outputs, a JSON schema made from a
  Pydantic model with the SDK's ``transform_schema``). The raw format is used
  instead of ``messages.parse`` because Message Batches need it anyway and
  because ``parse`` validates inside the SDK, before ``stop_reason`` can be
  checked; here the stage validates the text itself after the checks.
- No ``thinking`` field (Opus 5.5 thinking is always on; ``disabled`` is a 400),
  no assistant prefill (a 400), no forced ``tool_choice`` (a 400).

Transports:
- direct (``async`` and ``sync`` modes): streamed (``messages.stream`` +
  ``get_final_message``), so a large ``max_tokens`` never trips the SDK's
  10-minute non-streaming guard. When ``settings.llm.server_side_fallback`` is
  on, these calls go through ``client.beta.messages`` with
  ``fallbacks="default"`` and beta ``server-side-fallback-2026-07-01``, so a
  policy decline is retried server-side on Anthropic's recommended model.
- ``batch``: Message Batches (half price). The docs say ``fallbacks`` is
  rejected on the Batches API, so batch requests never carry it; a refused
  batch item is logged and can be retried with ``--mode async``.

Every failure (refusal, max_tokens, empty output, validation error, API error,
errored/expired batch item) is appended to ``<stage>_errors.jsonl``; token
usage per response goes to ``<stage>_usage.jsonl``.
"""

from __future__ import annotations

import asyncio
import json
import math
import re
import sys
import time
import traceback
from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import anthropic
from pydantic import BaseModel

from ..config import Settings
from ..io_utils import append_jsonl

FALLBACK_BETA = "server-side-fallback-2026-07-01"  # header for the scalar fallbacks="default" form
MAX_RETRIES = 5  # SDK retries 408/409/429/5xx and connection errors with backoff
MAX_BATCH_REQUESTS = 100_000  # documented per-batch limit
MAX_BATCH_BYTES = 200 * 1024 * 1024  # documented limit is 256 MB; leave room for framing
MODES = ("batch", "async", "sync")

# USD per million tokens, from the bundled docs (Claude Opus 5.5). Batch is 50%
# off every token, cache reads and writes included.
PRICES_PER_MTOK: dict[str, dict[str, float]] = {
    "claude-opus-5-5": {
        "input": 4.00,
        "output": 20.00,
        "cache_write_5m": 5.00,
        "cache_write_1h": 8.00,
        "cache_read": 0.20,
    },
}
BATCH_DISCOUNT = 0.5


def log(*args: Any) -> None:
    """Progress to stderr (stdout carries the CLI's result)."""
    print(*args, file=sys.stderr, flush=True)


@dataclass
class LLMRequest:
    custom_id: str
    case_id: str
    params: dict[str, Any]


@dataclass
class LLMResult:
    custom_id: str
    case_id: str
    ok: bool
    text: str = ""
    model: str = ""
    stop_reason: str | None = None
    error: str | None = None
    detail: str = ""
    usage: dict[str, int] = field(default_factory=dict)


# A handler parses one successful result and writes the stage output. It
# returns None on success or a short problem description, which is logged as a
# validation error (the case stays undone, so a re-run retries it).
Handler = Callable[[LLMResult], "str | None"]


class FatalAPIError(RuntimeError):
    """Errors that would fail every remaining request (bad key, bad model id, billing)."""


# --------------------------------------------------------------------------
# Request building
# --------------------------------------------------------------------------


def structured_format(output_model: type[BaseModel]) -> dict[str, Any]:
    """``output_config.format`` for replies that match ``output_model``."""
    return {"type": "json_schema", "schema": anthropic.transform_schema(output_model)}


def build_params(
    *,
    model: str,
    system: str,
    user: str,
    max_tokens: int,
    effort: str,
    output_model: type[BaseModel],
    cache_ttl: str | None = None,
) -> dict[str, Any]:
    """Messages API params for one single-turn structured request."""
    cache_control: dict[str, str] = {"type": "ephemeral"}
    if cache_ttl:
        cache_control["ttl"] = cache_ttl
    return {
        "model": model,
        "max_tokens": max_tokens,
        "system": [{"type": "text", "text": system, "cache_control": cache_control}],
        "messages": [{"role": "user", "content": user}],
        "output_config": {"effort": effort, "format": structured_format(output_model)},
    }


def cache_ttl_for(mode: str) -> str | None:
    """1-hour cache in batches (processing order and timing are opaque), default 5 minutes otherwise."""
    return "1h" if mode == "batch" else None


def make_custom_ids(case_ids: Iterable[str]) -> dict[str, str]:
    """case_id -> unique batch-safe custom_id ([A-Za-z0-9_-], at most 64 chars)."""
    out: dict[str, str] = {}
    used: set[str] = set()
    for case_id in case_ids:
        base = re.sub(r"[^A-Za-z0-9_-]", "_", case_id)[:56] or "case"
        cid, n = base, 1
        while cid in used:
            n += 1
            cid = f"{base}-{n}"
        used.add(cid)
        out[case_id] = cid
    return out


# --------------------------------------------------------------------------
# Cost estimate (local, no API call)
# --------------------------------------------------------------------------


@dataclass
class CostEstimate:
    model: str
    n_requests: int
    input_tokens: int
    cache_write_tokens: int
    cache_read_tokens: int
    output_tokens: int
    batch: bool
    usd: float
    note: str = ""

    def summary(self) -> str:
        return (
            f"ESTIMATE ({'batch, 50% off' if self.batch else 'standard'}): {self.n_requests} requests to "
            f"{self.model}, ~{self.input_tokens:,} uncached input + {self.cache_write_tokens:,} cache-write + "
            f"{self.cache_read_tokens:,} cache-read input tokens, ~{self.output_tokens:,} output tokens "
            f"(assumed) -> about ${self.usd:,.2f}. {self.note}"
        )


def estimate_tokens(text: str) -> int:
    """Rough token count (characters / 4); an estimate, not a tokenizer."""
    return math.ceil(len(text) / 4)


def price_usd(
    model: str,
    *,
    input_tokens: int = 0,
    output_tokens: int = 0,
    cache_write_tokens: int = 0,
    cache_read_tokens: int = 0,
    batch: bool = False,
    cache_ttl: str | None = None,
) -> float:
    prices = PRICES_PER_MTOK.get(model, PRICES_PER_MTOK["claude-opus-5-5"])
    write = prices["cache_write_1h" if cache_ttl == "1h" else "cache_write_5m"]
    usd = (
        input_tokens * prices["input"]
        + output_tokens * prices["output"]
        + cache_write_tokens * write
        + cache_read_tokens * prices["cache_read"]
    ) / 1_000_000
    return usd * (BATCH_DISCOUNT if batch else 1.0)


def estimate_cost(
    requests: Sequence[LLMRequest], *, output_tokens_per_request: int, batch: bool
) -> CostEstimate:
    """Estimate from characters/4 and list prices, assuming the first request
    with each system prompt writes the cache and every later one reads it.
    Output tokens include thinking and are an assumption per stage."""
    if not requests:
        return CostEstimate("-", 0, 0, 0, 0, 0, batch, 0.0, "nothing to do")
    model = requests[0].params["model"]
    seen_systems: set[str] = set()
    inp = write = read = 0
    usd = 0.0
    for req in requests:
        p = req.params
        system = "".join(b["text"] for b in p["system"])
        sys_tokens = estimate_tokens(system)
        ttl = p["system"][-1].get("cache_control", {}).get("ttl")
        user = "".join(m["content"] if isinstance(m["content"], str) else json.dumps(m["content"]) for m in p["messages"])
        schema = json.dumps(p.get("output_config", {}).get("format", {}), sort_keys=True)
        uncached = estimate_tokens(user) + estimate_tokens(schema)
        if system in seen_systems:
            w, r = 0, sys_tokens
        else:
            seen_systems.add(system)
            w, r = sys_tokens, 0
        inp, write, read = inp + uncached, write + w, read + r
        usd += price_usd(
            p["model"],
            input_tokens=uncached,
            output_tokens=output_tokens_per_request,
            cache_write_tokens=w,
            cache_read_tokens=r,
            batch=batch,
            cache_ttl=ttl,
        )
    note = "Estimate only: chars/4 token counts, list prices, cache hits assumed"
    if model not in PRICES_PER_MTOK:
        note += f"; no price for {model}, Opus 5.5 prices used"
    return CostEstimate(model, len(requests), inp, write, read, output_tokens_per_request * len(requests), batch, usd, note + ".")


def write_preview(path: Path, requests: Sequence[LLMRequest], n: int = 3) -> Path:
    """The first ``n`` would-be requests, for inspection before spending money."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for req in requests[:n]:
            fh.write(json.dumps({"custom_id": req.custom_id, "case_id": req.case_id, "params": req.params}) + "\n")
    return path


# --------------------------------------------------------------------------
# Runner
# --------------------------------------------------------------------------


def classify_api_error(exc: Exception) -> tuple[str, bool]:
    """(reason, fatal) for an SDK exception; most specific class first."""
    if isinstance(exc, (anthropic.AuthenticationError, anthropic.PermissionDeniedError)):
        return "auth_error", True
    if isinstance(exc, anthropic.NotFoundError):
        return "not_found", True  # e.g. a bad model id in settings
    if isinstance(exc, anthropic.RequestTooLargeError):
        return "request_too_large", False
    if isinstance(exc, anthropic.BadRequestError):
        return "bad_request", False
    if isinstance(exc, anthropic.RateLimitError):
        return "rate_limited", False  # after the SDK's own retries
    if isinstance(exc, anthropic.APIStatusError):
        return f"api_error_{exc.status_code}", exc.status_code == 402  # 402: billing
    if isinstance(exc, anthropic.APITimeoutError):
        return "timeout", False
    if isinstance(exc, anthropic.APIConnectionError):
        return "connection_error", False
    return type(exc).__name__, False


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _usage_dict(usage: Any) -> dict[str, int]:
    keys = ("input_tokens", "output_tokens", "cache_creation_input_tokens", "cache_read_input_tokens")
    return {k: int(getattr(usage, k, 0) or 0) for k in keys}


class LLMRunner:
    """Sends a stage's requests by batch, async or sync, and routes results."""

    def __init__(
        self,
        settings: Settings,
        stage: str,
        *,
        client: Any = None,
        async_client: Any = None,
        poll_interval: float = 60.0,
        max_wait: float | None = None,
    ) -> None:
        self.settings = settings
        self.stage = stage
        self._client = client
        self._async_client = async_client
        self.poll_interval = poll_interval
        self.max_wait = max_wait
        self.fallbacks = settings.llm.server_side_fallback
        self.errors_path = settings.work(f"{stage}_errors.jsonl")
        self.usage_path = settings.work(f"{stage}_usage.jsonl")
        self.state_path = settings.work("batches", f"{stage}.json")
        self.counts: Counter[str] = Counter()

    # ------------------------------------------------------------- plumbing

    def resolve_mode(self, mode: str) -> str:
        if mode == "auto":
            return "batch" if self.settings.llm.use_batches else "async"
        if mode not in MODES:
            raise ValueError(f"mode must be auto or one of {MODES}, not {mode!r}")
        return mode

    def client(self) -> Any:
        if self._client is None:
            self._client = anthropic.Anthropic(max_retries=MAX_RETRIES)
        return self._client

    def log_error(self, case_id: str, reason: str, detail: str = "", **extra: Any) -> None:
        rec = {"time": _now(), "stage": self.stage, "case_id": case_id, "reason": reason, "detail": detail}
        rec.update(extra)
        append_jsonl(self.errors_path, rec)
        self.counts[f"error:{reason}"] += 1

    def _log_usage(self, res: LLMResult, *, batch: bool, fallback_used: bool) -> None:
        rec = {
            "time": _now(),
            "stage": self.stage,
            "case_id": res.case_id,
            "custom_id": res.custom_id,
            "model": res.model,
            "batch": batch,
            "stop_reason": res.stop_reason,
            "fallback_used": fallback_used,
            **res.usage,
        }
        append_jsonl(self.usage_path, rec)

    def _failure(self, custom_id: str, case_id: str, reason: str, detail: str = "", model: str = "") -> LLMResult:
        return LLMResult(custom_id=custom_id, case_id=case_id, ok=False, error=reason, detail=detail, model=model)

    def result_from_message(
        self, custom_id: str, case_id: str, msg: Any, *, batch: bool, requested_model: str, max_tokens: int
    ) -> LLMResult:
        """Check stop_reason before reading content; log usage either way."""
        model = getattr(msg, "model", None) or requested_model
        iterations = getattr(msg.usage, "iterations", None) or []
        fallback_used = any(getattr(it, "type", None) == "fallback_message" for it in iterations)
        res = LLMResult(
            custom_id=custom_id,
            case_id=case_id,
            ok=False,
            model=model,
            stop_reason=msg.stop_reason,
            usage=_usage_dict(msg.usage),
        )
        self._log_usage(res, batch=batch, fallback_used=fallback_used)
        if msg.stop_reason == "refusal":
            details = getattr(msg, "stop_details", None)
            category = getattr(details, "category", None)
            explanation = getattr(details, "explanation", None) or ""
            res.error, res.detail = "refusal", f"category={category}; {explanation}".strip("; ")
            return res
        if msg.stop_reason == "max_tokens":
            res.error, res.detail = "max_tokens", f"max_tokens={max_tokens} reached (thinking counts toward it)"
            return res
        # A fallback block marks a model switch; text blocks on both sides form
        # the answer (a mid-stream switch continues the partial text).
        text = "".join(b.text for b in msg.content if getattr(b, "type", None) == "text")
        if not text.strip():
            res.error, res.detail = "no_text", f"stop_reason={msg.stop_reason}"
            return res
        res.ok, res.text = True, text
        return res

    def dispatch(self, res: LLMResult, handler: Handler) -> None:
        if not res.ok:
            self.log_error(res.case_id, res.error or "unknown", res.detail, custom_id=res.custom_id, model=res.model)
            return
        try:
            problem = handler(res)
        except Exception as exc:  # a bug in a handler must not lose the rest of a batch
            problem = f"handler_error: {exc!r}\n{traceback.format_exc(limit=3)}"
        if problem:
            reason = "handler_error" if problem.startswith("handler_error") else "validation_error"
            self.log_error(
                res.case_id, reason, problem[:4000], custom_id=res.custom_id, model=res.model, raw_text=res.text[:4000]
            )
        else:
            self.counts["ok"] += 1

    def run(self, requests: Sequence[LLMRequest], handler: Handler, mode: str) -> Counter[str]:
        mode = self.resolve_mode(mode)
        for req in requests:
            if "fallbacks" in req.params or "thinking" in req.params:
                raise ValueError("params must not carry fallbacks or thinking; the runner adds fallbacks itself")
        if mode == "batch":
            self.run_batch(requests, handler)
        elif mode == "async":
            asyncio.run(self.run_async(requests, handler))
        else:
            for req in requests:
                self.dispatch(self.call(req), handler)
        log(f"[{self.stage}] done: " + ", ".join(f"{k}={v}" for k, v in sorted(self.counts.items())))
        return self.counts

    # --------------------------------------------------------------- direct

    def _fallback_kwargs(self) -> dict[str, Any]:
        return {"betas": [FALLBACK_BETA], "fallbacks": "default"}

    def _drop_fallbacks(self, exc: Exception) -> bool:
        """If the API rejects the fallbacks parameter, carry on without it."""
        if self.fallbacks and isinstance(exc, anthropic.BadRequestError) and "fallback" in str(exc).lower():
            self.fallbacks = False
            log(f"[{self.stage}] server-side fallback rejected ({exc}); continuing without it")
            return True
        return False

    def _api_failure(self, req: LLMRequest, exc: Exception) -> LLMResult:
        reason, fatal = classify_api_error(exc)
        detail = f"{type(exc).__name__}: {exc}"
        request_id = getattr(exc, "request_id", None)
        if fatal:
            self.log_error(req.case_id, reason, detail, custom_id=req.custom_id, request_id=request_id)
            raise FatalAPIError(f"{reason}: {detail}") from exc
        return self._failure(req.custom_id, req.case_id, reason, detail, req.params["model"])

    def call(self, req: LLMRequest) -> LLMResult:
        """One streamed request with SDK retries; never raises except on fatal errors."""
        client = self.client()
        while True:
            try:
                if self.fallbacks:
                    with client.beta.messages.stream(**req.params, **self._fallback_kwargs()) as stream:
                        msg = stream.get_final_message()
                else:
                    with client.messages.stream(**req.params) as stream:
                        msg = stream.get_final_message()
            except anthropic.APIError as exc:
                if self._drop_fallbacks(exc):
                    continue
                return self._api_failure(req, exc)
            return self.result_from_message(
                req.custom_id,
                req.case_id,
                msg,
                batch=False,
                requested_model=req.params["model"],
                max_tokens=req.params["max_tokens"],
            )

    async def call_async(self, client: Any, req: LLMRequest) -> LLMResult:
        while True:
            try:
                if self.fallbacks:
                    async with client.beta.messages.stream(**req.params, **self._fallback_kwargs()) as stream:
                        msg = await stream.get_final_message()
                else:
                    async with client.messages.stream(**req.params) as stream:
                        msg = await stream.get_final_message()
            except anthropic.APIError as exc:
                if self._drop_fallbacks(exc):
                    continue
                return self._api_failure(req, exc)
            return self.result_from_message(
                req.custom_id,
                req.case_id,
                msg,
                batch=False,
                requested_model=req.params["model"],
                max_tokens=req.params["max_tokens"],
            )

    async def run_async(self, requests: Sequence[LLMRequest], handler: Handler) -> None:
        owned = self._async_client is None
        client = self._async_client or anthropic.AsyncAnthropic(max_retries=MAX_RETRIES)
        sem = asyncio.Semaphore(max(1, self.settings.llm.concurrency))
        aborted: list[BaseException] = []
        done = 0

        async def one(req: LLMRequest) -> None:
            nonlocal done
            if aborted:
                return
            async with sem:
                if aborted:
                    return
                try:
                    res = await self.call_async(client, req)
                except FatalAPIError as exc:
                    aborted.append(exc)
                    raise
            self.dispatch(res, handler)
            done += 1
            if done % 25 == 0 or done == len(requests):
                log(f"[{self.stage}] {done}/{len(requests)}")

        try:
            if requests:
                # A cache entry is readable only once a response has started, so
                # the first request goes alone and the rest read its cache.
                await one(requests[0])
            await asyncio.gather(*(one(r) for r in requests[1:]))
        finally:
            if owned:
                await client.close()

    # ---------------------------------------------------------------- batch

    def load_state(self) -> dict[str, Any]:
        if self.state_path.exists():
            return json.loads(self.state_path.read_text(encoding="utf-8"))
        return {"stage": self.stage, "batches": []}

    def save_state(self, state: dict[str, Any]) -> None:
        tmp = self.state_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(state, indent=1, sort_keys=True), encoding="utf-8")
        tmp.replace(self.state_path)

    def pending_case_ids(self) -> set[str]:
        """Case ids in submitted batches whose results are not yet collected."""
        state = self.load_state()
        return {cid for b in state["batches"] if b["status"] != "collected" for cid in b["custom_ids"].values()}

    @staticmethod
    def chunk_requests(requests: Sequence[LLMRequest]) -> list[list[LLMRequest]]:
        """Split into batches within the documented count and size limits."""
        chunks: list[list[LLMRequest]] = []
        current: list[LLMRequest] = []
        size = 0
        for req in requests:
            n = len(json.dumps({"custom_id": req.custom_id, "params": req.params}).encode("utf-8"))
            if current and (len(current) >= MAX_BATCH_REQUESTS or size + n > MAX_BATCH_BYTES):
                chunks.append(current)
                current, size = [], 0
            current.append(req)
            size += n
        if current:
            chunks.append(current)
        return chunks

    def run_batch(self, requests: Sequence[LLMRequest], handler: Handler) -> None:
        """Submit what is not already in flight, then poll and collect.

        State (batch ids and custom_id -> case_id) is saved after every
        submission and collection, so an interrupted run resumes: re-running
        the same command collects earlier batches instead of resubmitting.
        """
        client = self.client()
        state = self.load_state()
        in_flight = {cid for b in state["batches"] if b["status"] != "collected" for cid in b["custom_ids"].values()}
        todo = [r for r in requests if r.case_id not in in_flight]
        for chunk in self.chunk_requests(todo):
            try:
                batch = client.messages.batches.create(
                    requests=[{"custom_id": r.custom_id, "params": r.params} for r in chunk]
                )
            except anthropic.APIError as exc:
                reason, fatal = classify_api_error(exc)
                for r in chunk:
                    self.log_error(r.case_id, f"batch_submit_{reason}", str(exc), custom_id=r.custom_id)
                if fatal:
                    raise FatalAPIError(f"{reason}: {exc}") from exc
                continue
            state["batches"].append(
                {
                    "batch_id": batch.id,
                    "status": "submitted",
                    "submitted_at": _now(),
                    "model": chunk[0].params["model"],
                    "max_tokens": chunk[0].params["max_tokens"],
                    "custom_ids": {r.custom_id: r.case_id for r in chunk},
                }
            )
            self.save_state(state)
            log(f"[{self.stage}] submitted batch {batch.id} ({len(chunk)} requests)")
        self.poll_and_collect(state, handler)

    def poll_and_collect(self, state: dict[str, Any], handler: Handler) -> None:
        client = self.client()
        start = time.monotonic()
        while True:
            open_batches = [b for b in state["batches"] if b["status"] != "collected"]
            if not open_batches:
                return
            for b in open_batches:
                if b["status"] == "submitted":
                    try:
                        info = client.messages.batches.retrieve(b["batch_id"])
                    except anthropic.APIError as exc:
                        reason, fatal = classify_api_error(exc)
                        if fatal:
                            raise FatalAPIError(f"{reason}: {exc}") from exc
                        log(f"[{self.stage}] could not poll {b['batch_id']} ({reason}); will retry")
                        continue
                    if info.processing_status != "ended":
                        counts = getattr(info, "request_counts", None)
                        log(f"[{self.stage}] {b['batch_id']}: {info.processing_status} {counts or ''}")
                        continue
                    b["status"] = "ended"
                    self.save_state(state)
                if b["status"] == "ended":
                    self.collect_batch(b, handler)
                    b["status"] = "collected"
                    b["collected_at"] = _now()
                    self.save_state(state)
            if all(b["status"] == "collected" for b in state["batches"]):
                return
            if self.max_wait is not None and time.monotonic() - start >= self.max_wait:
                log(f"[{self.stage}] batches still processing; re-run the same command later to collect them")
                return
            time.sleep(self.poll_interval)

    def collect_batch(self, b: dict[str, Any], handler: Handler) -> None:
        """Stream results (any order) and route each by custom_id."""
        client = self.client()
        mapping: dict[str, str] = b["custom_ids"]
        seen: set[str] = set()
        for item in client.messages.batches.results(b["batch_id"]):
            case_id = mapping.get(item.custom_id)
            if case_id is None:
                self.log_error("", "unknown_custom_id", f"{item.custom_id} in {b['batch_id']}")
                continue
            seen.add(item.custom_id)
            result = item.result
            if result.type == "succeeded":
                res = self.result_from_message(
                    item.custom_id,
                    case_id,
                    result.message,
                    batch=True,
                    requested_model=b.get("model", ""),
                    max_tokens=b.get("max_tokens", 0),
                )
            elif result.type == "errored":
                err = getattr(getattr(result, "error", None), "error", None)
                detail = f"{getattr(err, 'type', '?')}: {getattr(err, 'message', '')}"
                res = self._failure(item.custom_id, case_id, "batch_errored", detail, b.get("model", ""))
            else:  # canceled / expired: not attempted; a re-run resubmits
                res = self._failure(item.custom_id, case_id, f"batch_{result.type}", "resubmitted on the next run")
            self.dispatch(res, handler)
        for custom_id in sorted(set(mapping) - seen):
            self.log_error(mapping[custom_id], "batch_missing_result", b["batch_id"], custom_id=custom_id)
