"""LLM stages with a fake SDK client: no network, no API key.

Every decision here is synthetic (invented landlords and wording).
"""

from __future__ import annotations

import json
import math
import re
import sys
import types
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import anthropic
import httpx2
import pytest

from hos_pipeline.config import LLM, Paths, Settings
from hos_pipeline.io_utils import read_jsonl, write_jsonl
from hos_pipeline.llm import client as client_mod
from hos_pipeline.llm.client import FALLBACK_BETA, LLMRunner, build_params, estimate_cost, make_custom_ids
from hos_pipeline.llm.codebook import DIMENSION_IDS, EVENT_TYPES, HYPOTHESIS_IDS, load_codebook
from hos_pipeline.llm.coder import (
    CodingOutput,
    build_coding_requests,
    postprocess,
    quote_in_text,
    run_coding,
)
from hos_pipeline.llm.irr import compare, export_irr_sheet, load_human_codes
from hos_pipeline.llm.outcomes import recall_report, regex_determinations, run_outcomes
from hos_pipeline.llm.verifier import grade_verdict, run_verification
from hos_pipeline.models import (
    Citation,
    ClaimVerification,
    CodedDecision,
    Decision,
    DimensionCode,
    HypothesisCode,
    OutcomeClassification,
    Paragraph,
    SampleRecord,
    ScreenResult,
)

ROOT = Path(__file__).resolve().parents[1]
CODEBOOK = ROOT / "codebook" / "codebook.yaml"
MODEL = "claude-opus-5-5"

# --------------------------------------------------------------------------
# Synthetic data
# --------------------------------------------------------------------------

PARAS = [
    ("The complaint", None, "The resident complained about the landlord's response to her reports of damp and mould in the bedroom."),
    ("Background", 5, "The resident reported black mould in the bedroom on 3 March 2023."),
    ("Background", 6, "The resident said the landlord ignored her calls."),
    ("Assessment and findings", 20, "The landlord did not inspect the property until 14 July 2023, more than four months after the report. This was an unreasonable delay."),
    ("Assessment and findings", 21, "The landlord's records do not show what was found at the inspection."),
    ("Assessment and findings", 22, "The landlord's stage 1 response was issued 45 working days after the complaint, well outside its policy timescale."),
    ("Determination", 30, "In accordance with paragraph 52 of the Scheme, there was severe maladministration in the landlord's response to the resident's reports of damp and mould."),
    ("Determination", 31, "In accordance with paragraph 52 of the Scheme, there was maladministration in the landlord's complaint handling."),
    ("Orders", 32, "The landlord must pay the resident £900 within 4 weeks, comprising £700 for distress and inconvenience and £200 for time and trouble."),
]


def make_decision(case_id: str, landlord: str = "Brightwater Homes") -> Decision:
    return Decision(
        case_id=case_id,
        landlord=landlord,
        decision_date=date(2024, 5, 1),
        year=2024,
        paragraphs=[
            Paragraph(pos=i + 1, number=num, section=sec, text=text) for i, (sec, num, text) in enumerate(PARAS)
        ],
    )


def coding_output(case_id: str) -> dict:
    """A valid model reply for the synthetic decision."""
    codebook = load_codebook(CODEBOOK)
    hyps = [
        {"id": h, "present": "no", "citations": [], "rationale": "Not found."} for h in HYPOTHESIS_IDS
    ]
    hyps[0] = {
        "id": "H1",
        "present": "yes",
        "citations": [{"pos": 4, "quote": "This was an unreasonable delay."}],
        "rationale": "The Ombudsman found the inspection was unreasonably delayed.",
    }
    dims = [{"id": d, "present": "no", "value": None, "citations": []} for d in DIMENSION_IDS]
    dims[1] = {"id": "D2", "present": "yes", "value": "900", "citations": [{"pos": 9, "quote": ""}]}
    return {
        "hypotheses": hyps,
        "remediation": dims,
        "admin_flags": [{"id": f, "value": "no"} for f in codebook.admin_flag_ids],
        "events": [
            {"type": "first_report", "date_text": "3 March 2023", "year_if_omitted": None, "precision": "day",
             "earliest": "2023-03-03", "latest": "2023-03-03", "citations": [{"pos": 2, "quote": ""}]},
            {"type": "investigation_started", "date_text": "14 July 2023", "year_if_omitted": None,
             "precision": "day", "earliest": "2023-07-14", "latest": "2023-07-14",
             "citations": [{"pos": 4, "quote": ""}]},
        ],
        "compensation": [
            {"amount_gbp": 900, "kind": "ordered", "purpose": "distress and inconvenience; time and trouble",
             "citations": [{"pos": 9, "quote": ""}]}
        ],
        "determinations": [
            {"element": "the landlord's response to reports of damp and mould", "category": "damp_mould",
             "finding": "severe_maladministration", "citations": [{"pos": 7, "quote": ""}]},
            {"element": "complaint handling", "category": "complaint_handling", "finding": "maladministration",
             "citations": [{"pos": 8, "quote": ""}]},
        ],
        "notes": f"coded {case_id}",
    }


def settings_for(tmp_path: Path, **llm) -> Settings:
    base = {"server_side_fallback": False, "concurrency": 3}
    base.update(llm)
    return Settings(paths=Paths(work_dir=tmp_path / "work", codebook=CODEBOOK), llm=LLM(**base))


def write_inputs(s: Settings, case_ids: list[str]) -> None:
    write_jsonl(
        s.work("sample.jsonl"),
        [SampleRecord(case_id=c, stratum="2024|ha", population_n=10, sample_n=4, weight=2.5, draw_order=i)
         for i, c in enumerate(case_ids)],
    )
    write_jsonl(s.work("decisions.jsonl"), [make_decision(c, landlord=f"Landlord {c}") for c in case_ids])


# --------------------------------------------------------------------------
# Fake SDK client
# --------------------------------------------------------------------------


def message(text: str | None = None, stop_reason: str = "end_turn", **kw) -> SimpleNamespace:
    content = [SimpleNamespace(type="thinking", thinking="")]
    if text is not None:
        content.append(SimpleNamespace(type="text", text=text))
    usage = SimpleNamespace(
        input_tokens=120, output_tokens=60, cache_creation_input_tokens=0, cache_read_input_tokens=7000, iterations=None
    )
    return SimpleNamespace(
        content=content, stop_reason=stop_reason, model=kw.get("model", MODEL), usage=usage,
        stop_details=kw.get("stop_details"),
    )


def case_of(params: dict) -> str:
    return re.search(r"Case ID: (\S+)", params["messages"][0]["content"])[1]


class _Stream:
    def __init__(self, msg):
        self.msg = msg

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def get_final_message(self):
        return self.msg


class _AsyncStream(_Stream):
    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def get_final_message(self):  # type: ignore[override]
        return self.msg


class FakeMessages:
    def __init__(self, responder, calls: list, kind: str, is_async: bool):
        self.responder, self.calls, self.kind, self.is_async = responder, calls, kind, is_async

    def stream(self, **kwargs):
        self.calls.append((self.kind, kwargs))
        msg = self.responder(kwargs)
        return _AsyncStream(msg) if self.is_async else _Stream(msg)


class FakeServer:
    """Batch state that survives across fake clients, like the real API."""

    def __init__(self, responder, result_type=None):
        self.responder = responder
        self.result_type = result_type or (lambda case_id: "succeeded")
        self.batches: dict[str, list[dict]] = {}
        self.status = "ended"


class FakeBatches:
    def __init__(self, server: FakeServer):
        self.server = server

    def create(self, requests):
        bid = f"msgbatch_{len(self.server.batches) + 1:03d}"
        self.server.batches[bid] = [dict(r) for r in requests]
        return SimpleNamespace(id=bid, processing_status="in_progress")

    def retrieve(self, bid):
        return SimpleNamespace(id=bid, processing_status=self.server.status, request_counts=None)

    def results(self, bid):
        items = []
        for r in self.server.batches[bid]:
            kind = self.server.result_type(case_of(r["params"]))
            if kind == "succeeded":
                result = SimpleNamespace(type="succeeded", message=self.server.responder(r["params"]))
            elif kind == "errored":
                err = SimpleNamespace(type="invalid_request_error", message="synthetic failure")
                result = SimpleNamespace(type="errored", error=SimpleNamespace(type="error", error=err))
            else:
                result = SimpleNamespace(type=kind)
            items.append(SimpleNamespace(custom_id=r["custom_id"], result=result))
        return list(reversed(items))  # results come back in any order


class FakeClient:
    def __init__(self, responder, server: FakeServer | None = None, is_async: bool = False):
        self.calls: list = []
        self.messages = FakeMessages(responder, self.calls, "messages", is_async)
        self.beta = SimpleNamespace(messages=FakeMessages(responder, self.calls, "beta", is_async))
        self.server = server or FakeServer(responder)
        self.messages.batches = FakeBatches(self.server)


def coding_responder(params: dict) -> SimpleNamespace:
    return message(json.dumps(coding_output(case_of(params))))


@pytest.fixture(autouse=True)
def no_real_client(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("a real Anthropic client was constructed")

    monkeypatch.setattr(anthropic, "Anthropic", boom)
    monkeypatch.setattr(anthropic, "AsyncAnthropic", boom)


# --------------------------------------------------------------------------
# Codebook
# --------------------------------------------------------------------------


def test_codebook_loads_and_renders_deterministically():
    cb = load_codebook(CODEBOOK)
    assert cb.version == "draft-0.1"
    assert cb.hypothesis_ids == list(HYPOTHESIS_IDS)
    assert cb.dimension_ids == list(DIMENSION_IDS)
    assert cb.event_type_ids == list(EVENT_TYPES)
    assert "landlord_failed_to_provide_records" in cb.admin_flag_ids
    md = cb.render_markdown()
    assert md == load_codebook(CODEBOOK).render_markdown() == cb.render_markdown()
    assert "### H12:" in md and "### D13:" in md and "### investigation_concluded:" in md
    assert not re.search(r"\d{4}-\d{2}-\d{2}T|\d{2}:\d{2}:\d{2}", md)
    sub = cb.render_markdown(["hypotheses", "rules"])
    assert sub.index("General coding rules") < sub.index("Failure hypotheses") and "### D1:" not in sub


def test_codebook_validation(tmp_path):
    import yaml

    data = yaml.safe_load(CODEBOOK.read_text())
    bad = dict(data, hypotheses=[h for h in data["hypotheses"] if h["id"] != "H5"])
    (tmp_path / "a.yaml").write_text(yaml.safe_dump(bad))
    with pytest.raises(ValueError, match="missing.*H5"):
        load_codebook(tmp_path / "a.yaml")
    dup = dict(data, dimensions=data["dimensions"] + [data["dimensions"][0]])
    (tmp_path / "b.yaml").write_text(yaml.safe_dump(dup))
    with pytest.raises(ValueError, match="duplicate"):
        load_codebook(tmp_path / "b.yaml")


# --------------------------------------------------------------------------
# Request params
# --------------------------------------------------------------------------


def test_request_params(tmp_path):
    s = settings_for(tmp_path, model="claude-opus-5-5", effort_coding="high", max_tokens=12000)
    cb = load_codebook(CODEBOOK)
    reqs = build_coding_requests(s, cb, [make_decision("CASE-1"), make_decision("CASE-2")], mode="async")
    p = reqs[0].params
    assert p["model"] == "claude-opus-5-5" and p["max_tokens"] == 12000
    assert p["output_config"]["effort"] == "high"
    assert p["output_config"]["format"]["type"] == "json_schema"
    assert p["output_config"]["format"]["schema"]["additionalProperties"] is False
    assert p["system"][-1]["cache_control"] == {"type": "ephemeral"}
    assert "### H1:" in p["system"][0]["text"]
    assert "thinking" not in p and "fallbacks" not in p and "tool_choice" not in p
    assert [m["role"] for m in p["messages"]] == ["user"]  # no assistant prefill
    assert "[p4] (para 20) The landlord did not inspect" in p["messages"][0]["content"]
    # identical cached prefix across requests
    assert reqs[0].params["system"] == reqs[1].params["system"]
    batch = build_coding_requests(s, cb, [make_decision("CASE-1")], mode="batch")[0].params
    assert batch["system"][-1]["cache_control"] == {"type": "ephemeral", "ttl": "1h"}


@pytest.mark.parametrize("fallback", [True, False])
def test_fallbacks_only_when_enabled(tmp_path, fallback):
    s = settings_for(tmp_path, server_side_fallback=fallback, use_batches=False)
    write_inputs(s, ["A1", "A2"])
    fake = FakeClient(coding_responder, is_async=True)
    run_coding(s, async_client=fake)
    assert len(fake.calls) == 2
    for kind, kwargs in fake.calls:
        if fallback:
            assert kind == "beta"
            assert kwargs["fallbacks"] == "default" and kwargs["betas"] == [FALLBACK_BETA]
        else:
            assert kind == "messages" and "fallbacks" not in kwargs and "betas" not in kwargs
        assert kwargs["output_config"]["effort"] == s.llm.effort_coding
    assert {c.case_id for c in read_jsonl(s.work("coded.jsonl"), CodedDecision)} == {"A1", "A2"}


def test_batch_requests_never_carry_fallbacks(tmp_path):
    s = settings_for(tmp_path, server_side_fallback=True)
    write_inputs(s, ["B1"])
    fake = FakeClient(coding_responder)
    run_coding(s, mode="batch", client=fake, poll_interval=0)
    (reqs,) = fake.server.batches.values()
    assert all("fallbacks" not in r["params"] and "betas" not in r["params"] for r in reqs)


def test_rejected_fallbacks_are_dropped(tmp_path):
    s = settings_for(tmp_path, server_side_fallback=True)
    write_inputs(s, ["F1"])
    request = httpx2.Request("POST", "https://example.invalid/v1/messages")

    def responder(params):
        if "fallbacks" in params:
            raise anthropic.BadRequestError(
                "fallbacks: not supported", response=httpx2.Response(400, request=request), body=None
            )
        return coding_responder(params)

    fake = FakeClient(responder)
    run_coding(s, mode="sync", client=fake)
    assert [k for k, _ in fake.calls] == ["beta", "messages"]
    assert len(list(read_jsonl(s.work("coded.jsonl")))) == 1


def test_fatal_api_error_stops_the_run(tmp_path):
    s = settings_for(tmp_path)
    write_inputs(s, ["K1", "K2"])
    request = httpx2.Request("POST", "https://example.invalid/v1/messages")

    def responder(params):
        raise anthropic.AuthenticationError("bad key", response=httpx2.Response(401, request=request), body=None)

    with pytest.raises(client_mod.FatalAPIError):
        run_coding(s, mode="sync", client=FakeClient(responder))
    errors = list(read_jsonl(s.work("coding_errors.jsonl")))
    assert errors[0]["reason"] == "auth_error"


# --------------------------------------------------------------------------
# Post-validation
# --------------------------------------------------------------------------


def test_postprocess_fixes_and_notes():
    cb = load_codebook(CODEBOOK)
    d = make_decision("PV-1")
    raw = coding_output("PV-1")
    hyps = {h["id"]: h for h in raw["hypotheses"]}
    hyps["H2"] = {"id": "H2", "present": "yes", "citations": [{"pos": 99, "quote": ""}], "rationale": "x"}
    hyps["H3"] = {"id": "H3", "present": "yes", "citations": [{"pos": 4, "quote": "the landlord blamed the resident"}],
                  "rationale": "x"}
    hyps["H4"] = {"id": "H4", "present": "yes", "citations": [], "rationale": "x"}
    hyps["H6"] = {"id": "H6", "present": "yes",
                  "citations": [{"pos": 5, "quote": "The landlord’s   records do not show"}], "rationale": "x"}
    del hyps["H12"]
    hyps["H13"] = {"id": "H13", "present": "yes", "citations": [{"pos": 4, "quote": ""}], "rationale": "x"}
    raw["hypotheses"] = list(hyps.values())
    raw["admin_flags"] = raw["admin_flags"][1:] + [{"id": "made_up_flag", "value": "yes"}]
    raw["events"] += [
        {"type": "landlord_aware", "date_text": "late June 2023", "year_if_omitted": None, "precision": "day",
         "earliest": "2023-06-25", "latest": "2023-06-25", "citations": [{"pos": 4, "quote": ""}]},
        {"type": "complaint_made", "date_text": "17 June", "year_if_omitted": 2023, "precision": "day",
         "earliest": "2023-06-17", "latest": "2023-06-17", "citations": [{"pos": 6, "quote": ""}]},
        {"type": "stage1_response", "date_text": "shortly afterwards", "year_if_omitted": None, "precision": "month",
         "earliest": "2023-08-01", "latest": "2023-08-31", "citations": [{"pos": 6, "quote": ""}]},
        {"type": "decant_offered", "date_text": "1 May 2023", "year_if_omitted": None, "precision": "day",
         "earliest": "2023-05-01", "latest": "2023-05-01", "citations": [{"pos": 77, "quote": ""}]},
    ]
    out = postprocess(CodingOutput.model_validate(raw), d, cb, coder=MODEL)
    h = {x.id: x for x in out.hypotheses}

    assert [x.id for x in out.hypotheses] == list(HYPOTHESIS_IDS)
    assert h["H1"].present == "yes" and h["H1"].citations[0].quote == "This was an unreasonable delay."
    assert h["H2"].present == "unclear" and h["H2"].citations == []  # only citation pointed nowhere
    assert h["H3"].present == "yes" and h["H3"].citations == [Citation(pos=4, quote="")]  # bad quote dropped
    assert h["H4"].present == "unclear"  # yes without citation
    assert h["H6"].citations[0].quote  # curly quote and spacing normalised
    assert h["H12"].present == "unclear"
    assert set(out.admin_flags) == set(cb.admin_flag_ids)
    assert out.admin_flags[cb.admin_flag_ids[0]] == "unclear"

    ev = {e.type: e for e in out.events}
    assert (ev["landlord_aware"].precision, ev["landlord_aware"].earliest, ev["landlord_aware"].latest) == (
        "month", date(2023, 6, 1), date(2023, 6, 30))
    assert ev["complaint_made"].precision == "day" and ev["complaint_made"].earliest == date(2023, 6, 17)
    assert ev["stage1_response"].precision == "month" and ev["stage1_response"].earliest == date(2023, 8, 1)
    assert "decant_offered" not in ev  # its only citation was invalid

    notes = out.notes
    assert notes.startswith("coded PV-1")
    for fragment in [
        "H2: dropped citation to non-existent [p99]",
        "H2: 'yes' without a valid citation; downgraded to unclear",
        "H3: quote not found in [p4]",
        "H4: 'yes' without a valid citation",
        "H12: missing from model output; set to unclear",
        "dropped unknown id 'H13'",
        "dropped unknown id 'made_up_flag'",
        "bounds recomputed from date_text as month",
        "year 2023 taken from context",
        "kept the model's month bounds",
        "event decant_offered ('1 May 2023'): no valid citation; dropped",
    ]:
        assert fragment in notes, fragment


def test_quote_matching():
    text = "The landlord’s stage 1 response was issued 45 working days after the complaint."
    assert quote_in_text("the landlord's stage 1 response", text)
    assert quote_in_text("“stage 1 response ... 45 working days”", text)
    assert not quote_in_text("stage 2 response", text)
    assert not quote_in_text("   ", text)


# --------------------------------------------------------------------------
# Batches
# --------------------------------------------------------------------------


def test_batch_results_out_of_order_and_resume(tmp_path):
    s = settings_for(tmp_path)
    ids = ["202301001", "202301002", "202301003"]
    write_inputs(s, ids)
    server = FakeServer(coding_responder)
    server.status = "in_progress"

    out = run_coding(s, mode="batch", client=FakeClient(coding_responder, server), poll_interval=0, max_wait=0)
    assert not out.exists() or not list(read_jsonl(out))
    state = json.loads(s.work("batches", "coding.json").read_text())
    assert len(state["batches"]) == 1 and state["batches"][0]["status"] == "submitted"
    assert sorted(state["batches"][0]["custom_ids"].values()) == ids

    # A later run resumes: no resubmission, results collected and matched by custom_id.
    server.status = "ended"
    run_coding(s, mode="batch", client=FakeClient(coding_responder, server), poll_interval=0)
    assert len(server.batches) == 1
    coded = list(read_jsonl(out, CodedDecision))
    assert sorted(c.case_id for c in coded) == ids
    assert all(c.notes.startswith(f"coded {c.case_id}") for c in coded)
    assert json.loads(s.work("batches", "coding.json").read_text())["batches"][0]["status"] == "collected"
    usage = list(read_jsonl(s.work("coding_usage.jsonl")))
    assert len(usage) == 3 and all(u["batch"] and u["cache_read_input_tokens"] == 7000 for u in usage)

    # Nothing left to do: no new batch.
    run_coding(s, mode="batch", client=FakeClient(coding_responder, server), poll_interval=0)
    assert len(server.batches) == 1


def test_batch_errored_and_expired_are_logged_and_resubmitted(tmp_path):
    s = settings_for(tmp_path)
    write_inputs(s, ["E1", "E2", "E3"])
    kinds = {"E1": "succeeded", "E2": "errored", "E3": "expired"}
    server = FakeServer(coding_responder, result_type=kinds.get)
    run_coding(s, mode="batch", client=FakeClient(coding_responder, server), poll_interval=0)
    errors = {e["case_id"]: e["reason"] for e in read_jsonl(s.work("coding_errors.jsonl"))}
    assert errors == {"E2": "batch_errored", "E3": "batch_expired"}
    assert [c["case_id"] for c in read_jsonl(s.work("coded.jsonl"))] == ["E1"]

    server.result_type = lambda case_id: "succeeded"
    run_coding(s, mode="batch", client=FakeClient(coding_responder, server), poll_interval=0)
    assert sorted(server.batches["msgbatch_002"][i]["custom_id"] for i in range(2)) == ["E2", "E3"]
    assert sorted(c["case_id"] for c in read_jsonl(s.work("coded.jsonl"))) == ["E1", "E2", "E3"]


def test_chunking_respects_limits(monkeypatch):
    reqs = [client_mod.LLMRequest(f"c{i}", f"c{i}", {"x": "y" * 100}) for i in range(5)]
    monkeypatch.setattr(client_mod, "MAX_BATCH_REQUESTS", 2)
    assert [len(c) for c in LLMRunner.chunk_requests(reqs)] == [2, 2, 1]
    monkeypatch.setattr(client_mod, "MAX_BATCH_REQUESTS", 100)
    monkeypatch.setattr(client_mod, "MAX_BATCH_BYTES", 300)
    assert [len(c) for c in LLMRunner.chunk_requests(reqs)] == [2, 2, 1]


def test_custom_ids_are_batch_safe():
    ids = make_custom_ids(["2023/01 001", "2023-01_001", "2023/01_001"])
    assert all(re.fullmatch(r"[A-Za-z0-9_-]{1,64}", v) for v in ids.values())
    assert len(set(ids.values())) == 3


# --------------------------------------------------------------------------
# Failures land in errors.jsonl
# --------------------------------------------------------------------------


def test_refusal_max_tokens_and_validation_errors(tmp_path):
    s = settings_for(tmp_path, use_batches=False)
    write_inputs(s, ["R1", "M1", "V1", "OK1"])

    def responder(params):
        cid = case_of(params)
        if cid == "R1":
            return message(None, "refusal", stop_details=SimpleNamespace(category="bio", explanation="declined"))
        if cid == "M1":
            return message('{"hypotheses": [', "max_tokens")
        if cid == "V1":
            return message('{"hypotheses": 1}')
        return coding_responder(params)

    run_coding(s, async_client=FakeClient(responder, is_async=True))
    errors = {e["case_id"]: e for e in read_jsonl(s.work("coding_errors.jsonl"))}
    assert {k: v["reason"] for k, v in errors.items()} == {
        "R1": "refusal", "M1": "max_tokens", "V1": "validation_error"
    }
    assert "category=bio" in errors["R1"]["detail"]
    assert errors["V1"]["raw_text"] == '{"hypotheses": 1}'
    assert [c["case_id"] for c in read_jsonl(s.work("coded.jsonl"))] == ["OK1"]
    assert len(list(read_jsonl(s.work("coding_usage.jsonl")))) == 4

    # Resumable: a re-run only retries the failures.
    fake = FakeClient(coding_responder, is_async=True)
    run_coding(s, async_client=fake)
    assert sorted(case_of(k) for _, k in fake.calls) == ["M1", "R1", "V1"]


def test_dry_run_makes_no_call(tmp_path, capsys):
    s = settings_for(tmp_path)
    write_inputs(s, ["D1", "D2", "D3", "D4"])
    preview = run_coding(s, dry_run=True)
    assert preview == s.work("coding_requests_preview.jsonl")
    lines = list(read_jsonl(preview))
    assert len(lines) == 3 and lines[0]["params"]["model"] == MODEL
    assert "ESTIMATE" in capsys.readouterr().err
    assert not s.work("coded.jsonl").exists()


def test_cost_estimate_uses_cache_and_batch_discount():
    params = build_params(model=MODEL, system="s" * 4000, user="u" * 400, max_tokens=100, effort="high",
                          output_model=CodingOutput)
    reqs = [client_mod.LLMRequest(str(i), str(i), params) for i in range(3)]
    std = estimate_cost(reqs, output_tokens_per_request=1000, batch=False)
    assert std.cache_write_tokens == 1000 and std.cache_read_tokens == 2000
    half = estimate_cost(reqs, output_tokens_per_request=1000, batch=True)
    assert half.usd == pytest.approx(std.usd / 2)


# --------------------------------------------------------------------------
# Verifier
# --------------------------------------------------------------------------


def test_grade_verdict_rules():
    v = grade_verdict("C", "H1", "verified", [4, 9], [4, 5], "Stated.", MODEL)
    assert (v.verdict, v.grade, v.supporting_pos) == ("verified", "A", [4])
    v = grade_verdict("C", "H1", "verified", [9], [4, 5], "Stated.", MODEL)
    assert (v.verdict, v.grade) == ("not_verified", "B") and "downgraded" in v.reason
    v = grade_verdict("C", "H1", "not_verified", [4], [4], "Allegation only.", MODEL)
    assert (v.grade, v.supporting_pos) == ("B", [])
    assert grade_verdict("C", "H1", "insufficient_citation", [], [], "", MODEL).grade == "B"


def test_run_verification(tmp_path):
    s = settings_for(tmp_path, use_batches=False)
    write_inputs(s, ["V1"])
    coded = postprocess(CodingOutput.model_validate(coding_output("V1")), make_decision("V1"), load_codebook(CODEBOOK), MODEL)
    hyps = {h.id: h for h in coded.hypotheses}
    hyps["H6"] = HypothesisCode(id="H6", present="yes", citations=[Citation(pos=5)])
    hyps["H10"] = HypothesisCode(id="H10", present="yes", citations=[Citation(pos=99)])
    coded.hypotheses = list(hyps.values())
    write_jsonl(s.work("coded.jsonl"), [coded])

    def responder(params):
        user = params["messages"][0]["content"]
        assert "[p4] (para 20)" in user and "[p5] (para 21)" in user
        assert "reported black mould" not in user  # uncited paragraph not sent
        assert "H10" not in user  # uncited claim graded without a call
        assert "### H1: Delayed inspection" in params["system"][0]["text"]
        return message(json.dumps({"verdicts": [
            {"claim_id": "H1", "verdict": "verified", "supporting_pos": [4], "reason": "Found unreasonable."},
            {"claim_id": "H6", "verdict": "verified", "supporting_pos": [2], "reason": "Wrong paragraph."},
        ]}))

    fake = FakeClient(responder, is_async=True)
    run_verification(s, async_client=fake)
    assert len(fake.calls) == 1
    assert fake.calls[0][1]["output_config"]["effort"] == s.llm.effort_verify
    got = {v.claim_id: v for v in read_jsonl(s.work("verified.jsonl"), ClaimVerification)}
    assert (got["H1"].verdict, got["H1"].grade) == ("verified", "A")
    assert (got["H6"].verdict, got["H6"].grade) == ("not_verified", "B")
    assert (got["H10"].verdict, got["H10"].grade) == ("insufficient_citation", "B")

    # Resumable: everything done, no new call.
    fake2 = FakeClient(responder, is_async=True)
    run_verification(s, async_client=fake2)
    assert fake2.calls == []


def test_verification_missing_verdict_is_logged(tmp_path):
    s = settings_for(tmp_path, use_batches=False)
    write_inputs(s, ["V2"])
    coded = postprocess(CodingOutput.model_validate(coding_output("V2")), make_decision("V2"), load_codebook(CODEBOOK), MODEL)
    write_jsonl(s.work("coded.jsonl"), [coded])
    run_verification(s, async_client=FakeClient(lambda p: message('{"verdicts": []}'), is_async=True))
    errors = list(read_jsonl(s.work("verification_errors.jsonl")))
    assert errors[0]["reason"] == "validation_error" and "H1" in errors[0]["detail"]
    assert not s.work("verified.jsonl").exists()


# --------------------------------------------------------------------------
# Outcomes
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text, expected",
    [
        ("There was severe maladministration in the landlord's response to reports of damp and mould.",
         [("severe_maladministration", "damp_mould")]),
        ("There was no maladministration in the landlord's handling of the resident's reports of mould.",
         [("no_maladministration", "damp_mould")]),
        ("There was maladministration in the landlord's complaint handling.",
         [("maladministration", "complaint_handling")]),
        ("There was maladministration in its handling of repairs and no maladministration in its handling of the roof leak.",
         [("maladministration", "repairs_other"), ("no_maladministration", "leak_water")]),
        ("There was service failure in the landlord's record keeping.", [("service_failure", "record_keeping")]),
        ("There was severe maladministration in the landlord's handling of the associated complaint.",
         [("severe_maladministration", "complaint_handling")]),
        ("There was maladministration in the landlord's handling of the resident's complaint about condensation.",
         [("maladministration", "damp_mould")]),
        ("The landlord has offered redress which, in the Ombudsman's opinion, resolves the complaint about damp and mould satisfactorily.",
         [("reasonable_redress", "damp_mould")]),
        ("The complaint about the resident's health is outside the Ombudsman's jurisdiction.",
         [("outside_jurisdiction", "other")]),
        ("The landlord's response to reports of anti-social behaviour: no maladministration.",
         [("no_maladministration", "anti_social_behaviour")]),
    ],
)
def test_regex_determinations(text, expected):
    d = Decision(case_id="X", landlord="Example Homes",
                 paragraphs=[Paragraph(pos=1, section="Background", text="There was severe maladministration elsewhere."),
                             Paragraph(pos=2, section="Determination", text=text)])
    got = regex_determinations(d)
    assert [(x.finding, x.category) for x in got] == expected
    assert all(x.citations[0].pos == 2 for x in got)  # Background paragraph ignored


def test_run_outcomes_and_recall_report(tmp_path):
    s = settings_for(tmp_path, use_batches=False, classifier_model="claude-opus-5-5", effort_classify="medium")
    rows = {  # case: (year, keyword_severe_damp, llm_severe_damp)
        "c1": (2023, True, True), "c2": (2023, False, True), "c3": (2023, False, False), "c4": (2023, True, False),
        "c5": (2024, True, True), "c6": (2024, False, False),
    }
    screen = [
        ScreenResult(case_id=c, landlord="L", year=y, mention_counts={"damp": 5}, complaint_definition_hit=True,
                     determination_hit=True, damp_flag=True, rule="test", keyword_severe_damp=kw)
        for c, (y, kw, _) in rows.items()
    ]
    screen.append(ScreenResult(case_id="c7", landlord="L", year=2023, mention_counts={}, complaint_definition_hit=False,
                               determination_hit=False, damp_flag=False, rule="test", keyword_severe_damp=True))
    write_jsonl(s.work("screen.jsonl"), screen)
    write_jsonl(s.work("decisions.jsonl"), [make_decision(c) for c in [*rows, "c7"]])

    def responder(params):
        cid = case_of(params)
        user = params["messages"][0]["content"]
        assert "[p7] (para 30)" in user and "[p9] (para 32)" in user and "[p4]" not in user
        finding = "severe_maladministration" if rows[cid][2] else "maladministration"
        return message(json.dumps({"determinations": [
            {"element": "damp and mould", "category": "damp_mould", "finding": finding, "pos": [7]},
            {"element": "complaint handling", "category": "complaint_handling", "finding": "maladministration", "pos": [8]},
            {"element": "made up", "category": "other", "finding": "other", "pos": [3]},  # not a sent paragraph
        ], "notes": ""}))

    fake = FakeClient(responder, is_async=True)
    run_outcomes(s, async_client=fake)
    assert len(fake.calls) == 6  # c7 is not damp-flagged
    assert fake.calls[0][1]["model"] == "claude-opus-5-5"
    assert fake.calls[0][1]["output_config"]["effort"] == "medium"
    out = {o.case_id: o for o in read_jsonl(s.work("outcomes.jsonl"), OutcomeClassification)}
    assert out["c1"].severe_damp and out["c1"].any_severe and not out["c1"].severe_complaint_handling
    assert not out["c3"].severe_damp and len(out["c3"].determinations) == 2
    dropped = [e for e in read_jsonl(s.work("outcomes_errors.jsonl")) if e["reason"] == "dropped_uncited_determination"]
    assert len(dropped) == 6

    rep = recall_report(s).set_index("year")
    assert s.work("outputs", "severe_recall_by_year.csv").exists()
    r23, r24, tot = rep.loc[2023], rep.loc[2024], rep.loc["all"]
    assert (r23.n, r23.llm_severe, r23.keyword_severe, r23.both, r23.keyword_only, r23.llm_only, r23.neither) == (
        4, 2, 2, 1, 1, 1, 1)
    assert r23.keyword_recall == pytest.approx(0.5) and r23.agreement == pytest.approx(0.5)
    assert (r24.n, r24.keyword_recall, r24.agreement) == (2, 1.0, 1.0)
    assert (tot.n, tot.both) == (6, 2)
    assert tot.keyword_recall == pytest.approx(2 / 3) and tot.keyword_precision == pytest.approx(2 / 3)
    assert tot.agreement == pytest.approx(4 / 6)


# --------------------------------------------------------------------------
# Inter-rater reliability
# --------------------------------------------------------------------------


def _coded(case_id: str, h1: str, h2: str, coder: str) -> CodedDecision:
    return CodedDecision(
        case_id=case_id, codebook_version="draft-0.1", coder=coder,
        hypotheses=[HypothesisCode(id="H1", present=h1), HypothesisCode(id="H2", present=h2)],
        remediation=[DimensionCode(id="D1", present="no")],
        admin_flags={"health_impact_raised": "yes"},
    )


@pytest.fixture
def stub_kappa(monkeypatch):
    calls: list[tuple[list, list]] = []
    stub = types.ModuleType("hos_pipeline.stats.core")

    def cohens_kappa(a, b):
        calls.append((list(a), list(b)))
        return 0.5

    stub.cohens_kappa = cohens_kappa
    monkeypatch.setitem(sys.modules, "hos_pipeline.stats.core", stub)
    return calls


def test_irr_compare(tmp_path, stub_kappa):
    a = [_coded("1", "yes", "no", "llm"), _coded("2", "yes", "unclear", "llm"),
         _coded("3", "no", "yes", "llm"), _coded("4", "unclear", "no", "llm")]
    b = [_coded("1", "yes", "no", "AB"), _coded("2", "no", "no", "AB"),
         _coded("3", "no", "yes", "AB"), _coded("5", "yes", "yes", "AB")]
    write_jsonl(tmp_path / "a.jsonl", a)
    write_jsonl(tmp_path / "b.jsonl", b)
    df = compare(tmp_path / "a.jsonl", tmp_path / "b.jsonl").set_index("item_id")
    assert list(df.index) == ["H1", "H2", "D1", "health_impact_raised"]
    h2 = df.loc["H2"]
    assert h2.n == 3  # cases 1-3 coded by both
    assert h2.agreement == pytest.approx(2 / 3)  # unclear vs no disagrees
    assert h2.agreement_collapsed == pytest.approx(1.0)  # unclear -> no agrees
    assert h2.kappa == 0.5 and h2.prevalence_unclear_a == pytest.approx(1 / 3)
    assert h2.prevalence_yes_a == pytest.approx(1 / 3) and h2.prevalence_yes_b == pytest.approx(1 / 3)
    assert (["no", "unclear", "yes"], ["no", "no", "yes"]) in stub_kappa
    assert (["no", "no", "yes"], ["no", "no", "yes"]) in stub_kappa
    assert df.loc["H1"].agreement == pytest.approx(2 / 3)
    sub = compare(tmp_path / "a.jsonl", tmp_path / "b.jsonl", items=["H1"])
    assert list(sub.item_id) == ["H1"]


def test_irr_export_and_human_sheet_roundtrip(tmp_path, stub_kappa):
    s = settings_for(tmp_path)
    write_inputs(s, ["I1", "I2"])
    cb = load_codebook(CODEBOOK)
    coded = [postprocess(CodingOutput.model_validate(coding_output(c)), make_decision(c), cb, MODEL) for c in ["I1", "I2"]]
    write_jsonl(s.work("coded.jsonl"), coded)
    sheet, llm_file = export_irr_sheet(s, ["I1", "I2"], s.work("outputs", "irr_sheet.csv"))
    import pandas as pd

    blank = pd.read_csv(sheet, dtype=str, keep_default_na=False)
    n_items = len(cb.hypothesis_ids) + len(cb.dimension_ids) + len(cb.admin_flag_ids)
    assert len(blank) == 2 * n_items and (blank.value == "").all()
    assert blank.landlord.iloc[0] == "Landlord I1"
    llm = pd.read_csv(llm_file, dtype=str, keep_default_na=False)
    assert set(llm.case_id) == {"I1", "I2"} and llm.loc[llm.item_id == "H1", "value"].tolist() == ["yes", "yes"]

    # The human fills in H1 and H2 only; blanks are "not coded".
    blank.loc[blank.item_id == "H1", "value"] = ["Y", "n"]
    blank.loc[blank.item_id == "H1", "citations"] = ["p4", ""]
    blank.loc[blank.item_id == "H2", "value"] = ["no", "?"]
    blank["coder"] = "AB"
    blank.to_csv(sheet, index=False)
    human = load_human_codes(sheet)
    assert {c.case_id for c in human} == {"I1", "I2"}
    i1 = next(c for c in human if c.case_id == "I1")
    assert i1.coder == "AB" and i1.hypotheses[0].citations == [Citation(pos=4)]

    df = compare(s.work("coded.jsonl"), sheet).set_index("item_id")
    assert list(df.index) == ["H1", "H2"]  # only items both coded
    assert df.loc["H1"].agreement == pytest.approx(0.5)
    assert df.loc["H2"].agreement == pytest.approx(0.5) and df.loc["H2"].agreement_collapsed == pytest.approx(1.0)


def test_irr_kappa_guarded_when_stats_missing(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "hos_pipeline.stats.core", None)  # import raises ImportError
    write_jsonl(tmp_path / "a.jsonl", [_coded("1", "yes", "no", "x"), _coded("2", "no", "no", "x")])
    with pytest.warns(UserWarning, match="cohens_kappa"):
        df = compare(tmp_path / "a.jsonl", tmp_path / "a.jsonl")
    assert math.isnan(df.kappa.iloc[0]) and df.agreement.iloc[0] == 1.0
