"""Command line entry point: ``hos-pipeline -c config.yaml <stage> [options]``.

Stages run in order and each reads the previous stage's JSONL from the work
directory, so any stage can be re-run on its own:

    inspect      show how the parser sees a few real archive pages
    ingest       archive html.gz  -> decisions.jsonl                 (stage 1)
    screen       decisions        -> screen.jsonl                    (stage 2)
    sample       screen           -> sample.jsonl, irr_sample.txt    (stage 2)
    code         sample           -> coded.jsonl         (API, grade B, 3-4)
    verify       coded            -> verified.jsonl      (API, grade A, 5)
    outcomes     screen           -> outcomes.jsonl      (API, replaces grade D)
    awaab        coded            -> awaab_checks.jsonl              (stage 6)
    table        all of the above -> analysis_table.csv
    report       all of the above -> outputs/ (tables, figures, summary.md)
    denominators RSH SDR + LAHS   -> outputs/landlord_rates.csv
    paragraphs   rewrite [pN] citations in a CSV/JSONL as official paragraph numbers
    dossier      right-of-reply pack for one landlord
    irr-export / irr-compare      second human reader on the 10% sample
    demo         synthetic data end to end, no archive and no API calls

Stages marked API call Claude and cost money. Run them with --dry-run first:
it writes sample requests and a cost estimate without calling the API.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .config import Settings, load_settings


def _ids(values: list[str] | None) -> list[str] | None:
    if not values:
        return None
    ids: list[str] = []
    for v in values:
        p = Path(v)
        if p.exists():
            ids += [line.strip() for line in p.read_text().splitlines() if line.strip()]
        else:
            ids += [x.strip() for x in v.split(",") if x.strip()]
    return ids


def cmd_inspect(s: Settings, a: argparse.Namespace) -> None:
    from .ingest.archive import load_html, read_manifest
    from .ingest.parse import describe_html

    entries = read_manifest(s.paths.archive_dir)
    chosen = _ids(a.case_ids) or sorted(entries)[:: max(1, len(entries) // a.n)][: a.n]
    for cid in chosen:
        html, _ = load_html(entries[cid])
        print(f"===== {cid}  {entries[cid].url}\n{describe_html(html)}\n")


def cmd_ingest(s: Settings, a: argparse.Namespace) -> None:
    from .ingest.run import ingest_archive

    ids = _ids(a.case_ids)
    print(ingest_archive(s, limit=a.limit, case_ids=set(ids) if ids else None, verify_sha=not a.no_sha))


def cmd_screen(s: Settings, a: argparse.Namespace) -> None:
    from .screen.screen import run_screen

    print(run_screen(s))


def cmd_sample(s: Settings, a: argparse.Namespace) -> None:
    from .screen.sample import run_sample

    print(run_sample(s))


def _llm_kwargs(a: argparse.Namespace) -> dict:
    return dict(case_ids=_ids(a.case_ids), limit=a.limit, dry_run=a.dry_run, mode=a.mode)


def cmd_code(s: Settings, a: argparse.Namespace) -> None:
    from .llm.coder import run_coding

    print(run_coding(s, **_llm_kwargs(a)))


def cmd_verify(s: Settings, a: argparse.Namespace) -> None:
    from .llm.verifier import run_verification

    print(run_verification(s, **_llm_kwargs(a)))


def cmd_outcomes(s: Settings, a: argparse.Namespace) -> None:
    from .llm.outcomes import recall_report, run_outcomes

    print(run_outcomes(s, **_llm_kwargs(a)))
    if not a.dry_run and s.work("outcomes.jsonl").exists():
        print(recall_report(s).to_string(index=False))


def cmd_awaab(s: Settings, a: argparse.Namespace) -> None:
    from .awaab.duties import run_awaab

    ids = _ids(a.case_ids)
    print(run_awaab(s, case_ids=set(ids) if ids else None, post_commencement_only=a.post_commencement_only))


def cmd_table(s: Settings, a: argparse.Namespace) -> None:
    from .stats.table import build_analysis_table

    df = build_analysis_table(s)
    print(f"{len(df)} rows -> {s.work('analysis_table.csv')}")


def cmd_report(s: Settings, a: argparse.Namespace) -> None:
    from .report.build import build_report

    print(build_report(s))


def cmd_denominators(s: Settings, a: argparse.Namespace) -> None:
    from .denominators.rsh import run_denominators

    print(run_denominators(s))


def cmd_paragraphs(s: Settings, a: argparse.Namespace) -> None:
    from .ingest.paragraphs import convert_file

    n = convert_file(a.input, s.work("decisions.jsonl"), a.output, columns=a.columns, case_id_column=a.case_id_column)
    print(f"{n} rows -> {a.output}")


def cmd_dossier(s: Settings, a: argparse.Namespace) -> None:
    from .report.dossier import landlord_dossier

    print(landlord_dossier(s, a.landlord, post_commencement_only=a.post_commencement_only))


def cmd_irr_export(s: Settings, a: argparse.Namespace) -> None:
    from .llm.irr import export_irr_sheet

    ids_file = s.work("irr_sample.txt")
    ids = _ids(a.case_ids) or [x for x in ids_file.read_text().split() if x]
    out = Path(a.out) if a.out else s.work("outputs", "irr_sheet.csv")
    export_irr_sheet(s, ids, out)
    print(out)


def cmd_irr_compare(s: Settings, a: argparse.Namespace) -> None:
    from .llm.irr import compare

    df = compare(Path(a.coded_a), Path(a.coded_b))
    out = s.work("outputs", "irr_agreement.csv")
    df.to_csv(out, index=False)
    print(df.to_string(index=False))
    print(out)


def cmd_demo(s: Settings, a: argparse.Namespace) -> None:
    from .report.build import build_report
    from .stats.table import build_analysis_table
    from .synthetic import make_synthetic_work_dir

    if a.work_dir:
        s.paths.work_dir = Path(a.work_dir).resolve()
    make_synthetic_work_dir(s.paths.work_dir, n_decisions=a.n, seed=a.seed)
    build_analysis_table(s)
    print(build_report(s))


def _add_llm_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--dry-run", action="store_true", help="write sample requests and a cost estimate; no API call")
    p.add_argument("--limit", type=int, help="process at most N decisions")
    p.add_argument("--case-ids", nargs="*", help="ids, comma lists, or files of ids")
    p.add_argument("--mode", choices=["auto", "batch", "async"], default="auto")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="hos-pipeline", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-c", "--config", help="config YAML (default: built-in defaults)")
    sub = ap.add_subparsers(dest="stage", required=True)

    p = sub.add_parser("inspect", help="describe how a few archive pages parse")
    p.add_argument("-n", type=int, default=5)
    p.add_argument("--case-ids", nargs="*")
    p.set_defaults(fn=cmd_inspect)

    p = sub.add_parser("ingest", help="stage 1: parse the archive")
    p.add_argument("--limit", type=int)
    p.add_argument("--case-ids", nargs="*")
    p.add_argument("--no-sha", action="store_true", help="skip SHA-256 checks against manifest.tsv")
    p.set_defaults(fn=cmd_ingest)

    sub.add_parser("screen", help="stage 2: damp/mould screening").set_defaults(fn=cmd_screen)
    sub.add_parser("sample", help="stage 2: stratified random sample").set_defaults(fn=cmd_sample)

    for name, fn, helptext in [
        ("code", cmd_code, "stages 3-4: structured coding (API)"),
        ("verify", cmd_verify, "stage 5: strict second reader (API)"),
        ("outcomes", cmd_outcomes, "Determination-section classifier for all damp decisions (API)"),
    ]:
        p = sub.add_parser(name, help=helptext)
        _add_llm_args(p)
        p.set_defaults(fn=fn)

    p = sub.add_parser("awaab", help="stage 6: Awaab's Law duty checks")
    p.add_argument("--case-ids", nargs="*")
    p.add_argument("--post-commencement-only", action="store_true")
    p.set_defaults(fn=cmd_awaab)

    sub.add_parser("table", help="build analysis_table.csv").set_defaults(fn=cmd_table)
    sub.add_parser("report", help="tables, figures and summary.md").set_defaults(fn=cmd_report)
    sub.add_parser("denominators", help="RSH stock denominators and rates").set_defaults(fn=cmd_denominators)

    p = sub.add_parser("paragraphs", help="convert [pN] citations to official paragraph numbers")
    p.add_argument("input")
    p.add_argument("output")
    p.add_argument("--columns", nargs="+", required=True)
    p.add_argument("--case-id-column", default="case_id")
    p.set_defaults(fn=cmd_paragraphs)

    p = sub.add_parser("dossier", help="right-of-reply dossier for one landlord")
    p.add_argument("landlord", help="case-insensitive substring, e.g. 'Lambeth'")
    p.add_argument("--post-commencement-only", action="store_true")
    p.set_defaults(fn=cmd_dossier)

    p = sub.add_parser("irr-export", help="blank coding sheet for the second human reader")
    p.add_argument("--case-ids", nargs="*")
    p.add_argument("--out")
    p.set_defaults(fn=cmd_irr_export)

    p = sub.add_parser("irr-compare", help="agreement and Cohen's kappa between two coders")
    p.add_argument("coded_a")
    p.add_argument("coded_b")
    p.set_defaults(fn=cmd_irr_compare)

    p = sub.add_parser("demo", help="synthetic end-to-end run (no archive, no API)")
    p.add_argument("--work-dir")
    p.add_argument("-n", type=int, default=400)
    p.add_argument("--seed", type=int, default=1)
    p.set_defaults(fn=cmd_demo)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    settings = load_settings(args.config)
    args.fn(settings, args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
