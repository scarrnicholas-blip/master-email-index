# HOS damp and mould evidence pipeline

A local, reproducible pipeline that rebuilds the Evidence Pack's analysis from
the private archive of Housing Ombudsman decisions (`~/NoMug HOS Archive`,
18,333 decisions saved as of 30 October 2026). Every stage writes plain JSON
Lines or CSV into a work folder, so each stage can be checked, re-run or
replaced without redoing the others.

Nothing in this folder contains decision text. The archive is a private
reference copy and is not for republishing, so `work/`, `*.jsonl` and
`config.yaml` are git-ignored. Test fixtures are invented decisions.

## Stages

| Stage | Command | Reads | Writes | Evidence grade |
|---|---|---|---|---|
| 0 Check the parser | `inspect` | archive | (screen output) | |
| 1 Ingest | `ingest` | `html/*/*.html.gz`, `manifest.tsv` | `decisions.jsonl`, `ingest_report.csv` | |
| 2 Screen | `screen` | decisions | `screen.jsonl`, `screen_summary.csv` | D (keywords) |
| 2 Sample | `sample` | screen | `sample.jsonl`, `sample_strata.csv`, `irr_sample.txt` | |
| 3–4 Code | `code` | sample, decisions | `coded.jsonl` | B |
| 5 Second reader | `verify` | coded, decisions | `verified.jsonl` | A |
| Outcome classifier | `outcomes` | screen, decisions | `outcomes.jsonl`, `outputs/severe_recall_by_year.csv` | C (replaces D) |
| 6 Awaab's Law | `awaab` | coded, decisions | `awaab_checks.jsonl`, `outputs/awaab_summary.csv` | |
| Analysis table | `table` | all of the above | `analysis_table.csv` | |
| Statistics and pack | `report` | all of the above | `outputs/tables/`, `outputs/figures/`, `outputs/summary.md` | |
| Stock denominators | `denominators` | RSH SDR, LAHS, aliases | `outputs/landlord_rates.csv`, `entity_match_review.csv` | |
| Paragraph numbers | `paragraphs IN OUT --columns …` | any CSV/JSONL with `[pN]` | the same file with official paragraph numbers | |
| Right of reply | `dossier Lambeth` | coded, verified, awaab | `outputs/dossiers/lambeth.md`, `…_claims.csv` | |
| Second human reader | `irr-export`, `irr-compare A B` | sample | blank sheet; agreement and Cohen's kappa | |

Grades follow `src/hos_pipeline/models.py`. **A** means the strict second reader
confirmed the failure against the cited paragraphs. **B** means the decision was
coded from its full text. **C** means a structured field (a finding, an order
or an amount) was read from the decision. **D** means keyword screening only.

## Set up (Mac, once)

```bash
cd hos-evidence-pipeline
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
cp config.example.yaml config.yaml          # then edit paths if needed
export ANTHROPIC_API_KEY=...                # only for code / verify / outcomes
python -m pytest -q                         # all tests use invented data
hos-pipeline demo --work-dir /tmp/hos-demo  # synthetic end-to-end run, no API
```

## Running it on the archive

```bash
hos-pipeline -c config.yaml inspect -n 8      # 1. check the parser on real pages first
hos-pipeline -c config.yaml ingest            # 2. ~18k pages; resumable
hos-pipeline -c config.yaml screen
hos-pipeline -c config.yaml sample            #    1,667 stratified, seeded
hos-pipeline -c config.yaml code --dry-run    # 3. writes 3 example requests + a cost estimate
hos-pipeline -c config.yaml code --limit 20 --mode async   #    pilot, then read coded.jsonl
hos-pipeline -c config.yaml code              #    full run via the Batch API (half price)
hos-pipeline -c config.yaml verify
hos-pipeline -c config.yaml outcomes          # 4. all damp-flagged decisions
hos-pipeline -c config.yaml awaab
hos-pipeline -c config.yaml table
hos-pipeline -c config.yaml report            # 5. outputs/summary.md, tables, figures
```

`inspect` matters. Nobody here has seen the archive's HTML, so the parser was
written to the published text layout and several HTML numbering styles. Run
`inspect` on a few pages from different years. If the paragraph numbers,
sections or landlord names come out wrong, fix that before you ingest
everything; `ingest_report.csv` lists every page with numbering warnings.

Stages that call the API are resumable. A batch that is still running is picked
up again on the next run. Failures are written to `<stage>_errors.jsonl`, never
dropped. Each stage's token use goes to `<stage>_usage.jsonl`.

## Before coding: replace the draft codebook

`codebook/codebook.yaml` holds **draft** wording for H1–H12, D1–D13, the
administration flags and the event types. Replace the labels and definitions
with the project's coding manual and keep the ids. The coder and the second
reader take their instructions from this file, and the ids are what the
statistics read. The event type keys are what the Awaab's Law engine reads.

## How the known limits of the pack are handled

- **Severe maladministration found by keyword search (48% recall, 13% in 2026).**
  `outcomes` reads the Determination section of every damp-flagged decision and
  records each finding against its complaint element: damp and mould,
  complaint handling, repairs and so on. `severe_recall_by_year.csv` then
  measures how much the keyword rule missed in each year, against that reference.
- **`[p109]` positions instead of the Ombudsman's paragraph numbers.** Paragraph
  numbers are kept at ingestion. Coders cite positions, and `paragraphs`,
  `dossier` and the report convert them to "paragraph 23" or "paragraph 29(a)".
- **No stock denominators.** `denominators` joins landlords to the RSH SDR and
  LAHS stock through `codebook/entity_aliases.yaml`. That file covers
  Metropolitan Thames Valley, Nottingham City Homes, ALMOs, Southern Housing and
  Sovereign Network Group. It then gives rates per 1,000 homes with exact
  Poisson intervals. Names it can't match go to `entity_match_review.csv`.

## Statistics (all deterministic, seeded)

- 95% Wilson intervals for every proportion.
- Population-weighted pooled 2019–25 estimates. Stratum weight = damp-flagged
  population ÷ sampled. Intervals use the Kish effective sample size.
- Mann–Whitney tests on durations, with Holm correction across each family.
- A landlord clustering test with 2,000 within-year shuffles.
- Multivariable logistic regression (odds ratios, 95% CI) with cross-validated AUC.
- Cohen's kappa for the 10% second human reader.

## Awaab's Law (SI 2025/1042)

`awaab` tests the dated events in each coded decision against the duty table in
`src/hos_pipeline/awaab/duties.yaml`: 24-hour emergency action, 10 working days
to investigate, 3 working days for the written summary, 5 working days to start
safety work, and 12 weeks for preventive work. It counts England and Wales
working days with a built-in bank holiday calendar. That calendar handles
substitute days and one-off holidays, and gov.uk's `bank-holidays.json` can
override it.

Dates given only to the month are carried as a range. So a duty can be "met",
"breached" or "possibly breached". Where no action appears in the decision the
status is "no action recorded", which is not proof of a breach. Triggers before
27 October 2025 are reported separately as pre-commencement.
