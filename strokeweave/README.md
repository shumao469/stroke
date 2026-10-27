# StrokeWeave

**An auditable stroke-research workflow prototype.** TACTIC is retained as the
historical control-research name. This portable snapshot runs **generated
technical inputs only**. It does not replace the private, versioned workbench,
include its qualified model runtimes, or establish clinical capability.

## Quick start

Python 3.10+; no external package is required for the core. CPython 3.12 on
WSL/Linux is tested. Windows locking has a native implementation but is not
claimed as tested on native Windows in this release.

```bash
cd strokeweave
python run_strokeweave.py info
python run_tests.py
python run_strokeweave.py demo --state-dir runtime/demo
python run_strokeweave.py serve --host 127.0.0.1 --port 8801 --state-dir runtime/workbench
```

Open `http://127.0.0.1:8801/` after starting the server. Keep it local; no
external hosting, proxy or clinical network use. For source installation,
`python -m pip install -e .` provides `strokeweave info`, `demo` and `serve`.
The source-checkout commands do not require installation or internet access.

## What is implemented

- English workbench and **rule-based** natural-language task preview, requiring
  an explicit queue confirmation. It is not an LLM medical planner.
- Batch jobs, SQLite persistence, idempotency keys, an exclusive owner lock,
  charged attempts, bounded fallback and restart-safe status handling.
- Generated spatial/probability/unit contracts, exact failure records,
  integrity-checked results and an event hash chain.
- JSON and HTML reports, allowlisted ZIP audit export and local HTTP API.
- Standard-library exact least-squares utilities: rank checking, independent
  residuals, source binding, Bareiss elimination and a tighter componentwise
  certificate at the unchanged `1e-8` numerical criterion.

The five fixed demo scenarios return two completed measurements and three
intentional referrals. Rejection/referral is not a software failure disguised
as success, or a validated clinical triage decision. The volume is that of an
analytic generated cuboid, not patient lesion segmentation. Timing in the demo
receipt is tool-path time, **not complete HTTP or application-launch cost**.

### API example

```python
from strokeweave.workflow import WorkflowStore

store = WorkflowStore("runtime/research")
try:
    batch = store.submit({
        "request_key": "my_generated_demo_01",
        "tasks": [{"recipe": "generated_volume"}, {"recipe": "fallback_recovery"}],
    })
    store.run_pending()
    print(store.report(batch["job_ids"][0]))
finally:
    store.close()
```

Local API routes: `GET /api/health`, `/api/tools`, `/api/jobs`,
`/api/jobs/{id}`, `/report.json`, `/report.html`, `/export` under a job;
`POST /api/plans` and `/api/batches`. JSON POSTs require header
`X-StrokeWeave-Request: local-research-v1`. Only exact loopback hosts and
same-origin requests are accepted. See `SECURITY.md` before use.

## What is deliberately disabled

Patient tasks, imaging-file upload, label access, training and real-model
execution are disabled. DeepISLES and SynthStroke are labelled **external
plugins disabled**; no weights or restricted datasets are bundled. The old
private full-model synthetic integration is not silently represented by the
new generated tool. Each future adapter requires separate licence, modality,
training-exposure and data-admission checks.

Original CAL/HOLD/test-label protections and risk protocols are not changed.
The portable core has no patient-label paths. Its deterministic technical
contracts are **not** a claim to estimate or satisfy population clinical risk,
CVaR or clinical safety thresholds. It does not diagnose or advise treatment.

## Results: engineering improvement, not a superiority claim

The closed prior NIST campaign had a TACTIC/direct complete-job time ratio of
**0.983445494**, missing the locked `<=0.8` target. All policies used the same
exact-solve-plus-verification route. Six of **18** raw floating-point outputs
were inaccurate and rejected; four accurate outputs were also conservatively
rejected. These old tasks were not rerun or tuned for this release.

The new exact componentwise bound is tested to cover error and is tighter on
fresh analytic cases. One anisotropic implementation fixture demonstrates an
old scalar-bound refusal becoming a correct acceptance at the same tolerance.
This **does not establish that the four old refusals are resolved**. It is not
an independent real-world estimate of false-refusal reduction.

A prospective generated common-kernel check used four new matrix sizes and
five repeats per method (40 timings, not 40 independent scientific samples).
Bareiss/reference complete **warm** median ratios were approximately
**0.858, 0.928, 0.977, 0.944** for 4, 8, 12 and 16 columns. All exact solutions
matched. These small, non-exclusive CPU timings include statistic preparation,
solving and verification, but not process startup. They suggest a modest shared
implementation improvement, not `20%` complete-agent speedup. Direct, strong
gate and strong VOI must share the same improvement. No new controller-ranking
or cold-start performance claim is made.

Protocols and detailed results are under `protocols/` and `evidence/`.
[Historical results and publication boundaries](docs/HISTORICAL_RESULTS.md),
[numerical audit](docs/NUMERICS_AUDIT.md) and
[primary-method references](docs/RELATED_WORK.md) separate established methods
from unproven claims. The current evidence does **not** establish Nature
Communications readiness, a novel vision architecture or autonomous stroke care.

## Verification and remaining work

`python run_tests.py` exercises the generated workflows and independent
rational mathematical oracles. `python tools/verify_release.py` records tests,
checks the public-file allowlist and creates an integrity manifest. This is
automatic implementation review, not institutional cohort admission or a
clinical independent review. The predeclared common-kernel timing is run once
with `python tools/engineering_benchmark.py`; it refuses to overwrite its
existing output.

The browser-control connection was unavailable during release review; HTTP,
static assets and export behavior were tested, **not live visual acceptance**.
Native-Windows execution and patient/model plugin admission remain open.
Future work should be one bounded new scientific workflow with an independent
reference, equally optimized baselines, cold/warm costs and independently
measured application benefit; not repeated tuning of exposed tasks.

## Publication and licence

This is a public research snapshot, not a clinically released model. See
`LICENSE_PENDING.md`: the owner has not selected an open-source licence.
Public availability must not be confused with permission to redistribute under
MIT/Apache terms. `CITATION.cff` supplies a versioned software citation without
an invented DOI. The name is a research label, not trademark clearance.
