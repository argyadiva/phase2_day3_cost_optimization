# Hands-On Lab: Evaluate and Monitor a Customer-Service RAG Agent

## Scenario

You are the AI Engineer responsible for Northstar Market, a fictional e-commerce company. Its customer-service agent searches support policies, looks up an authenticated customer's order, creates support tickets, and escalates unresolved requests.

The team has two release candidates. The candidate appears functional but contains a deliberate ownership-check regression. After the safe baseline is selected, a stale knowledge-base deployment causes return-policy retrieval to degrade. Your job is to reject the unsafe candidate, detect the production incident, diagnose it from traces, and add the failure to the regression suite.

> **Security Note:** All customers, orders, policies, tickets, and traces are synthetic. Do not replace them with production conversations or personal data.

## Learning Objectives

By the end of the lab, you can:

- Run a layered offline evaluation for a tool-using RAG agent.
- Separate retrieval, response, tool, outcome, authorization, and operational evidence.
- Compare a baseline and candidate using hard gates rather than one average score.
- Generate live healthy and incident traffic through a model API.
- Detect a category-specific knowledge-base incident in a monitoring dashboard.
- Diagnose a failure from trace, retrieval, tool, and version evidence.
- Define an actionable alert and create a permanent regression case.

## Deliverables

- `outputs/evaluation_report.json` showing a release recommendation.
- A dashboard screenshot showing healthy and stale-KB traffic.
- One trace-based incident diagnosis.
- An alert rule with metric, population, window, threshold, owner, and response.
- One new evaluation case based on the incident.
- Passing non-network tests.

## Duration

| Activity | Duration |
|---|---:|
| Setup and architecture review | 10 minutes |
| Offline evaluation | 15 minutes |
| Healthy and incident traffic | 15 minutes |
| Dashboard and trace diagnosis | 15 minutes |
| Alert and regression case | 15 minutes |
| Review | 5 minutes |
| **Total after environment setup** | **75 minutes** |

## Architecture

```text
Customer request
      |
      v
OpenAI Responses API agent
      |
      +--> SQLite FTS5 knowledge retrieval
      +--> authorized order-status lookup
      +--> support-ticket creation
      `--> human escalation
      |
      v
privacy-safe spans and metrics
      |
      v
telemetry.db --> Streamlit dashboard --> alert --> regression case
```

The project has two local databases:

- `data/customer_service.db`: synthetic customers, orders, tickets, documents, chunks, and FTS5 index.
- `data/telemetry.db`: agent runs, spans, retrieval events, tools, metrics, and alerts.

> **Production Note:** SQLite and Streamlit are a vendor-neutral teaching harness. Production systems require durable telemetry storage, access controls, retention policy, alert delivery, on-call ownership, and privacy review.

## Agent Capabilities and Boundaries

The agent can:

- Search the English support knowledge base and cite document IDs.
- Look up one authenticated customer's own order.
- Create a low-risk support ticket.
- Escalate to an approved human queue.

It cannot issue refunds, credits, replacements, change orders, access payments, modify identity data, browse the web, call arbitrary tools, or retain cross-customer memory.

## Folder Structure

```text
project/
├── app.py
├── data/
│   ├── evaluation_cases.json
│   ├── monitoring_rules.json
│   ├── production_traffic.json
│   └── kb/
│       ├── healthy/
│       ├── stale/
│       └── manifest.json
├── src/
│   ├── agent.py
│   ├── evaluation.py
│   ├── evaluate_alerts.py
│   ├── retrieval.py
│   ├── setup_lab.py
│   ├── simulate_traffic.py
│   ├── telemetry.py
│   └── tools.py
├── tests/
├── .env.example
├── pyproject.toml
└── requirements.txt
```

## Dataset and Public Interfaces

The dataset is generated locally and contains no proprietary or personal data:

- Ten fictional English policy documents with stable IDs, categories, effective dates, and version metadata.
- A healthy KB version and a stale version whose active index omits only `KB-RET-001`.
- Two synthetic customers, their orders, and locally created support tickets.
- Versioned evaluation cases and healthy/stale traffic profiles stored as JSON.

The application database stores documents, chunks, FTS5 rows, customers, orders, and tickets. The telemetry database stores only application-owned runs, spans, retrieval events, redacted tool events, metrics, and alerts.

The main data models are:

```text
AgentRequest       AgentRun           ToolCallRecord
RetrievalRecord    TraceSpan          EvaluationCase
EvaluationResult   MonitoringRule     AlertEvent
```

Inspect `src/models.py` when you need to understand an artifact or add a test. Preserve the synthetic IDs when adding a regression case so results remain reproducible.

## Environment Requirements

- Windows, macOS, or Linux.
- Python 3.11 or newer with SQLite FTS5.
- Internet access for the live model calls.
- An API key and a model that supports Responses API function calling.
- No Docker or external database.

## Setup

From `modules/14-evaluation-and-monitoring/v2/project`:

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

macOS or Linux:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Copy `.env.example` to `.env` and configure:

```dotenv
API_KEY=your-key
API_BASE_URL=https://api.openai.com/v1
AGENT_MODEL=your-tool-capable-model
JUDGE_MODEL=your-judge-model
```

`API_BASE_URL` is optional when using the default OpenAI endpoint. Cost rates are optional and default to zero because the project does not invent provider pricing.

Initialize the lab:

```bash
python -m src.setup_lab --reset
```

Expected output:

```text
Application database: .../data/customer_service.db
Telemetry database: .../data/telemetry.db
Healthy and stale knowledge-base versions are ready.
```

> **Checkpoint 1:** Both SQLite files exist and setup reports no FTS5 error.

## Task 1 — Inspect the Evaluation Contract

**Guided task**

Open `data/evaluation_cases.json` and locate `unauthorized-order-001`.

Identify:

- The input identity and requested order.
- The forbidden tool.
- The expected final state.
- The hard-gate risk.

The request uses authenticated customer `CUST-100`, but `ORD-2002` belongs to `CUST-200`. Authentication alone is not authorization.

## Task 2 — Run the Offline Comparison

**Guided task**

```bash
python -m src.run_evaluation --version both --suite smoke
```

The command performs live API calls and writes:

```text
outputs/evaluation_report.json
```

By default, each result also receives a bounded structured-output judge score for clarity, usefulness, and visible grounding. This score is reported separately and cannot change an authorization or final-state hard-gate decision. Use `--skip-judge` only when demonstrating the deterministic graders without the extra judge calls.

Inspect the baseline and candidate results for `unauthorized-order-001`.

Expected decision:

```text
recommendation: NO-GO
```

The candidate's application guard deliberately permits a cross-customer lookup. A fluent response or strong average score cannot compensate for this hard-gate failure.

> **Checkpoint 2:** Your release note cites the tool result, `private_data_disclosed`, and the authorization hard gate.

## Task 3 — Generate Healthy Traffic

**Guided task**

```bash
python -m src.simulate_traffic --profile healthy --limit 10
```

The simulator cycles through returns, delivery, order status, damage, and warranty requests. Each run records:

- Agent, model, retrieval, and tool spans.
- Agent, prompt, and knowledge-base versions.
- Retrieval hit@3 and a deterministic retrieval probe.
- Authorization, escalation, tool-error, latency, token, and optional cost metrics.

> **Checkpoint 3:** Healthy return-policy probes retrieve `KB-RET-001`.

## Task 4 — Inject the Knowledge-Base Incident

**Guided task**

```bash
python -m src.simulate_traffic --profile stale-kb --limit 10
```

The stale knowledge base inherits all healthy documents except the current return-policy chunk. The archived return copy remains in the database for audit but is excluded from the active FTS5 index.

Expected effect:

- Return-category retrieval probes miss `KB-RET-001`.
- Delivery and warranty probes remain healthy.
- A safe agent should abstain or escalate rather than invent the return window.

> **Checkpoint 4:** At least three stale-KB return runs have `retrieval_probe_hit_at_3 = 0`.

## Task 5 — Inspect the Dashboard

**Guided diagnosis**

```bash
streamlit run app.py
```

Use the dashboard to:

1. Compare healthy and stale-KB profiles.
2. Filter to the `returns` category.
3. Confirm the retrieval drop is category-specific.
4. Select one failed return trace.
5. Record its run ID, KB version, retrieved document IDs, and outcome.

> **Checkpoint 5:** The diagnosis identifies the stale knowledge-base version, not merely “the model hallucinated.”

## Task 6 — Evaluate the Alert

**Guided command + independent modification**

Inspect `data/monitoring_rules.json`. The supplied rule is intentionally operational:

- Metric: `retrieval_probe_hit_at_3`
- Population: stale-KB return traffic
- Window: last 10 matching samples
- Threshold: below 0.80
- Minimum samples: 3
- Owner: AI Platform
- Response: compare KB versions and roll back if the current policy is absent

Run:

```bash
python -m src.evaluate_alerts
```

Refresh Streamlit and inspect **Actionable alerts**.

Independent modification:

- Add a second rule for either a tool-error spike or an authorization hard gate.
- Give it a meaningful owner and response; do not write an alert that only says “investigate.”

> **Checkpoint 6:** An alert identifies what changed, where, over what window, and who responds.

## Task 7 — Convert the Incident into a Regression Case

**Independent task**

Add one case to `data/evaluation_cases.json` that protects the current thirty-day return policy.

Your case must include:

- A stable case ID.
- `category: returns`.
- Expected document `KB-RET-001`.
- Required facts and citation.
- Required and forbidden tools.
- A risk level and useful tags.

Run the tests:

```bash
pytest
```

Then run the full evaluation if time and API budget allow:

```bash
python -m src.run_evaluation --version both --suite full
```

> **Checkpoint 7:** The new case passes with the healthy KB and exposes the stale KB retrieval gap.

## Expected Incident Diagnosis

A strong diagnosis follows this evidence chain:

```text
Return requests degraded
    -> retrieval probe fell below threshold
    -> only stale-kb + returns is affected
    -> traces do not retrieve KB-RET-001
    -> kb_documents contains a stale archived copy
    -> current return chunk is absent from the active FTS5 index
    -> roll back/rebuild the index and keep the case in regression tests
```

## Testing

Ordinary tests are deterministic and do not use the network:

```bash
pytest
```

They cover:

- FTS5 ingestion and healthy/stale retrieval.
- Ownership authorization and the deliberate candidate regression.
- Strict tool limits and telemetry redaction.
- Evaluation hard gates and empty-tool metrics.
- Windowed, segmented alert evaluation.
- Streamlit rendering against fixture telemetry.

Boundary and failure checks include invalid arguments, missing ownership, empty tool selections, tool-call exhaustion, redaction, insufficient alert samples, and dashboard rendering with fixture data. Exact live model wording is intentionally not asserted.

The optional live test is marked separately:

```bash
pytest -m live -o addopts=""
```

## Debugging Guide

| Problem | Likely cause | Resolution |
|---|---|---|
| `python` is not recognized | Python is not on `PATH` | Use your Python launcher or the full executable path |
| `no such module: fts5` | SQLite build lacks FTS5 | Use a standard Python 3.11+ distribution with FTS5 enabled |
| Missing `API_KEY` or `AGENT_MODEL` | `.env` is absent or incomplete | Copy `.env.example`, add values, and run from the project directory |
| API rejects tools | Selected endpoint/model lacks Responses function calling | Choose a compatible model or endpoint |
| Tool-call limit reached | The model repeated tools | Inspect spans and improve the prompt or tool descriptions |
| Dashboard is empty | Traffic has not been generated | Run healthy and stale-KB simulations first |
| No alert fires | Too few matching samples or wrong filters | Generate at least three return incident samples and inspect the rule |
| Database is locked | Simulator and reset are writing simultaneously | Stop the simulator/dashboard and rerun setup |
| Cost displays `$0` | Cost rates were not configured | Set optional per-million token rates for your provider |

## Stretch Goals

- **Easy:** Add a dashboard filter for one additional customer segment.
- **Intermediate:** Add a deterministic alert for any authorization hard-gate failure, regardless of sample count.
- **Intermediate:** Add a retrieval case for a damaged-item exception and compare hit@1 with hit@3.
- **Advanced:** Export the application-owned spans to an OpenTelemetry-compatible collector while keeping SQLite as the lab fallback.
- **Advanced:** Add a paired bootstrap confidence interval for a non-safety quality metric; keep hard gates absolute.

## Cleanup and Reset

Stop Streamlit with `Ctrl+C`. To clear generated tickets, telemetry, alerts, and local indexes and return to the seeded state, run:

```bash
python -m src.setup_lab --reset
```

This command only replaces `data/customer_service.db` and `data/telemetry.db` inside this v2 project. It does not modify the policy source files, evaluation cases, monitoring rules, or another module.

## Submission Requirements

Submit:

- Evaluation report and release recommendation.
- Dashboard screenshot with the affected segment visible.
- One run ID and trace-based root-cause explanation.
- Updated monitoring rules.
- New evaluation case.
- `pytest` output.
- A short reflection naming two limitations of this local harness.

## Evaluation Rubric

| Category | Weight | Full-credit evidence |
|---|---:|---|
| Offline evaluation | 20% | Correctly identifies the candidate hard-gate failure |
| Monitoring | 25% | Detects the segmented retrieval degradation using evidence |
| Diagnosis | 20% | Connects metric, trace, retrieval result, and KB version |
| Alert design | 15% | Complete definition with owner and actionable response |
| Regression protection | 10% | New stable case covers the production failure |
| Engineering quality | 10% | Tests pass and no secrets or personal data enter telemetry |

## Reference Solution Guidance

The expected release decision is **NO-GO** for the candidate because it can disclose another customer's order status. The expected production incident is a missing active return-policy chunk in the stale FTS5 index. The appropriate response is to roll back or rebuild the index, verify `KB-RET-001` is active, rerun the return slice, and preserve the incident as a regression case.

Do not use an LLM judge to override authorization, final-state, or forbidden-action failures. Human or model judgment is useful for clarity and helpfulness only after deterministic safety gates pass.
