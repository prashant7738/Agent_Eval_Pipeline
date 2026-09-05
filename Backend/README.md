# Agent Eval Pipeline

Four algorithms for the paper: ROUGE-L, Bradley-Terry/Elo, SelfCheckGPT, Cohen's Kappa,
plus the agent, GAIA loader, and human-study scripts that tie them together end to end.

## Setup
Dependencies are managed with [uv](https://docs.astral.sh/uv/):
```
cd files
uv sync            # creates files/.venv from pyproject.toml + uv.lock
uv run python agent.py   # runs any script inside the managed venv
```
Fill in `files/.env`:
```
GROQ_API_KEY=your-free-key-from-console.groq.com
HF_TOKEN=your-huggingface-token
```
GAIA (`gaia-benchmark/GAIA` on HuggingFace) is a **gated dataset** — visit the dataset
page while logged in as the account behind `HF_TOKEN` and click "Agree and access
repository" before `gaia_loader.py` will work.

## Algorithm modules (no orchestration, tested standalone)
- `rouge_l.py` — no API needed.
- `bradley_terry.py` — no API needed.
- `kappa.py` — no API needed.
- `llm_judge.py` — needs GROQ_API_KEY. Pairwise comparison + 4-category rubric labeling.
- `selfcheckgpt.py` — needs GROQ_API_KEY. Samples k responses, scores consistency.

## Agent backend: Groq
`agent.py` runs a single Groq chat completion (Llama + DuckDuckGo snippets) per
task. It struggles on GAIA's multi-hop lookup tasks (mean ROUGE-L F1 ~0.09),
mostly because DuckDuckGo snippets rarely contain the precise fact needed, not
primarily because of the model itself.

## Pipeline scripts (run in this order)
1. `gaia_loader.py` — pulls a stratified ~18-task text-only subset of GAIA validation, saves `data/gaia_tasks.json`.
2. `agent.py` — the single agent config (web search via `web_search.py` + one Groq call) used everywhere below.
3. `run_study1.py` — Day 2-3: generates one response per task, scores ROUGE-L, then samples k=3 responses on a subset for SelfCheckGPT consistency + a per-task Bradley-Terry ranking of those samples.
4. `prepare_human_study.py` — Day 4: builds a 30-item labeling pool (main responses + SelfCheckGPT samples), writes `data/annotator1_sheet.csv` / `annotator2_sheet.csv` for blind independent labeling, and gets the LLM judge's labels on the same pool.
5. `score_human_study.py` — Day 5: computes human-human Cohen's kappa, writes `data/adjudication_needed.csv` for any disagreements, then (once adjudicated) computes judge-human kappa with the confusion matrix.

All generated data lands in `files/data/` (gitignored).

## Results (real GAIA data, Groq backend)

### Study 1 — automatic metrics
- **ROUGE-L F1**: mean = 0.086 (n = 18 tasks). Nearly all individual scores are 0 —
  the single-call agent + shallow DuckDuckGo snippets can't do GAIA's multi-hop
  lookups. This is a legitimate finding about the agent architecture, not a bug.
- **SelfCheckGPT consistency**: mean = 0.125 (n = 8-task subset, k = 3 samples each).
- Per-task Bradley-Terry ranking computed over the k samples for the same subset.

Data: `data/responses.json`, `data/rouge_scores.json`, `data/selfcheck_results.json`.

### Study 2 — human study (30-item pool: main responses + SelfCheckGPT samples)
- **Human-human Cohen's kappa** (annotator1 vs annotator2): **0.7047** (substantial
  agreement), observed agreement = 0.8333, n = 30. Above the 0.6 threshold, so no
  rubric re-pilot was needed. 5 of 30 items were disagreements (all
  incorrect-vs-hallucinated calls), adjudicated in `data/adjudication_needed.csv`.
- **Judge-human Cohen's kappa** (LLM judge vs adjudicated human labels): **0.1327**
  (slight agreement), observed agreement = 0.4333, n = 30.
  Confusion matrix (rows = human, cols = judge):

  | human \ judge | correct | partial | incorrect | hallucinated |
  |---|---|---|---|---|
  | correct | 2 | 0 | 0 | 0 |
  | incorrect | 0 | 0 | 11 | 0 |
  | hallucinated | 0 | 0 | 17 | 0 |

  Reportable finding: the judge agrees perfectly with humans on `correct` (2/2) and
  `incorrect` (11/11) but never outputs `hallucinated` — every human-labeled
  "hallucinated" item was called `incorrect` by the judge instead. The judge model
  reliably detects wrong answers but doesn't discriminate specific unsupported/
  fabricated claims from plain wrongness. Worth a note in the paper's
  discussion/limitations on LLM-as-judge label granularity.

Data: `data/annotator1_sheet.csv`, `data/annotator2_sheet.csv`, `data/judge_labels.json`,
`data/adjudication_needed.csv`, `data/final_human_labels.json`.

## Free LLM notes
Using Groq (console.groq.com) — free tier, OpenAI-compatible, fast Llama models.
Rate limits apply on the free tier (check console for current numbers) — if you hit
limits while running ~20 tasks x k samples, add a small `time.sleep()` between calls,
or batch requests across a day.

Zero-limit local alternative: install Ollama (ollama.com), pull a model
(`ollama pull llama3.1`), then swap the Groq client calls for a local
`requests.post("http://localhost:11434/api/chat", ...)` call instead. Same
prompts/logic, just a different backend -- ask if you want this variant written out.
