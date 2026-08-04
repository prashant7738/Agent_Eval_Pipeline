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

## Agent backend: Groq (default) vs. Gemini
`agent.py` supports two backends. The results already saved in `files/data/` were
generated with **Groq** (Llama + DuckDuckGo snippets), which struggles on GAIA's
multi-hop lookup tasks (mean ROUGE-L F1 ~0.09) mostly because DuckDuckGo snippets
rarely contain the precise fact needed, not primarily because of the model itself.

To try **Gemini** instead (native Google Search grounding tool, does real iterative
search rather than one snippet dump):
```
uv sync --extra gemini
```
Add to `files/.env`:
```
GEMINI_API_KEY=your-free-key-from-aistudio.google.com
```
Then set `AGENT_BACKEND=gemini` before re-running `run_study1.py` (or pass
`backend="gemini"` directly to `run_agent()`). Switching the backend does not
overwrite existing data — re-running writes over `data/responses.json` etc., so
back up the Groq-generated results first if you want to keep both as a comparison.

## Pipeline scripts (run in this order)
1. `gaia_loader.py` — pulls a stratified ~18-task text-only subset of GAIA validation, saves `data/gaia_tasks.json`.
2. `agent.py` — the single agent config (web search via `web_search.py` + one Groq call) used everywhere below.
3. `run_study1.py` — Day 2-3: generates one response per task, scores ROUGE-L, then samples k=3 responses on a subset for SelfCheckGPT consistency + a per-task Bradley-Terry ranking of those samples.
4. `prepare_human_study.py` — Day 4: builds a 30-item labeling pool (main responses + SelfCheckGPT samples), writes `data/annotator1_sheet.csv` / `annotator2_sheet.csv` for blind independent labeling, and gets the LLM judge's labels on the same pool.
5. `score_human_study.py` — Day 5: computes human-human Cohen's kappa, writes `data/adjudication_needed.csv` for any disagreements, then (once adjudicated) computes judge-human kappa with the confusion matrix.

All generated data lands in `files/data/` (gitignored).

## Free LLM notes
Using Groq (console.groq.com) — free tier, OpenAI-compatible, fast Llama models.
Rate limits apply on the free tier (check console for current numbers) — if you hit
limits while running ~20 tasks x k samples, add a small `time.sleep()` between calls,
or batch requests across a day.

Zero-limit local alternative: install Ollama (ollama.com), pull a model
(`ollama pull llama3.1`), then swap the Groq client calls for a local
`requests.post("http://localhost:11434/api/chat", ...)` call instead. Same
prompts/logic, just a different backend -- ask if you want this variant written out.
