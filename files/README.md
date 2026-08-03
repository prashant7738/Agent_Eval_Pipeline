# Agent Eval Pipeline

Four algorithms for the paper: ROUGE-L, Bradley-Terry/Elo, SelfCheckGPT, Cohen's Kappa.

## Setup
```
pip install -r requirements.txt
export GROQ_API_KEY="your-free-key-from-console.groq.com"
```

## Files
- `rouge_l.py` — no API needed. Tested, works standalone.
- `bradley_terry.py` — no API needed. Tested, works standalone.
- `kappa.py` — no API needed. Tested, works standalone.
- `llm_judge.py` — needs GROQ_API_KEY. Pairwise comparison + 4-category rubric labeling.
- `selfcheckgpt.py` — needs GROQ_API_KEY. Samples k responses, scores consistency.

## Free LLM notes
Using Groq (console.groq.com) — free tier, OpenAI-compatible, fast Llama models.
Rate limits apply on the free tier (check console for current numbers) — if you hit
limits while running ~20 tasks x k samples, add a small `time.sleep()` between calls,
or batch requests across a day.

Zero-limit local alternative: install Ollama (ollama.com), pull a model
(`ollama pull llama3.1`), then swap the Groq client calls for a local
`requests.post("http://localhost:11434/api/chat", ...)` call instead. Same
prompts/logic, just a different backend -- ask if you want this variant written out.
