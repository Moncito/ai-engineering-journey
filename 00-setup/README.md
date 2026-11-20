# Phase 00 — Setup & Tooling

Goal: a reproducible Python environment, safe secret handling, and a test-driven workflow,
so every later phase is easy to run on any machine.

## Concepts

### 1. `uv` — one tool for Python, venvs, and packages
| Old way | uv way |
|---|---|
| install Python manually / pyenv | `uv python install 3.12` |
| `python -m venv .venv` + activate | automatic (`.venv/` created for you) |
| `pip install x` + `requirements.txt` | `uv add x` (writes `pyproject.toml` + `uv.lock`) |
| `python script.py` | `uv run python script.py` |

- **`pyproject.toml`** — what you *want* (`python-dotenv>=1.2`).
- **`uv.lock`** — exactly what you *got* (every version + hash). Commit it, so anyone running
  `uv sync` gets an identical environment. Reproducibility matters a lot in ML: different library
  versions can give different results.
- **`.python-version`** — pins the interpreter (3.12).
- `uv run` always uses the project's `.venv`, so you never "forget to activate".

### 2. Secrets: `.env` + `python-dotenv`
API keys are passwords that cost money. Rules:
1. Keys live in `.env` (git-ignored). `.env.example` (committed) documents *which* keys exist.
2. Code reads them with `os.getenv("ANTHROPIC_API_KEY")` after `load_dotenv()`.
3. Never print, log, or commit a full key. That's why `mask_secret` exists.
4. If a key ever reaches GitHub, **revoke it immediately**: bots scan public repos in minutes.

### 3. Tests as a spec (`pytest`)
`test_env_check.py` defines the expected behavior, and you write code until it passes.
In AI engineering you'll test data pipelines, parsers, prompt templates, and eval metrics
this way. Model outputs are fuzzy, so the code around them needs to be solid.

Key idea in this exercise: **separate pure logic from I/O.**
`parse_nvidia_smi(text)` is pure (string in, data out), so it's easy to test with fake input, even an
80GB A100 you don't own. `gpu_info()` does the I/O (runs a program) and just calls the parser.

### 4. Lint (`ruff`)
`uv run ruff check .` catches unused imports, bugs, and messy import order.
`uv run ruff format .` auto-formats. Run both before every commit.

### 5. Git workflow
One exercise = one or more small commits with clear messages:
```
feat(00-setup): implement mask_secret
test(00-setup): ...
docs(00-setup): add what-I-learned notes
```

## Exercise

1. Open `env_check.py`. Implement the 4 functions marked `NotImplementedError`.
2. Loop until green:
   ```
   uv run pytest 00-setup
   ```
3. Run the real thing:
   ```
   uv run python 00-setup/env_check.py
   ```
4. Copy `.env.example` to `.env`. Put a fake value in it (`ANTHROPIC_API_KEY=sk-ant-test-1234567890`)
   and re-run. The key should show masked, never in full.
5. Lint clean: `uv run ruff check .` then `uv run ruff format .`
6. Fill in "What I learned" below, commit, push.

**Stretch:** add a `test_gpu_info_returns_empty_when_no_nvidia_smi` test using pytest's
`monkeypatch` to fake `shutil.which` returning `None`.

## What I learned

<!-- Your own words. 3–5 bullets. This is what recruiters read. -->
-
