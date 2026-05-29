# Contributing

Thanks for your interest. This is a reference / portfolio project, but issues and PRs are welcome.

## Dev setup

```bash
pip install -e ".[dev]"      # offline mode — no API keys needed
pytest -q                    # run the test suite
ruff check src evals tests   # lint
python -m evals.run_evals    # the eval gate (must pass)
```

## Ground rules

- **Everything must pass offline.** Tests and the eval gate run with the deterministic `FakeLLM` +
  hashing embedder — no network, no keys. Don't add tests that require a live provider to the default path.
- **The eval gate is a gate.** If you change retrieval/agent logic, keep `python -m evals.run_evals`
  green (or update thresholds in `evals/run_evals.py` with justification).
- **Add a test for every fix.** Found a bug? Add a case to `tests/` or a row to `evals/dataset.jsonl`.
- **Keep secrets out.** Never commit keys; `.env` is git-ignored. CI runs a SAST/secrets scan.
- **Style:** `ruff` clean, type hints on new public functions.

## PR checklist

- [ ] `ruff check` clean
- [ ] `pytest -q` green
- [ ] `python -m evals.run_evals` exits 0
- [ ] new behavior covered by a test
