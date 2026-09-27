# Contributing

Run:

```bash
python -m pip install -e ".[dev]"
pytest
```

Changes to provider routing should keep the deterministic fallback functional.

Never add a feature that executes raw LLM output.
