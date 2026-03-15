# Contributing to Sentinel

Thanks for your interest. Contributions of all kinds are welcome — bug fixes, new builtin models, docs improvements, or new features.

---

## Setup

```bash
git clone https://github.com/your-org/sentinel
cd sentinel
uv sync
cp .env.example .env
```

## Running tests

```bash
uv run pytest tests/unit -v
```

## Linting

```bash
uv run ruff check .
uv run ruff format .
```

## Adding a builtin model

1. Create a directory under `models/builtin/your-model-name/`
2. Add `manifest.yaml`, `architecture.py`, `retrain.py`, `infer.py`
3. Follow the contract in [docs/adding-a-model.md](docs/adding-a-model.md)
4. Add unit tests in `tests/unit/`
5. Add the model name to `BUILTIN_NAMES` in `cli/commands/add.py`
6. Open a pull request

## Pull request checklist

- [ ] Tests pass locally (`uv run pytest tests/unit -v`)
- [ ] Lint passes (`uv run ruff check .`)
- [ ] New code has docstrings
- [ ] `sentinel.yaml` schema changes are reflected in `core/config.py` and `docs/configuration.md`