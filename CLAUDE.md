# saudi-logistics-hub-classifier

Classifies potential logistics hub sites across Saudi Arabia.

## Setup
```bash
python -m venv .venv
source .venv/Scripts/activate
pip install -r requirements.txt
```

## Test
```bash
pytest tests/ -v
```

## Run
```bash
python -m src.main
```

## Key Files
- `src/hub_features.py` — feature extraction
- `src/classify.py` — classification logic
- `src/generate_sites.py` — synthetic site generation

Mixed docstring styles. Mixed case commits. Deps pinned with `>=`.
