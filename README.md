# Saudi Logistics Hub Classifier

REST API and ML pipeline for classifying logistics hub locations across Saudi Arabia's rail corridors.

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Health check |
| `POST` | `/classify` | Classify a candidate location |
| `POST` | `/classify/batch` | Classify multiple locations |
| `GET` | `/corridors` | List corridor definitions and scores |
| `GET` | `/models` | Available models and current best |

### Classify a Location

```bash
curl -X POST http://localhost:8501/classify \
  -H "Content-Type: application/json" \
  -d '{
    "lat": 26.43,
    "lon": 50.10,
    "distance_to_rail_km": 12.5,
    "distance_to_highway_km": 8.0,
    "distance_to_port_km": 45.0,
    "population_50km_radius": 850000,
    "industrial_zones_nearby": 4,
    "commercial_activity_index": 8.2,
    "land_cost_index": 6.5,
    "water_availability_score": 7.0,
    "power_grid_capacity_mw": 320.0,
    "labor_market_size": 35000,
    "existing_warehouse_sqm": 120000,
    "avg_temperature": 34.5,
    "rail_freight_volume_nearby": 750000,
    "region": "Eastern"
  }'
```

**Response:**

```json
{
  "suitability": "optimal",
  "confidence": 0.87,
  "topsis_score": 0.74,
  "features": {
    "accessibility_score": 0.82,
    "market_potential_index": 9.1,
    "infrastructure_readiness": 0.71
  }
}
```

### Batch Classification

```bash
curl -X POST http://localhost:8501/classify/batch \
  -H "Content-Type: application/json" \
  -d '{"locations": [{"lat": 26.43, "lon": 50.10, ...}, {"lat": 21.54, "lon": 39.17, ...}]}'
```

**Response:**

```json
{
  "count": 2,
  "results": [
    {"name": "Dammam-East", "suitability": "optimal", "topsis_score": 0.74},
    {"name": "Jeddah-Port", "suitability": "good", "topsis_score": 0.62}
  ]
}
```

### Corridor Summary

```bash
curl http://localhost:8501/corridors
```

```json
{
  "corridors": [
    {
      "name": "Eastern",
      "description": "Dammam-Jubail-Ras Al-Khair industrial corridor",
      "num_locations": 60,
      "pct_optimal": 35.0,
      "pct_good": 28.3
    }
  ]
}
```

---

## Setup

```bash
git clone https://github.com/AhmedTAlZahrani/saudi-logistics-hub-classifier.git
cd saudi-logistics-hub-classifier
pip install -r requirements.txt
```

## Running

### Streamlit Dashboard

```bash
streamlit run app.py
```

### Generate Data

```python
from src.generate_sites import generate_locations, save_locations

locations = generate_locations(n=300)
save_locations(locations)
```

### Train and Evaluate

```python
from src.data_loader import load_location_data
from src.hub_features import HubFeatureBuilder
from src.classify import ClassificationBenchmark

X, y, metadata = load_location_data()
fe = HubFeatureBuilder()
X_processed = fe.fit_process(X)
y_encoded, encoder = fe.encode_target(y)

benchmark = ClassificationBenchmark()
results = benchmark.compare_models(X_processed, y_encoded)
print(results)
```

## Suitability Classes

| Class | Description |
|-------|-------------|
| `optimal` | Excellent accessibility, infrastructure, and market conditions |
| `good` | Strong fundamentals with minor gaps |
| `marginal` | Feasible but requires significant investment |
| `unsuitable` | Poor location for logistics operations |

## Models

- **XGBoost** (best): F1 = 87.1%
- **Random Forest**: F1 = 84.5%
- **SVM**: F1 = 81.9%
- **TOPSIS** (non-ML baseline): Accuracy = 78.3%

## Project Structure

```
saudi-logistics-hub-classifier/
├── src/
│   ├── __init__.py
│   ├── generate_sites.py      # Synthetic location data generation
│   ├── data_loader.py         # Data loading and validation
│   ├── hub_features.py        # Feature transforms and encoding
│   ├── classify.py            # Multi-model training with TOPSIS
│   ├── evaluation.py          # Metrics, confusion matrix, comparison
│   └── spatial_analysis.py    # Geospatial analysis and corridors
├── app.py                     # Streamlit dashboard
├── requirements.txt
├── LICENSE
└── README.md
```

## License

BSD 3-Clause License -- see [LICENSE](LICENSE).
