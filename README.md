# Traffic Forecast

Short-horizon traffic demand prediction over a grid of geohash cells. Given a full day of
observed demand plus the first two hours of the next day, predict demand for the rest of
that next day's morning and early afternoon.

The target is a normalised demand value in `[0, 1]`, evaluated with R² (the leaderboard
score is `100 × R²`).

---

## The problem

Demand is recorded per **(geohash cell, 15-minute slot)**. The split is temporal, not random:

| Split | Day | Time window | Rows |
| --- | --- | --- | --- |
| `train.csv` | 48 | 00:00 – 23:45 (full day) | 69,427 |
| `train.csv` | 49 | 00:00 – 02:00 (9 slots) | 7,872 |
| `test.csv` | 49 | 02:15 – 13:45 (47 slots) | 41,778 |

1,249 geohash cells appear in training; 99.9% of test cells were seen before. This shapes
every solution in the repo:

- **Day 48 is a template.** 88.9% of test rows have an exact `(geohash, timestamp)` match
  in day 48, so day-48 demand at the same cell and same time of day is a strong base
  prediction.
- **The day-49 early hours are a calibration signal.** Comparing 00:00–02:00 on day 49
  against the same slots on day 48 gives a per-cell offset (additive) or scale
  (multiplicative) that corrects the template for whatever is different about day 49.

Every model here is some combination of *day-48 lookup* + *day-49 shift correction* +
*gradient boosting on top*.

## Data columns

| Column | Description | Missing |
| --- | --- | --- |
| `Index` | Row id; the submission key | — |
| `geohash` | 6-character geohash (~1.2 km × 0.6 km cell) | — |
| `day` | Day index, 48 or 49 | — |
| `timestamp` | `H:M` start of a 15-minute slot | — |
| `demand` | **Target**, normalised to `[0, 1]`; mean 0.094, median 0.048 | — (train only) |
| `RoadType` | `Residential` / `Street` / `Highway` | 0.8% |
| `NumberofLanes` | Integer, 1–5 | — |
| `LargeVehicles` | `Allowed` / `Not Allowed` | — |
| `Landmarks` | `Yes` / `No` | — |
| `Temperature` | °C, range −14.9 to 48.3 | 3.2% |
| `Weather` | `Sunny` / `Rainy` / `Foggy` / `Snowy` | 1.0% |

The target is heavily right-skewed: the 75th percentile is 0.109 but the maximum is 1.0.
All geohashes share the prefix `qp0` and decode to a single small area, so the coordinates
look anonymised rather than real-world — treat lat/lon as relative position only.

## Repository layout

```
.
├── project.ipynb                            # Main solution: 3-model stack
├── make_93_plus_candidate.py                # LightGBM on day-48 lookup features
├── make_best_800_200.py                     # LGBM + CatBoost ensemble, blended with a base file
├── train.csv                                # Day 48 (full) + day 49 (00:00–02:00)
├── test.csv                                 # Day 49 (02:15–13:45)
├── sample_submission.csv                    # Expected format
├── submission.csv                           # Copy of submission_93plus_lgb_offset.csv
├── submission_93plus_lgb_offset.csv         # Output of make_93_plus_candidate.py
├── submission_crown.csv                     # Output of project.ipynb
├── submission_kapow_killer.csv              # Input to make_best_800_200.py (see Notes)
└── submission_rescue_killer_800_200.csv     # Output of make_best_800_200.py
```

## Setup

```bash
pip install pandas numpy scikit-learn lightgbm xgboost catboost pygeohash
```

**The scripts read from a `dataset/` subfolder, but the CSVs sit at the repository root.**
Before running anything, either move the data or create the folder:

```bash
mkdir -p dataset && cp train.csv test.csv dataset/
```

Then:

```bash
python make_93_plus_candidate.py     # → submission_93plus_lgb_offset.csv
python make_best_800_200.py          # → submission_rescue_killer_800_200.csv
jupyter notebook project.ipynb       # → submission_crown.csv
```

## Approaches

### 1. `project.ipynb` — stacked ensemble (best reported score)

The strongest single artefact. Builds spatial and cyclical features, then stacks three
gradient boosters.

- **Spatial:** lat/lon decoded from the geohash, plus 4- and 5-character prefixes, plus 60
  KMeans clusters over lat/lon.
- **Temporal:** 15-minute slot index, sine/cosine encodings of slot and hour, day of week
  (`day % 7`), weekend flag, morning (07–09) and evening (17–19) peak flags.
- **Composite keys:** `geohash_timeslot` and `geohash_dayofweek_timeslot` — the granular
  "traffic footprint" of a cell at a moment.
- **Target encoding:** out-of-fold (5-fold) mean encoding of `geohash`, spatial cluster,
  and both composite keys, with the global mean as fallback.
- **Level 1:** LightGBM, XGBoost, and CatBoost, each 5-fold with early stopping,
  categoricals passed natively. No log transform on the target — the peaks are the point.
- **Level 2:** Ridge meta-learner over the three out-of-fold prediction vectors. Learned
  weights came out roughly LGB 0.27 / XGB 0.49 / CatBoost 0.25.

Reported out-of-fold R² of **0.9541**. Note this is measured with random K-fold on the
combined training data, while the real task is a forward-in-time prediction — with target
encodings this granular, the OOF figure is optimistic relative to the leaderboard.

### 2. `make_93_plus_candidate.py` — lookup cascade + LightGBM

Leans directly on the day-48 template. For each row it builds a cascade of day-48 lookups
from most to least specific — exact `(geohash, timestamp)`, then `(geohash, time-of-day)`,
then `(geohash, hour)`, then geohash mean, then timestamp mean — and takes the first hit as
a `base` value.

It then computes a per-geohash additive offset and multiplicative scale from the day-49
early hours, and feeds *analytical predictions* (`base + offset`, `base × scale`) into
LightGBM as features, effectively telling the model what the answer should be before it
starts. The final output is `0.7 × LightGBM + 0.3 × (base + offset)`.

### 3. `make_best_800_200.py` — smoothed calibration ensemble

The most carefully regularised of the three, and the only one with a validation scheme that
respects time order.

- **Hierarchical fallbacks:** day-48 group means at `geohash`, `geo5`, `geo4`, `geo3`,
  weather, and road-type levels, cascading until a value is found.
- **Shrunk calibration:** per-cell delta and scale from the day-49 early hours, smoothed
  toward the global value with a prior of 6 observations and clipped (delta to
  `[-0.12, 0.20]`, scale to `[0.35, 3.0]`) so thin cells can't produce wild corrections.
- **Models:** 5-seed LightGBM average + CatBoost with native categoricals + a hand-weighted
  formula prediction, blended 0.45 / 0.40 / 0.15.
- **Forward validation:** trains on day-49 slots 0–5 and validates on slots 6–8, printing
  R² and RMSE per component. This is the honest estimate in the repo.
- **Final blend:** `0.80 × submission_kapow_killer.csv + 0.20 × the above` — a conservative
  nudge to an existing strong submission rather than a replacement.

## Submissions

Format is two columns, `Index` and `demand`, one row per test row (41,778), in the same
Index order as `test.csv`.

| File | Produced by | Mean demand |
| --- | --- | --- |
| `submission_crown.csv` | `project.ipynb` | 0.1283 |
| `submission_93plus_lgb_offset.csv` | `make_93_plus_candidate.py` | 0.1221 |
| `submission_rescue_killer_800_200.csv` | `make_best_800_200.py` | 0.1293 |
| `submission_kapow_killer.csv` | - | 0.1312 |
| `submission.csv` | byte-identical to `submission_93plus_lgb_offset.csv` | 0.1221 |

All five are highly correlated (pairwise r ≥ 0.977), which is expected — they share the same
day-48 lookup backbone and differ mainly in how the day-49 shift is estimated and smoothed.

## Notes and known gaps

- **`submission_kapow_killer.csv` has no generating script.** `make_best_800_200.py` reads
  it as a required input and will fail without it. If you want that pipeline reproducible
  end to end, the script that produced it needs to be added.
- **Hard-coded paths.** All three entry points expect `dataset/train.csv` and
  `dataset/test.csv`. There is no CLI or config.
- **No `requirements.txt`.** Versions are unpinned; `pygeohash`'s `decode` API has changed
  across major versions, so pin it if you hit an error there.
- **`day_of_week` is near-constant.** The data only covers days 48 and 49, so `day % 7` and
  the weekend flag take at most two values and carry little signal — the day-49 rows also
  only cover 00:00–02:00, so the notebook's `geo_dow_time_key` mostly reduces to
  `geo_time_key` on day 48.
- **Prediction clipping is inconsistent** across scripts (`[0, 1]` in one, `[0, 1.1]` in
  another, unclipped upper bound in the notebook). Since the observed target maxes out at
  exactly 1.0, clipping to `[0, 1]` everywhere is likely the safer default.
