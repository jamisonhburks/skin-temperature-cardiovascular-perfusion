# `data/raw/` — real wearable data (not included)

This directory is the drop-in location for **real** per-participant wearable
exports. It is intentionally empty and git-ignored: the TemPredict / Oura Ring
data used in the paper are **not redistributable** (see the manuscript's Data
Availability statement — Oura's data-use policy prohibits sharing device data
with third parties). Self-report data can be requested from the study authors.

**Do not commit participant data here.** The `.gitignore` blocks it by default.

## Expected schema

To run the pipeline on real data, place one **Parquet** file per participant,
named `<participant_id>.parquet`, with minute-resolution rows and these columns:

| column      | dtype   | description                                              |
|-------------|---------|----------------------------------------------------------|
| `temp_skin` | float   | distal skin temperature, °C                              |
| `met`       | float   | metabolic equivalents (activity); device baseline ≈ 0.9  |
| `is_awake`  | bool    | sleep/wake label (Oura's algorithm)                      |
| `HR`        | float   | heart rate (optional; unused by the core models)         |

Plus a demographics table `demographics.parquet` (or `.csv`) indexed by
participant id with at least: `age` (years), `sex` (`male`/`female`), and the
condition flags `cd_none`, `cd_hypertension`, `cd_diabetes`, `cd_coronary`,
`cd_afib` (1 = reported). See `skin_temp_perfusion.config.CONDITION_COLUMNS`.

Then point any stage script at it, e.g.:

```bash
python scripts/02_extract_perturbations.py --data-dir data/raw
```

The schema is validated on load (`skin_temp_perfusion.io.validate_timeseries`).
