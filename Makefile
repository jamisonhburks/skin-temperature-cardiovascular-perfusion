# Convenience targets for the skin-temp-dynamics pipeline.
# All stages read --data-dir (default data/synthetic) and write to data/processed.

PYTHON ?= python
N_PARTICIPANTS ?= 600
N_NIGHTS ?= 14
SEED ?= 7

.PHONY: help install data global perturb linear mechanistic cohort figures demo test clean

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

install: ## Editable install with dev + transfer-entropy extras
	$(PYTHON) -m pip install -e ".[te,dev]"

data: ## Generate the synthetic dataset
	$(PYTHON) scripts/00_generate_synthetic_data.py --n-participants $(N_PARTICIPANTS) --n-nights $(N_NIGHTS) --seed $(SEED)

global: ## Stage 1 — Pearson / lead-lag / transfer entropy (Fig. 2)
	$(PYTHON) scripts/01_global_relationships.py

perturb: ## Stage 2 — extract isolated-perturbation trajectories
	$(PYTHON) scripts/02_extract_perturbations.py

linear: ## Stage 3 — fit the 1/2/3-variable OLS models (Fig. 3)
	$(PYTHON) scripts/03_fit_linear_models.py

mechanistic: ## Stage 4 — fit the mechanistic model (Figs. 4–5)
	$(PYTHON) scripts/04_fit_mechanistic.py

cohort: ## Stage 5 — cohort effect sizes + Table 1
	$(PYTHON) scripts/05_cohort_analysis.py

figures: ## Render paper-analogous figures to results/figures
	$(PYTHON) scripts/make_figures.py

demo: data perturb linear mechanistic cohort figures ## Full pipeline (skips slow TE stage)
	@echo "Done. See results/figures/ and data/processed/."

test: ## Run the test suite
	pytest

clean: ## Remove generated data, artifacts, and figures
	rm -f data/synthetic/*.parquet data/processed/* results/figures/*
