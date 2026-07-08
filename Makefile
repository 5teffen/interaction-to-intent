# ── From Interaction to Intent — reproduction pipeline ───────────────────────
# Reproduces the results of the paper from the published Zenodo dataset.
#
# Quick start:
#   make data       # download the dataset from Zenodo into data/processed/
#   make all        # reproduce every table and figure
#
# Or reproduce a single paper artifact:
#   make table1     # Table 1 — atomic + online multi-task classification
#   make table2     # Table 2 — feature ablations
#   make figures    # all figures  (fig-confusion, fig-dataset, fig-composition)
#
# Knobs (defaults reproduce the reported numbers):
#   SCHEME = fine|coarse     7-class tasks (fine) or 3-class groups (coarse)
#   NORM   = per-source|global   feature standardization (per-source is reported)
#   SEEDS  = 42[,43,...]      seeds for ablations / multi-seed runs

SHELL  := /bin/bash
PYTHON := python

SCHEME ?= fine
NORM   ?= per-source
SEEDS  ?= 42

# ── Paths ────────────────────────────────────────────────────────────────────
DATA_DIR  := data
PROJ_DIR  := $(DATA_DIR)/projected
PROC_DIR  := $(DATA_DIR)/processed
CKPT_DIR  := checkpoints
FIG_DIR   := figures
RES_DIR   := results

TRAIN_CSV := $(PROC_DIR)/train_filtered.csv
TEST_CSV  := $(PROC_DIR)/test_filtered.csv

# Zenodo dataset (concept DOI resolves to the latest version).
ZENODO_DOI ?= 10.5281/zenodo.21207995
ZENODO_ID   = $(lastword $(subst ., ,$(ZENODO_DOI)))

# Stage 2 hyperparameters (defaults reproduce the reported numbers).
S2_DROPOUT ?= 0.3
S2_WD      ?= 1e-4
S2_DECAY   ?= 1.0

.PHONY: all data table1 table2 figures \
        fig-confusion fig-dataset fig-composition \
        features sequences evaluate \
        _stage1 _stage2 clean help

# ── One-shot ─────────────────────────────────────────────────────────────────
all: table1 table2 figures

help:
	@sed -n '1,20p' Makefile

# ── Step 0: dataset ──────────────────────────────────────────────────────────
# The full dataset — interaction sequences AND projection coordinates — comes
# from Zenodo; nothing is vendored in this repo. Files are located by name so
# the target is robust to the archive's internal folder layout.
data: $(TRAIN_CSV)
$(TRAIN_CSV):
	@mkdir -p $(PROC_DIR) $(PROJ_DIR)
	@echo "Downloading dataset from Zenodo ($(ZENODO_DOI))..."
	curl -L "https://zenodo.org/api/records/$(ZENODO_ID)/files-archive" -o /tmp/h2i-data.zip
	unzip -o /tmp/h2i-data.zip -d /tmp/h2i-data
	cp "$$(find /tmp/h2i-data -name train.csv | head -1)" $(TRAIN_CSV)
	cp "$$(find /tmp/h2i-data -name test.csv  | head -1)" $(TEST_CSV)
	find /tmp/h2i-data -name '*_projected_data_*.csv' -exec cp {} $(PROJ_DIR)/ \;
	rm -rf /tmp/h2i-data /tmp/h2i-data.zip
	@echo "Dataset ready: $(PROC_DIR)/ (sequences) and $(PROJ_DIR)/ (projections)"

# ── Shared preprocessing ─────────────────────────────────────────────────────
features: $(PROC_DIR)/train_temporal.pkl
$(PROC_DIR)/train_temporal.pkl: $(TRAIN_CSV)
	$(PYTHON) scripts/prepare_features.py \
		--train-csv $(TRAIN_CSV) --test-csv $(TEST_CSV) \
		--proj-dir $(PROJ_DIR) --output-dir $(PROC_DIR)

sequences: $(PROC_DIR)/stage2_varB_atomics.pkl
$(PROC_DIR)/stage2_varB_atomics.pkl: $(TRAIN_CSV)
	$(PYTHON) scripts/prepare_sequences.py \
		--data-csv $(TRAIN_CSV) --test-data-csv $(TEST_CSV) \
		--proj-dir $(PROJ_DIR) --output-dir $(PROC_DIR) \
		--n-sessions 2000 --seed 42

# ── Table 1: atomic + online multi-task classification ───────────────────────
# Trains both Stage 1 models (XGBoost summary, BiGRU temporal) and both Stage 2
# variants, at both label granularities (7-class fine and 3-class coarse), then
# evaluates on the held-out recipes source.
table1: features sequences
	$(MAKE) _stage1 SCHEME=fine
	$(MAKE) _stage1 SCHEME=coarse
	$(MAKE) _stage2 SCHEME=fine
	$(MAKE) _stage2 SCHEME=coarse
	$(MAKE) evaluate

# Stage 1: LOSO CV + full-train + held-out test, one label scheme.
_stage1:
	@mkdir -p $(CKPT_DIR)
	$(PYTHON) scripts/train_stage1_xgboost.py \
		--data $(PROC_DIR)/train_summary.pkl --test-data $(PROC_DIR)/test_summary.pkl \
		--mode both --save-model $(CKPT_DIR)/ --seed 42 \
		--label-scheme $(SCHEME) --normalize $(NORM)
	$(PYTHON) scripts/train_stage1_bigru.py \
		--data $(PROC_DIR)/train_temporal.pkl --test-data $(PROC_DIR)/test_temporal.pkl \
		--mode both --save-model $(CKPT_DIR)/ --seed 42 \
		--label-scheme $(SCHEME) --normalize $(NORM)

# Stage 2: UniGRU-CRF, Variant A (log-level) + Variant B (feature-level).
_stage2:
	@mkdir -p $(CKPT_DIR)
	$(PYTHON) scripts/train_stage2_unigru.py --variant A --data-dir $(PROC_DIR) \
		--test-data $(PROC_DIR)/test_stage2_variantA.pkl --mode both \
		--save-model $(CKPT_DIR)/ --seed 42 --label-scheme $(SCHEME) --normalize $(NORM) \
		--dropout $(S2_DROPOUT) --weight-decay $(S2_WD) --decay $(S2_DECAY)
	$(PYTHON) scripts/train_stage2_unigru.py --variant B --data-dir $(PROC_DIR) \
		--test-data $(PROC_DIR)/test_stage2_variantB.pkl --mode both \
		--save-model $(CKPT_DIR)/ --seed 42 --label-scheme $(SCHEME) --normalize $(NORM) \
		--dropout $(S2_DROPOUT) --weight-decay $(S2_WD) --decay $(S2_DECAY)

evaluate:
	@mkdir -p $(FIG_DIR)
	$(PYTHON) scripts/evaluate.py --checkpoint-dir $(CKPT_DIR) --test-dir $(PROC_DIR) \
		--output $(PROC_DIR)/results.json --figures-dir $(FIG_DIR)

# ── Table 2: feature ablations ───────────────────────────────────────────────
# XGBoost feature-group leave-one-out + BiGRU temporal ablation (incl. shuffled
# timestep order, hypothesis H4), at both granularities.
table2: features
	@mkdir -p $(RES_DIR)
	$(MAKE) _ablate SCHEME=fine
	$(MAKE) _ablate SCHEME=coarse

_ablate:
	$(PYTHON) scripts/run_ablations.py --model xgboost --mode ablation \
		--label-scheme $(SCHEME) --normalize $(NORM) --seeds $(SEEDS) \
		--out $(RES_DIR)/ablate_xgboost_$(SCHEME)_$(NORM).md
	$(PYTHON) scripts/run_ablations.py --model bigru --mode all \
		--label-scheme $(SCHEME) --normalize $(NORM) --seeds $(SEEDS) \
		--out $(RES_DIR)/ablate_bigru_$(SCHEME)_$(NORM).md

# ── Figures ──────────────────────────────────────────────────────────────────
figures: fig-confusion fig-dataset fig-composition

# Per-model confusion matrices (needs Stage 1/2 checkpoints — run `make table1` first).
fig-confusion:
	@mkdir -p $(FIG_DIR)
	$(PYTHON) scripts/plot_confusion.py --out $(FIG_DIR)/confusion_matrices

# Dataset composition summary (task / dataset / projection counts, sequence lengths).
fig-dataset: $(TRAIN_CSV)
	@mkdir -p $(FIG_DIR)
	cat $(TRAIN_CSV) <(tail -n +2 $(TEST_CSV)) > $(PROC_DIR)/all_filtered.csv
	$(PYTHON) scripts/plot_dataset_summary.py --csv $(PROC_DIR)/all_filtered.csv \
		--out $(FIG_DIR)/dataset_summary

# Composed multi-task session statistics (needs `make sequences`).
fig-composition: sequences
	@mkdir -p $(FIG_DIR)
	$(PYTHON) scripts/plot_session_composition.py --data-dir $(PROC_DIR) \
		--out $(FIG_DIR)/session_composition

# ── Cleanup ──────────────────────────────────────────────────────────────────
clean:
	rm -f $(PROC_DIR)/*.pkl $(PROC_DIR)/results.json $(PROC_DIR)/all_filtered.csv
	rm -rf $(CKPT_DIR)/* $(FIG_DIR)/*
