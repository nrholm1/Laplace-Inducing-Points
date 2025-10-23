.SHELLFLAGS := -eu -o pipefail -c
SHELL       := bash

.PHONY: help venv requirements clean \
        run run-toy-dense run-toy-mf run-scale \
        train_map train_inducing visualize visualize_full \
        train_map_scale train_inducing_scale \
        debug_run debug_map debug_inducing debug_visualize \
        data all-data

# Globals / paths
PROJECT_NAME   ?= laplace_inducing_points
PYTHON         ?= python
MAIN           ?= main.py

# Default knobs
MODE           ?= full_pipeline      # train_map | train_inducing | visualize | full_pipeline
EXTRA          ?=                    # extra CLI args appended at the end

# Toy presets (small scale)
TOY_DATASET   ?= banana
TOY_MODEL     ?= toyclassifier
CONFIG_TOY    := config/toy/$(TOY_MODEL)_$(TOY_DATASET).yml

# Scale presets (large scale)
SCALE_DATASET ?= cifar10
SCALE_MODEL   ?= resnet1
CONFIG_SCALE  := config/scale/$(SCALE_MODEL)_$(SCALE_DATASET).yml

# Runners
run:
	@if [ -z "$(VARIANT)" ]; then echo "ERROR: VARIANT is required (toy-dense | toy-mf | scale)"; exit 1; fi
	@if [ -z "$(DATASET)" ]; then echo "ERROR: DATASET is required"; exit 1; fi
	@if [ -z "$(CONFIG)" ]; then echo "ERROR: CONFIG is required"; exit 1; fi
	$(PYTHON) $(MAIN) $(MODE) \
		--variant "$(VARIANT)" \
		--dataset "$(DATASET)" \
		--config "$(CONFIG)" \
		$(EXTRA)

# Convenience wrappers that auto-select CONFIG/DATASET/VARIANT
run-toy-dense:
	@$(MAKE) run VARIANT=toy-dense DATASET="$(TOY_DATASET)" CONFIG="$(CONFIG_TOY)" MODE="$(MODE)" EXTRA='$(EXTRA)'

run-toy-mf:
	@$(MAKE) run VARIANT=toy-mf DATASET="$(TOY_DATASET)" CONFIG="$(CONFIG_TOY)" MODE="$(MODE)" EXTRA='$(EXTRA)'

run-scale:
	@$(MAKE) run VARIANT=scale DATASET="$(SCALE_DATASET)" CONFIG="$(CONFIG_SCALE)" MODE="$(MODE)" EXTRA='$(EXTRA)'

# Toy shortcuts
train_map:
	@$(MAKE) run-toy-dense MODE=train_map

train_inducing:
	@$(MAKE) run-toy-dense MODE=train_inducing EXTRA='--alpha_ip 1 $(EXTRA)'

# matrix-free toy (uses toy-mf variant)
strain_inducing:
	@$(MAKE) run-toy-mf MODE=train_inducing EXTRA='--alpha_ip 1 $(EXTRA)'

visualize:
	@$(MAKE) run-toy-dense MODE=visualize EXTRA='--alpha_ip 1 $(EXTRA)'

visualize_full:
	@$(MAKE) run-toy-dense MODE=visualize EXTRA='--full --alpha_ip 1 $(EXTRA)'

# matrix-free visualize with MC samples (env var mcs or default 1024)
svisualize:
	@$(MAKE) run-toy-mf MODE=visualize EXTRA='--alpha_ip 1 $(EXTRA)'


# Scale shortcuts
train_map_scale:
	@$(MAKE) run-scale MODE=train_map

train_inducing_scale:
	@$(MAKE) run-scale MODE=train_inducing


# Debug (debugpy)
debug_run:
	@if [ -z "$(VARIANT)" ]; then echo "ERROR: VARIANT is required (toy-dense | toy-mf | scale)"; exit 1; fi
	@if [ -z "$(DATASET)" ]; then echo "ERROR: DATASET is required"; exit 1; fi
	@if [ -z "$(CONFIG)" ]; then echo "ERROR: CONFIG is required"; exit 1; fi
	nohup $(PYTHON) -m debugpy --listen 5678 --wait-for-client $(MAIN) $(MODE) \
		--variant $(VARIANT) \
		--dataset $(DATASET) \
		--config $(CONFIG) \
		$(EXTRA) > debug.log 2>&1 & \
	sleep 1 ; echo "debugpy ready (port 5678)"

debug_map:
	@$(MAKE) debug_run VARIANT=toy-dense DATASET=$(TOY_DATASET) CONFIG=$(CONFIG_TOY) MODE=train_map

debug_inducing:
	@$(MAKE) debug_run VARIANT=toy-mf DATASET=$(TOY_DATASET) CONFIG=$(CONFIG_TOY) MODE=train_inducing EXTRA='--alpha_ip 1 $(EXTRA)'

debug_visualize:
	@$(MAKE) debug_run VARIANT=toy-mf DATASET=$(TOY_DATASET) CONFIG=$(CONFIG_TOY) MODE=visualize




# Toy data generation helpers
DATA_MAIN ?= src/toydata.py
D         ?= spiral
N         ?= 1000
EPS       ?= 0.08
SEED      ?= 1234
ARGS      ?=

data:
	$(PYTHON) $(DATA_MAIN) --dataset $(D) --n_samples $(N) --noise $(EPS) --seed $(SEED) $(ARGS)

# Presets
N1=300  ; EPS1=0.7    ; SEED1=1526  ; ARGS1=--split_in_middle
N2=1280 ; EPS2=0.25   ; SEED2=6251  ; ARGS2=
N3=500  ; EPS3=0.090  ; SEED3=584848; ARGS3=

all-data:
	$(PYTHON) $(DATA_MAIN) --dataset sine   --n_samples $(N1) --noise $(EPS1) --seed $(SEED1) $(ARGS1)
	$(PYTHON) $(DATA_MAIN) --dataset xor    --n_samples $(N2) --noise $(EPS2) --seed $(SEED2) $(ARGS2)
	$(PYTHON) $(DATA_MAIN) --dataset banana --n_samples $(N3) --noise $(EPS3) --seed $(SEED3) $(ARGS3)

# -----------------------------
# Help
# -----------------------------
.DEFAULT_GOAL := help

help:
	@echo ""
	@echo "Targets:"
	@echo "  run               (generic)          VARIANT=toy-dense|toy-mf|scale DATASET=... CONFIG=..."
	@echo "  run-toy-dense     | run-toy-mf       (toy presets)  MODE=<...> [EXTRA='...']"
	@echo "  run-scale                           (scale preset) MODE=<...> [EXTRA='...']"
	@echo ""
	@echo "  train_map | train_inducing | strain_inducing"
	@echo "  visualize | visualize_full | svisualize | svisualize_full"
	@echo "  train_map_scale | train_inducing_scale"
	@echo ""
	@echo "  debug_map | debug_inducing | debug_visualize"
	@echo "  data | all-data"
	@echo ""
	@echo "Variables (override with VAR=value):"
	@printf "  %-14s %s\n" "TOY_DATASET"   "$(TOY_DATASET)"
	@printf "  %-14s %s\n" "TOY_MODEL"     "$(TOY_MODEL)"
	@printf "  %-14s %s\n" "CONFIG_TOY"    "$(CONFIG_TOY)"
	@printf "  %-14s %s\n" "SCALE_DATASET" "$(SCALE_DATASET)"
	@printf "  %-14s %s\n" "SCALE_MODEL"   "$(SCALE_MODEL)"
	@printf "  %-14s %s\n" "CONFIG_SCALE"  "$(CONFIG_SCALE)"
	@echo ""
	@echo "Examples:"
	@echo "  make run-toy-mf MODE=train_inducing EXTRA='--alpha_ip 1'"
	@echo "  make run-scale MODE=train_map"
	@echo "  make visualize_full"
