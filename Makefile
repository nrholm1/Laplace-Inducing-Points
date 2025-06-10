
.PHONY: requirements dev_requirements clean data build_documentation serve_documentation

#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = laplace_inducing_points
PYTHON_VERSION = 3.13
PYTHON_INTERPRETER = python

#################################################################################
# COMMANDS                                                                      #
#################################################################################

## Set up python interpreter environment
create_venv:
	$(PYTHON_INTERPRETER) -m venv venv

## Install Python Dependencies
requirements:
	$(PYTHON_INTERPRETER) -m pip install -U pip setuptools wheel
	$(PYTHON_INTERPRETER) -m pip install -r requirements.txt
	$(PYTHON_INTERPRETER) -m pip install -e .

## Delete all compiled Python files
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete


#################################################################################
# PROJECT RULES                                                                 #
#################################################################################

main = main.py
scaletrain = scale_experiments/train.py
eval = scale_experiments/evaluate.py
datamain = src/toydata.py

DEVICE = cpu

DATASET = spiral
MODEL = toyclassifier
SCALE_DATASET = cifar10
SCALE_MODEL = resnet1


run:
	$(PYTHON_INTERPRETER) $(main) $(mode) \
		--dataset $(DATASET) \
		--model_config config/toy/$(MODEL)_$(DATASET).yml \
		--optimization_config config/toy/optimization_$(MODEL)_$(DATASET).yml \
		$(EXTRA_ARGS)

debug_run:
	nohup $(PYTHON_INTERPRETER) -m debugpy --listen 5678 --wait-for-client $(main) $(mode) \
		--dataset $(DATASET) \
		--model_config config/toy/$(MODEL)_$(DATASET).yml \
		--optimization_config config/toy/optimization_$(MODEL)_$(DATASET).yml > debug.log 2>&1 &
	sleep 1
	@echo "debugpy ready"

train_scale:
	$(PYTHON_INTERPRETER) $(scaletrain) $(mode) \
		--dataset $(SCALE_DATASET) \
		--config config/scale/$(SCALE_MODEL)_$(SCALE_DATASET).yml \
		$(EXTRA_ARGS)
eval_scale:
	$(PYTHON_INTERPRETER) $(eval) $(mode) \
		--dataset $(SCALE_DATASET) \
		--config config/scale/$(SCALE_MODEL)_$(SCALE_DATASET).yml \
		--scalable $(EXTRA_ARGS)
eval:
	$(PYTHON_INTERPRETER) $(eval) $(mode) \
		--dataset $(DATASET) \
		--config config/toy/$(MODEL)_$(DATASET).yml \
		$(EXTRA_ARGS)
seval:
	$(PYTHON_INTERPRETER) $(eval) $(mode) \
		--dataset $(DATASET) \
		--config config/toy/$(MODEL)_$(DATASET).yml \
		--scalable $(EXTRA_ARGS)

# run targets
train_map:
	$(MAKE) run mode=train_map
train_inducing:
	$(MAKE) run mode=train_inducing
strain_inducing:
	$(MAKE) run mode=train_inducing EXTRA_ARGS=--scalable
full_pipeline:
	$(MAKE) run mode=full_pipeline
sfull_pipeline:
	$(MAKE) run mode=full_pipeline EXTRA_ARGS=--scalable
visualize:
	$(MAKE) run mode=visualize
visualize_full:
	$(MAKE) run mode=visualize EXTRA_ARGS=--full
svisualize:
	$(MAKE) run mode=visualize EXTRA_ARGS="--scalable --num_mc_samples_lla $(mcs)"
svisualize_full:
	$(MAKE) run mode=visualize EXTRA_ARGS="--full --scalable --num_mc_samples_lla $(mcs)"

train_map_scale:
	$(MAKE) train_scale mode=train_map
train_ip_scale:
	$(MAKE) train_scale mode=train_inducing 
#EXTRA_ARGS="--alpha_ip 10000"


# debug targets
debug_map:
	$(MAKE) debug_run mode=train_map
debug_inducing:
	$(MAKE) debug_run mode=train_inducing
debug_visualize:
	$(MAKE) debug_run mode=visualize

data:
	$(PYTHON_INTERPRETER) $(datamain) --dataset $(D) --n_samples $(N) --noise $(EPS) --seed $(SEED) $(ARGS)


N1 = 300
N2 = 1280
N3 = 500
EPS1 = 0.7
EPS2 = 0.25
EPS3 = 0.090
SEED1 = 1526
SEED2 = 6251
SEED3 = 584848
ARGS1="--split_in_middle"
all-data:
	$(PYTHON_INTERPRETER) $(datamain) --dataset sine --n_samples $(N1) --noise $(EPS1) --seed $(SEED1) $(ARGS1)
	$(PYTHON_INTERPRETER) $(datamain) --dataset xor --n_samples $(N2) --noise $(EPS2) --seed $(SEED2) $(ARGS2)
	$(PYTHON_INTERPRETER) $(datamain) --dataset banana --n_samples $(N3) --noise $(EPS3) --seed $(SEED3) $(ARGS3)

# make data D=spiral N=1000 EPS=0.08 SEED=1234