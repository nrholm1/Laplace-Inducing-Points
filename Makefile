
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

main = src/main.py
datamain = src/toydata.py

DEVICE = cpu

DATASET = sine
MODEL = toyregressor

run:
	$(PYTHON_INTERPRETER) src/main.py $(mode) \
		--dataset $(DATASET) \
		--model_config config/$(MODEL).yml \
		--optimization_config config/optimization.yml


train_map:
	$(MAKE) run mode=train_map
train_inducing:
	$(MAKE) run mode=train_inducing
full_pipeline:
	$(MAKE) run mode=full_pipeline
visualize:
	$(MAKE) run mode=visualize

data:
	$(PYTHON_INTERPRETER) $(datamain) --dataset $(D) --n_samples $(N) --noise $(EPS) --seed $(SEED) $(ARGS)


N1 = 256
N2 = 256
EPS1 = 0.3
EPS2 = 0.3
SEED1 = 1526
SEED2 = 6251
all-data:
	$(PYTHON_INTERPRETER) $(datamain) --dataset xor --n_samples $(N1) --noise $(EPS1) --seed $(SEED1) $(ARGS1)
	$(PYTHON_INTERPRETER) $(datamain) --dataset sine --n_samples $(N2) --noise $(EPS2) --seed $(SEED2) $(ARGS2)