include help.mk

.DEFAULT_GOAL := start

ROOT_DIR := $(dir $(realpath $(lastword $(MAKEFILE_LIST))))

.PHONY: init
init: venv update

.PHONY: git-pull
git-pull:
	@git pull

.PHONY: venv
venv:
	@python -m venv ${ROOT_DIR}.venv

.PHONY: update
update: git-pull ## pulls git repo and installs all dependencies
	${ROOT_DIR}.venv/Scripts/python -m pip install -r ${ROOT_DIR}requirements.txt

.PHONY: setup-python
setup-python: venv update ## init setup of project after checkout

.PHONY: save-dependencies
save-dependencies:
	"${ROOT_DIR}.venv/Scripts/pip" list --not-required --format=freeze | grep -v "pip" > ${ROOT_DIR}requirements.txt

.PHONY: test
test: ## runs all tests
	@${ROOT_DIR}.venv/Scripts/pytest ${ROOT_DIR}test/

.PHONY: play
play: ## play a game of traitors roulette
	${ROOT_DIR}.venv/Scripts/python ${ROOT_DIR}justplay.py

.PHONY: bruteforce-simulation
bruteforce-simulation: ## bruteforces best static percentage strategy
	${ROOT_DIR}.venv/Scripts/python ${ROOT_DIR}bruteforce.py

.PHONY: ml-train
ml-train: ## train a model for traitors roulette of non exists and show results
	${ROOT_DIR}.venv/Scripts/python ${ROOT_DIR}machine_learning_training.py

.PHONY: ml-evaluate
ml-evaluate: ## evaluate the ml model provide the model name as such `make ml-evaluate name=<your_model_name>`
	${ROOT_DIR}.venv/Scripts/python ${ROOT_DIR}machine_learning_evaluate.py --model-name "$(name)"
