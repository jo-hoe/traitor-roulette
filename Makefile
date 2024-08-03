include help.mk

.DEFAULT_GOAL := start

ROOT_DIR := $(dir $(realpath $(lastword $(MAKEFILE_LIST))))

.PHONY: git-pull
git-pull:
	@git pull

.PHONY: venv
venv:
	@python -m venv ${ROOT_DIR}.venv

.PHONY: update
update: git-pull ## pulls git repo and installs all dependencies
	${ROOT_DIR}.venv/Scripts/pip install -r ${ROOT_DIR}requirements.txt

.PHONY: setup-python
setup-python: venv update ## init setup of project after checkout

.PHONY: save-dependencies
save-dependencies: ## save current dependencies
	${ROOT_DIR}.venv/Scripts/pip freeze > ${ROOT_DIR}requirements.txt

.PHONY: test
test: ## runs all tests
	@${ROOT_DIR}.venv/Scripts/pytest ${ROOT_DIR}test/

.PHONY: play
play: ## play a game of traitors roulette
	${ROOT_DIR}.venv/Scripts/python ${ROOT_DIR}justplay.py

.PHONY: bruteforce-simulation
bruteforce-simulation: ## bruteforces best static percentage strategy
	${ROOT_DIR}.venv/Scripts/python ${ROOT_DIR}bruteforce.py

.PHONY: ml-play
ml-play: ## train a model for traitors roulette of non exists and show results
	${ROOT_DIR}.venv/Scripts/python ${ROOT_DIR}machine_learning.py

.PHONY: jupyter
jupyter: ## starts a jupyter kernel in docker container
	docker run --rm -p 8888:8888 \
		-v ${ROOT_DIR}src:/home/jovyan/src \
		-v ${ROOT_DIR}output:/home/jovyan/output \
		quay.io/jupyter/pytorch-notebook:latest \
		start-notebook.py --IdentityProvider.token=''