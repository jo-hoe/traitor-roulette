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
	@${ROOT_DIR}.venv/Scripts/python ${ROOT_DIR}justplay.py

.PHONY: bruteforce
bruteforce: ## bruteforces best static strategy
	@${ROOT_DIR}.venv/Scripts/python ${ROOT_DIR}bruteforce.py
