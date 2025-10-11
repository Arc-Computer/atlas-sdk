SHELL := /bin/bash
PYTHON ?= python
COMPOSE := docker compose -f docker/docker-compose.yaml

.PHONY: demo-db demo demo-driver demo-stream demo-clean

demo-db:
	$(COMPOSE) up -d postgres

demo:
	$(COMPOSE) up -d postgres
	@echo "Launching streaming driver (ctrl+c to stop)…"
	@bash -lc 'set -a; [ -f .env ] && source .env; set +a; $(PYTHON) -m examples.streaming_sre_demo.run_streaming_demo --speed slow'

demo-driver:
	@bash -lc 'set -a; [ -f .env ] && source .env; set +a; $(PYTHON) -m examples.streaming_sre_demo.run_streaming_demo --speed slow'

demo-stream:
	@bash -lc 'set -a; [ -f .env ] && source .env; set +a; $(PYTHON) -m examples.streaming_sre_demo.data_stream --speed slow'

demo-clean:
	$(COMPOSE) down -v

