.DEFAULT_GOAL := help
.PHONY: help validate-dataset-ids preview-dataset-site

DATASET ?= src/ua_tts_bench/data/dataset.jsonl
SITE_DIR := _site
PORT ?= 8000

help: ## Show available commands
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-22s\033[0m %s\n", $$1, $$2}'

validate-dataset-ids: ## Check that every id equals the md5 hash of its text
	@python3 scripts/validate_dataset_ids.py $(DATASET)

preview-dataset-site: ## Build the dataset site into _site/ and serve it on the first free port from 8000
	@rm -rf $(SITE_DIR) && mkdir -p $(SITE_DIR)
	@cp -r site/. $(SITE_DIR)/
	@cp $(DATASET) $(SITE_DIR)/dataset.jsonl
	@python3 scripts/serve_site.py $(SITE_DIR) --port $(PORT)
