.PHONY: install-cuda install-mlx test lint inspect
install-cuda: ; pip install -e ".[cuda,dev]"
install-mlx:  ; pip install -e ".[mlx,dev]"
test:         ; pytest -q
lint:         ; ruff check aloe tests
