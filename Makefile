IMAGE := openai-emulator
PORT  := 8000

.PHONY: install dev build run stop

install:
	uv sync

dev:
	uv run uvicorn service:app --reload --port $(PORT)

build:
	docker build -t $(IMAGE) .

run:
	docker run --rm -p $(PORT):8000 $(IMAGE)

stop:
	docker stop $$(docker ps -q --filter ancestor=$(IMAGE))
