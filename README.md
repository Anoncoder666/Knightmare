# Chess Engine (PyTorch)

Lightweight Python 3.10+ chess engine with legal move generation, alpha-beta search, and a PyTorch evaluation network. The engine plays legal chess, supports castling, promotion, and en passant, and ships with a synthetic training pipeline.

## Installation
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Play vs Engine
```bash
python main.py --play --depth 3 --engine-color black
# Load a trained model:
python main.py --play --load-model models/best_model.pth
```
Enter moves in coordinate form (`e2e4`, `e7e8q` for promotion). Use `quit` to exit.

## Training the Evaluator
Trains on synthetic positions labeled by a material heuristic.
```bash
python training/train.py --epochs 5 --batch-size 64 --dataset-size 2000 --save-path models/best_model.pth
```
Adjust `--device cuda` if a GPU is available.

## Self-Play Data (optional)
Generates JSONL with FEN/value pairs for later training.
```bash
python training/self_play.py --games 10 --depth 2 --output data/selfplay.jsonl
```

## Tests
```bash
python -m unittest discover -s tests
```

## Project Layout
- `main.py` — CLI to play against the engine.
- `chess_engine/` — core engine (board, game state, move generation, search, model, evaluation).
- `training/` — dataset generation, training loop, and self-play helper.
- `tests/` — smoke tests for board, move generation, and search.
