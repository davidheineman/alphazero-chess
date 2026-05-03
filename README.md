# AlphaZero Chess

Learns chess from scratch via self-play and MCTS.

## Local setup

```bash
uvinit
uva
uv sync
```

## Train locally

```bash
python main.py
```

Checkpoints save to `checkpoints/`. Tune via CLI flags or edit `alphazero/config.py`.

```bash
python main.py --res-blocks 5 --channels 64 --simulations 100 --games 25 --iterations 50
```

## Play against it

```bash
python -m alphazero.play checkpoints/best.pt white
```

Moves are in UCI format (e.g. `e2e4`). Pass `black` to play as black.

## Run on RunPod

Build and push the Docker image:

```bash
docker build -t alphazero-chess .
docker tag alphazero-chess your-dockerhub-user/alphazero-chess
docker push your-dockerhub-user/alphazero-chess
```

Then on [runpod.io](https://runpod.io):

1. Create a **GPU Pod** (A40 or better)
2. Set docker image to `your-dockerhub-user/alphazero-chess`
3. Mount a volume at `/output` to persist checkpoints
4. Start the pod — training begins automatically

Override defaults by setting the container command:

```
python main.py --simulations 800 --games 200 --iterations 200 --checkpoint-dir /output/checkpoints
```
