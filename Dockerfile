FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

WORKDIR /app

RUN pip install --no-cache-dir python-chess==1.999 tqdm

COPY alphazero/ alphazero/
COPY main.py .

CMD ["python", "main.py", \
     "--res-blocks", "10", \
     "--channels", "128", \
     "--simulations", "400", \
     "--mcts-batch", "64", \
     "--games", "100", \
     "--max-moves", "200", \
     "--epochs", "10", \
     "--batch-size", "256", \
     "--eval-games", "20", \
     "--iterations", "100", \
     "--checkpoint-dir", "/output/checkpoints"]
