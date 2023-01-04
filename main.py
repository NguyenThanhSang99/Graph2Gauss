import argparse
import random

import numpy as np

import sklearn.linear_model as sklm
import sklearn.metrics as skm
import sklearn.model_selection as skms
import torch
import torch.optim as optim
from ignite.engine import Engine, Events
from ignite.handlers import ModelCheckpoint
from torch.utils.data import DataLoader, get_worker_info
from dataloader import *
from model import Encoder

CHECKPOINT_PREFIX = "g2g"

def train_test_split(n, train_ratio=0.5):
    nodes = list(range(n))
    split_index = int(n * train_ratio)

    random.shuffle(nodes)
    return nodes[:split_index], nodes[split_index:]


def reset_seeds(seed=None):
    if seed is None:
        seed = get_worker_info().seed

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=2023)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "-k", type=int, default=1, help="Maximum depth to consider in level sets"
    )
    parser.add_argument("-c", "--checkpoint")
    parser.add_argument("--checkpoints")    
    parser.add_argument("--dataset", default="data/citeseer.npz")
    args = parser.parse_args()

    epochs = args.epochs
    nsamples = args.samples
    learning_rate = args.lr
    seed = args.seed
    n_workers = args.workers
    K = args.k
    checkpoint_path = args.checkpoint
    checkpoints_path = args.checkpoints
    dataset_path = args.dataset

    if seed is not None:
        reset_seeds(seed)

    A, X, z = load_dataset(dataset_path)

    n = A.shape[0]
    train_nodes, val_nodes = train_test_split(n, train_ratio=1.0)
    A_train = A[train_nodes, :][:, train_nodes]
    X_train = X[train_nodes]
    z_train = z[train_nodes]
    A_val = A[val_nodes, :][:, val_nodes]
    X_val = X[val_nodes]
    z_val = z[val_nodes]

    train_data = AttributedGraph(A_train, X_train, z_train, K)
    val_data = AttributedGraph(A_val, X_val, z_val, K)

    L = 64
    encoder = Encoder(X.shape[1], L)
    if checkpoint_path:
        encoder.load_state_dict(torch.load(checkpoint_path))

    optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)

    def step(engine, args):
        optimizer.zero_grad()

        loss = encoder.compute_loss(*args[0])
        loss.backward()

        optimizer.step()

        return loss.item()

    trainer = Engine(step)

    if checkpoints_path:
        handler = ModelCheckpoint(
            checkpoints_path, CHECKPOINT_PREFIX, n_saved=3, save_interval=50
        )
        trainer.add_event_handler(
            Events.ITERATION_COMPLETED, handler, {"encoder": encoder}
        )

    @trainer.on(Events.ITERATION_STARTED)
    def enable_train_mode(engine):
        encoder.train()

    @trainer.on(Events.ITERATION_COMPLETED)
    def log_loss(engine):
        if engine.state.iteration % 10 == 0:
            print(f"Epoch {engine.state.iteration:2d} - Loss {engine.state.output:.3f}")

    @trainer.on(Events.ITERATION_COMPLETED)
    def run_validation(engine):
        if engine.state.iteration % 50 != 1:
            return

        # Skip if there is no validation set
        if val_data.A.shape[0] == 0:
            return

        encoder.eval()
        loss = encoder.compute_loss(val_data, nsamples)
        print(f"Validation loss {loss:.3f}")

    @trainer.on(Events.ITERATION_COMPLETED)
    def node_classification(engine):
        if engine.state.iteration % 50 != 1:
            return

        f1_scorer = skm.SCORERS["f1_macro"]

        # Apparently evaluating on the training data is normal for graph
        # learning?
        X = train_data.X
        z = train_data.z

        encoder.eval()
        mu, sigma = encoder(X)
        X_learned = mu.detach().numpy()

        f1 = 0.0
        n_rounds = 1
        for i in range(n_rounds):
            X_train, X_test, z_train, z_test = skms.train_test_split(
                X_learned, z, train_size=0.1, stratify=z
            )

            lr = sklm.LogisticRegressionCV(
                multi_class="auto", solver="lbfgs", cv=3, max_iter=1000
            )
            lr.fit(X_train, z_train)

            f1 += f1_scorer(lr, X_test, z_test)
        f1 /= n_rounds

        print(f"LR F1 score {f1}")

    iterations = epochs // n_workers
    dataset = GraphDataset(train_data, nsamples, iterations)
    loader = DataLoader(
        dataset,
        batch_size=1,
        num_workers=n_workers,
        worker_init_fn=reset_seeds,
        collate_fn=lambda args: args,
    )
    epochs = 1
    trainer.run(loader, epochs)


if __name__ == "__main__":
    main()
