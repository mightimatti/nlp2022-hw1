"""
    Hyperparameter tuning based on pytoch Ray-tune example
    https://pytorch.org/tutorials/beginner/hyperparameter_tuning_tutorial.html
"""


import pickle
from functools import partial
import numpy as np
import os

from . import FILE_DIR
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from seqeval.metrics import accuracy_score, f1_score

from stud.model import Model, prepare_batch
from stud.data_pre_processor import DataPreprocessor, IDX2TAG
from torch.utils.data import DataLoader

from torch.utils.data import random_split
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler

cached_embeddings = [
    # ("glove-wiki-gigaword-50", 50),
    # ("glove-wiki-gigaword-100", 100),
    # ("glove-wiki-gigaword-200", 200),
    ("glove-twitter-25", 25),
    # ("glove-twitter-50", 50),
    # ("glove-twitter-100", 100),
]

# min_config = {
#     "freeze_embedding": True,
#     "embedding_name": cached_embeddings[2],
#     "learning_rate": 1e-2,
#     "batch_size": 128,
#     "max_epochs": 1,
# }


def train_min_config():
    train_ner(min_config, None)

def train_ner(config, checkpoint_dir, data_dir=None):

    # load cached pre-processed training-set corresponding with the current choice of embedding
    embedding_filepath = os.path.join(
        FILE_DIR,
        "stud",
        "pre_trained_models",
        "cached_training_sets",
        config["embedding_name"][0],
    )

    with open(embedding_filepath, "rb") as f:
        dataset = pickle.load(f)
    model = Model(
        config["embedding_name"][0],
        config["embedding_name"][1] + 1,
        config["freeze_embedding"],
    )

    split_lengths = [int(len(dataset) * 0.8), int(len(dataset) * 0.2)]
    train_set, test_set = random_split(dataset, split_lengths)

    device = "cpu"

    if torch.cuda.is_available():
        device = "cuda:0"
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])

    if checkpoint_dir:
        model_state, optimizer_state = torch.load(
            os.path.join(checkpoint_dir, "checkpoint")
        )
        model.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)

    train_data_loader = DataLoader(
        train_set,
        collate_fn=prepare_batch,
        shuffle=True,
        batch_size=config["batch_size"],
        num_workers=3,
    )

    test_data_loader = DataLoader(
        test_set,
        collate_fn=(lambda x: prepare_batch(x, pad_label=False)),
        shuffle=False,
        batch_size=config["batch_size"],
        num_workers=3,
    )

    for epoch in range(config["max_epochs"]):  # loop over the dataset multiple times
        running_loss = 0.0
        epoch_steps = 0
        for i, (
            sentences,
            labels,
            pos_tags,
        ) in enumerate(train_data_loader):
            sentences = sentences.to(device)
            pos_tags = pos_tags.to(device)
            labels = labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            fc, mask = model(sentences, pos_tags)
            loss = -model.crf(fc.log_softmax(2), labels, mask=mask)

            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            epoch_steps += 1
            if i % 20 == 19:  # print every 2000 mini-batches
                print(
                    "[%d, %5d] loss: %.3f"
                    % (epoch + 1, i + 1, running_loss / epoch_steps)
                )
                running_loss = 0.0

        # Validation loss
        val_loss = 0.0
        val_steps = 0
        total = 0
        for i, (
            sentences,
            labels,
            pos_tags,
        ) in enumerate(test_data_loader):
            with torch.no_grad():
                sentences = sentences.to(device)
                pos_tags = pos_tags.to(device)

                output, mask = model(sentences, pos_tags)
                predictions = model.crf.decode(output, mask=mask)

                predictions = [list(map(lambda x: IDX2TAG[x], pred)) for pred in predictions]
                labels = [list(map(lambda x: IDX2TAG[x], pred)) for pred in labels]

                f = f1_score(predictions, labels, average="macro")
                acc = accuracy_score(labels, predictions)

                total += len(labels)

                val_steps += 1

        with tune.checkpoint_dir(epoch) as checkpoint_dir:
            path = os.path.join(checkpoint_dir, "checkpoint")
            torch.save((model.state_dict(), optimizer.state_dict()), path)

        tune.report(f1=f, accuracy=acc)
    print("Finished Training")


def main(num_samples=50, gpus_per_trial=0):
    config = {
        "freeze_embedding": False,
        # "freeze_embedding": tune.choice([True, False]),
        "embedding_name": tune.choice(cached_embeddings),
        "learning_rate": tune.uniform(0.045, 0.065),
        # "learning_rate": 0.05,
        "batch_size": tune.choice([ 32]),
        "max_epochs": 3,
    }

    # config = min_config 

    scheduler = ASHAScheduler(
        metric="f1",
        mode="max",
        max_t=config["max_epochs"],
        grace_period=1,
        reduction_factor=2,
    )
    reporter = CLIReporter(
        # parameter_columns=["l1", "l2", "lr", "batch_size"],
        metric_columns=["loss", "f1", "training_iteration"]
    )
    result = tune.run(
        partial(train_ner),
        resources_per_trial={"cpu": 1, "gpu": gpus_per_trial},
        config=config,
        num_samples=num_samples,
        scheduler=scheduler,
        progress_reporter=reporter,
    )

    best_trial = result.get_best_trial("f1", "max", "all")
    print("Best trial config: {}".format(best_trial.config))
    print("Best trial final f0.623631: {}".format(best_trial.last_result["f1"]))
    print(
        "Best trial final validation accuracy: {}".format(
            best_trial.last_result["accuracy"]
        )
    )

    best_trained_model = Model(
        best_trial.config["embedding_name"][0],
        best_trial.config["embedding_name"][1] + 1,
        best_trial.config["freeze_embedding"],
    )

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if gpus_per_trial > 1:
            best_trained_model = nn.DataParallel(best_trained_model)
    best_trained_model.to(device)

    best_checkpoint_dir = best_trial.checkpoint.value
    model_state, optimizer_state = torch.load(
        os.path.join(best_checkpoint_dir, "checkpoint")
    )
    best_trained_model.load_state_dict(model_state)
    torch.save(best_trained_model, os.path.join(FILE_DIR, "winner.model"))
    
    # dp = DataPreprocessor(word2vec_name=best_trial.config["embedding_name"][0])
    # dp.process_data_file(os.path.join(FILE_DIR, "..", 'data', "dev.tsv"))

    # output, mask = best_trained_model(sentences, pos_tags)
    # predictions = best_trained_model.crf.decode(output, mask=mask)

    # f = f1_score(predictions, labels, average="macro")
    # acc = accuracy_score(labels, predictions)

    # # test_acc = test_accuracy(best_trained_model, device)
    # print("Best trial test set accuracy: {}".format(test_acc))


if __name__ == "__main__":
    # You can change the number of GPUs per trial here:
    main(num_samples=12,  gpus_per_trial=0,)
