import pickle

import gensim.downloader as api
from gensim.models import KeyedVectors
from stud.data_pre_processor import DataPreprocessor
import os
import random

PRETRAINED_MODEL_DIR = os.path.join("stud", "pre_trained_models", "word2vec")


embeddings = [
    "glove-wiki-gigaword-50",
    "glove-wiki-gigaword-100",
    "glove-wiki-gigaword-200",
    "glove-twitter-25",
    "glove-twitter-50",
    "glove-twitter-100",
]

# download various gensim word2vec
# embeddings to streamline later model-selection
def download_embeddings():
    for embedding in embeddings:
        model = api.load(embedding)
        out_file_path = os.path.join(PRETRAINED_MODEL_DIR, embedding)
        print("storing model to '{}'".format(out_file_path))
        model.save(out_file_path)


def preprocess_training_data():
    """
    build and cache the training data for the various embeddings and their resulting represenations in the feature space
    """

    # for training data. Results in pickled tensors containing features of EMBEDDING_DIM+1 dimensions and labels
    result_dict = {}

    for embedding in embeddings:
        dp = DataPreprocessor(word2vec_name=embedding)
        dp.process_data_file("../data/train.tsv")
        dp.pre_process_pos_tags()
        dataset = [tup for tup in dp]
        print("length ", len(dataset))
        print("dataset[0] ", dataset[0])
        filepath = os.path.join(
            "stud", "pre_trained_models", "cached_training_sets", embedding
        )
        with open(filepath, "wb+") as f:
            pickle.dump(dataset, f)
        result_dict[embedding] = {"fp": filepath, "dims": len(dataset[0][0]) + 1}
    print(result_dict)
