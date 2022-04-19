import torch
import torch.nn as nn
import numpy as np
import os
from gensim.models import KeyedVectors
from . import FILE_DIR

# WORD2VEC_MODEL = "glove-wiki-gigaword-50"


# loosely based on QnA notebook
def build_torch_embedding_layer(
    word2vec_filename, padding_idx: int = 0, freeze: bool = False
):
    """
    Builds an embeddinglayer transforming tokens to weightvectors.
    Loosely based on Notebook #8(QnA)
    """

    WORD2VEC_PATH = os.path.join(
        FILE_DIR, "stud", "pre_trained_models", "word2vec", word2vec_filename
    )
    # Load model, cache or downloaded
    weights = KeyedVectors.load(WORD2VEC_PATH, mmap="r")

    # random vector for pad
    pad = np.random.rand(1, weights.vectors.shape[1])
    # print("shape of padding: ", pad.shape)

    # mean vector for unknowns
    unk = np.mean(weights.vectors, axis=0, keepdims=True)
    # print("shape of unknowns: ", unk.shape)

    # concatenate pad and unk vectors on top of pre-trained weights
    vectors = np.concatenate((pad, unk, weights.vectors))

    # and return the embedding layer
    return torch.nn.Embedding.from_pretrained(
        torch.FloatTensor(vectors), padding_idx=padding_idx, freeze=freeze
    )
