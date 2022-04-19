import numpy as np
from typing import List, Tuple
from stud.data_pre_processor import DataPreprocessor, IDX2TAG
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os

from stud.model import Model, prepare_batch


def build_model(device: str) -> Model:
    model = StudentModel(device)
    return model


class RandomBaseline(Model):
    options = [
        (3111, "B-CORP"),
        (3752, "B-CW"),
        (3571, "B-GRP"),
        (4799, "B-LOC"),
        (5397, "B-PER"),
        (2923, "B-PROD"),
        (3111, "I-CORP"),
        (6030, "I-CW"),
        (6467, "I-GRP"),
        (2751, "I-LOC"),
        (6141, "I-PER"),
        (1800, "I-PROD"),
        (203394, "O"),
    ]


class StudentModel:
    def __init__(self, device):
        self.pre_processor = DataPreprocessor()

        model = Model("glove-twitter-25", 26, False)
        model_path = os.path.join("stud", "test_3.model")
        print("loading cached model in '{}'".format(model_path))
        model.load_state_dict(torch.load(model_path))
        model.to(device)
        self.model = model

    def predict(self, tokens: List[List[str]]) -> List[List[str]]:
        self.pre_processor.process_sentences(tokens)
        self.pre_processor.pre_process_pos_tags()
        batch = [fv for fv in self.pre_processor]

        with torch.no_grad():
            res, mask = self.model(*prepare_batch(batch, training=False))
            predictions = self.model.crf.decode(res, mask=mask)
        predictions = [list(map(lambda x: IDX2TAG[x], pred)) for pred in predictions]

        return predictions
