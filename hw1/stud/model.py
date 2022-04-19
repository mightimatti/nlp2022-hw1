from typing import List
from torch import nn
from stud.embedding_builder import build_torch_embedding_layer
from stud.data_pre_processor import DataPreprocessor, TAG2IDX
from torchcrf import CRF
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


CRF_TAGS = len(TAG2IDX)
SEQUENCE_LENGTH = 42
BATCH_SIZE = 16
EPOCHS = 12
LEARNING_RATE = 0.005
# Hyper-parameters
LSTM_UNITS = 42
LSTM_DROUPOUT = 0.0
LSTM_DEPTH = 1
ANNOTATED_INPUT_SIZE = 26
LSTM_IS_BIDIRECTIONAL = 1

# based on QnA Notebook
def prepare_batch(batch, training=True, pad_label=True):
    # extract features and labels from batch
    x = [42 * [0]]
    # convert features to tensor and pad them
    x += [sample[0] for sample in batch]
    x = pad_sequence(
        [torch.as_tensor(sample) for sample in x], batch_first=True, padding_value=0
    )
    x = x[1:, :]

    z = [42 * [0]]
    z += [sample[2] for sample in batch]
    z = pad_sequence(
        [torch.as_tensor(sample) for sample in z],
        batch_first=True,
        padding_value=17,
    )
    z = z[1:, :]

    if training:
        # convert and pad labels too
        y = [42 * [0]]
        y += [sample[1] for sample in batch]
        if pad_label:
            y = pad_sequence(
                [torch.as_tensor(sample) for sample in y],
                batch_first=True,
                padding_value=13,
            )  # convert and pad POS-tags as well
            y = y[1:, :]
        else:
            y=y[1:]
        return (x, y, z)

    else:
        return (
            x,
            z,
        )


class Model(nn.Module):
    def __init__(self, WORD2VEC_FN, ANNOTATED_INPUT_SIZE, FREEZE_EMBEDDING):
        super(Model, self).__init__()
        self.embedding = build_torch_embedding_layer(
            WORD2VEC_FN, freeze=FREEZE_EMBEDDING
        )
        self.lstm = nn.LSTM(
            ANNOTATED_INPUT_SIZE,
            CRF_TAGS,
            LSTM_DEPTH,
            bidirectional=bool(LSTM_IS_BIDIRECTIONAL),
            dropout=LSTM_DROUPOUT,
            batch_first=True,
        )
        self.fc = nn.Linear(2 * CRF_TAGS, CRF_TAGS)
        self.crf = CRF(CRF_TAGS, batch_first=True)

    def forward(self, sentences, pos_tags):

        # mask-tensor indicating what entries should
        # be disregarded while scoring the current batch
        padding_mask_tensor = sentences != 0
        embedded_sentences = self.embedding(sentences)

        # combine the POS-tags with the embedded phrases
        combined_data = torch.cat(
            (
                embedded_sentences,
                pos_tags.reshape((-1, SEQUENCE_LENGTH, 1)),
            ),
            2,
        )

        # Size[BATCH_SIZE, SENTENCE_LENGTH, ANNOTATED_INPUT_SIZE])
        # ex. [32, 42, 26]
        out, _ = self.lstm(combined_data)

        # calculate a linear combination of the two output directions of
        # of the LSTM
        # ex. [32, 42, 13]
        linear_comb = self.fc(out)

        # return the linear combination and the mask tensor
        return linear_comb, padding_mask_tensor


def train():
    # set up Data for training

    # `DataPreprocessor` objects containing training set and dev set
    # derived from Data
    dp_train = DataPreprocessor(cache_label="train")
    dp_dev = DataPreprocessor(cache_label="dev")

    num_workers = min(os.cpu_count(), 4)  # it is usually 4 workers per GPU

    train_data_loader = DataLoader(
        dp_train,
        collate_fn=prepare_batch,
        shuffle=True,
        batch_size=BATCH_SIZE,
        num_workers=num_workers,
    )

    test_data_loader = DataLoader(
        dp_dev,
        collate_fn=prepare_batch,
        shuffle=False,
        batch_size=BATCH_SIZE,
        num_workers=num_workers,
    )

    # Instantiate Model
    model = Model().to(device)

    # Loss and optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Train the model
    TOTAL_STEPS = len(train_data_loader)
    for epoch in range(EPOCHS):
        for i, (
            sentences,
            labels,
            pos_tags,
        ) in enumerate(train_data_loader):

            sentences = sentences.to(device)
            pos_tags = pos_tags.to(device)
            labels = labels.to(device)

            # Forward pass
            fc, mask = model(sentences, pos_tags)
            loss = -model.crf(fc.log_softmax(2), labels, mask=mask)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i + 1) % 100 == 0:
                print(
                    f"Epoch [{epoch + 1}/{EPOCHS}], Step [{i + 1}/{TOTAL_STEPS}], Loss: {loss.item():.4f}"
                )

    torch.save(model.state_dict(), "stud/test_2.model")
