import csv
import stanza
from gensim.models import KeyedVectors
from torch.utils.data import Dataset
import pickle
import os


WORD2VEC_DIR = os.path.join("stud", "pre_trained_models", "word2vec")

TAG2IDX = {
    "O": 0,
    "B-CORP": 1,
    "B-CW": 2,
    "B-GRP": 3,
    "B-LOC": 4,
    "B-PER": 5,
    "B-PROD": 6,
    "I-CORP": 7,
    "I-CW": 8,
    "I-GRP": 9,
    "I-LOC": 10,
    "I-PER": 11,
    "I-PROD": 12,
    "PAD": 13,
}

IDX2TAG = [
    "O",
    "B-CORP",
    "B-CW",
    "B-GRP",
    "B-LOC",
    "B-PER",
    "B-PROD",
    "I-CORP",
    "I-CW",
    "I-GRP",
    "I-LOC",
    "I-PER",
    "I-PROD",
    "PAD",
]

POS2INT = {
    "X": 0,
    "SYM": 1,
    "AUX": 2,
    "PUNCT": 3,
    "VERB": 4,
    "PRON": 5,
    "ADP": 6,
    "PART": 7,
    "NOUN": 8,
    "DET": 9,
    "PROPN": 10,
    "NUM": 11,
    "CCONJ": 12,
    "INTJ": 13,
    "SCONJ": 14,
    "ADV": 15,
    "ADJ": 16,
    "PAD": 17,
}


class DataPreprocessor(Dataset):
    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, index):
        return (
            self.sentences[index],
            self.tags[index] if self.tags else None,
            self.pos_tags_int[index],
        )

    def __init__(
        self, word2vec_name="glove-twitter-25", sentences=None, cache_label=None
    ):
        self.word2Vec = KeyedVectors.load(os.path.join(WORD2VEC_DIR, word2vec_name))
        self.stanza_pos_pipeline = stanza.Pipeline(
            "en",
            dir="stud/pre_trained_models/stanza_pos",
            processors="tokenize,mwt,pos",
            tokenize_pretokenized=True,
        )

        self.plaintext_sentences = None
        self.tags = None
        self.pos_tags = None
        self.pos_tags_int = None
        self.cache_dir = "stud/pre_trained_models/pre_processor"

        if cache_label:
            self._get_cached_data(cache_label)
            return
        # initialize variables
        if not sentences:
            self.sentences = None
        else:
            # TODO: Implement instantiation-time processing of sentences
            pass

    def store_training_preprocessor(self, cache_label=""):
        for att in [
            "sentences",
            "plaintext_sentences",
            "tags",
            "pos_tags",
            "pos_tags_int",
        ]:
            with open(
                os.path.join(self.cache_dir, cache_label + att + ".pk"), "wb+"
            ) as f:
                pickle.dump(getattr(self, att), f)

    def _get_cached_data(self, cache_label):
        for att in [
            "sentences",
            "plaintext_sentences",
            "tags",
            "pos_tags",
            "pos_tags_int",
        ]:
            with open(
                os.path.join(self.cache_dir, cache_label + att + ".pk"), "rb"
            ) as f:
                setattr(self, att, pickle.load(f))

    def pre_process_pos_tags(self):
        if not self.sentences:
            raise ValueError("'self.sentences' must be set")
        else:
            self.pos_tags = []
            self.pos_tags_int = []
            res = self.stanza_pos_pipeline(self.plaintext_sentences)
            for sentence in res.sentences:
                sentence_tags = [word.pos for word in sentence.words]
                pos_tags_int = [POS2INT[word.pos] for word in sentence.words]

                self.pos_tags.append(sentence_tags)
                self.pos_tags_int.append(pos_tags_int)

    def process_data_file(self, input_data_path):
        # set up
        sentences = []
        tagged_sentences = []
        plaintext_sentences = []

        with open(input_data_path) as f:
            tsvin = csv.reader(f, delimiter="\t")
            for line in tsvin:
                # CASE: Empty line, end of sentence
                if not line:
                    if tags:
                        # in case the dataset is labeled
                        tagged_sentences.append(tags)
                        assert len(tags) == len(words)

                    # If both tags and word-indices are defined(labeled set)
                    sentences.append(words)
                    plaintext_sentences.append(plaintext_sentence)

                # CASE: hash, beginning of sentence
                elif line[0] == "#":
                    words = []
                    tags = []
                    plaintext_sentence = []
                    continue
                else:
                    try:
                        idx = self.word2Vec.get_index(line[0])
                    except KeyError:
                        idx = -1
                    finally:
                        plaintext_sentence.append(line[0])
                        # offset the `idx` by two, as unknown and
                        # padding entries need to be considered
                        words.append(idx + 2)

                if len(line) > 1:
                    tags.append(TAG2IDX[line[1]])

            self.sentences = sentences
            self.tags = tagged_sentences
            self.plaintext_sentences = plaintext_sentences

    def process_sentences(self, list_of_sentences):
        sentences = []
        for plaintext_sentence in list_of_sentences:
            sentence = []
            for word in plaintext_sentence:
                try:
                    idx = self.word2Vec.get_index(word)
                except KeyError:
                    idx = -1
                finally:
                    # offset the `idx` by two, as unknown and
                    # padding entries need to be considered
                    sentence.append(idx + 2)
            sentences.append(sentence)

        self.sentences = sentences
        self.plaintext_sentences = list_of_sentences
