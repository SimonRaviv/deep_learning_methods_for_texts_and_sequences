"""
A library module to serve RNN assignment.

Author:
Simon Raviv.
"""
import os
import argparse
import random
import numpy as np
import pandas as pd
import torch
import re
import copy
import datetime

from collections import Counter
from typing import List, Tuple
from matplotlib import pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torch.utils.checkpoint import checkpoint
from itertools import chain
from functools import partial


# Global variables:
SEED = 2022
GLOBAL_RAND_GEN = torch.Generator()
POS_NEG_SEPARATOR = '\t'
MODULE_ROOT_DIR = os.path.dirname(__file__)
DEFAULT_CHARACTERS = "abcd123456789"
PADDING_LABEL = "<PADDING_LABEL>"
PADDING_TOKEN = "<PADDING>"
UNKNOWN_TOKEN = "<UNKNOWN>"
PADDING_CHAR = "<PAD>"
UNKNOWN_CHAR = "<UNK>"
NUM_TOKEN = "<NUMBER>"
DATE_TOKEN = "<DATE>"
EMPTY_TOKEN = ""
NER_SEPARATOR = "\t"
POS_SEPARATOR = " "
NER_COMMON_TAG = "O"
NER_TAGGING = "ner"
POS_TAGGING = "pos"
POS_NEG_TAGGING = "pos_neg"
SPECIAL_TOKENS = [PADDING_TOKEN, UNKNOWN_TOKEN, NUM_TOKEN, DATE_TOKEN, EMPTY_TOKEN]
MAX_WORD_SIZE = 15
DS_TO_SEPARATOR = {"ner": NER_SEPARATOR, "pos": POS_SEPARATOR, "pos_neg": POS_NEG_SEPARATOR}
EMPTY_LABEL_INDEX = -1
WORD_REPR_EMBEDDING = "(a) Embedding"
WORD_REPR_CHAR_LSTM = "(b) Character level LSTM"
WORD_REPR_SUBWORD = "(c) Embedding + Subword"
WORD_REPR_EMBEDDING_CHAR_LSTM = "(d) Embedding + Character level LSTM followed by a linear layer"
WORD_REPRESENTATION_TYPES = [
    WORD_REPR_EMBEDDING, WORD_REPR_CHAR_LSTM, WORD_REPR_SUBWORD, WORD_REPR_EMBEDDING_CHAR_LSTM]
WORD_REPRESENTATION_TYPES_CLI = ["a", "b", "c", "d"]
TIME_FORMAT = "%H:%M:%S"


def get_time_now() -> str:
    """
    @brief: Returns the current time in the format HH:MM:SS.

    @return: The current time.
    """
    return datetime.datetime.strptime(datetime.datetime.now().strftime(TIME_FORMAT), TIME_FORMAT)


class LSTM(torch.nn.Module):
    """
    @brief: LSTM layer.

    A class to implement LSTM layer using LSTMCell.
    """

    def __init__(self, input_size: int, hidden_size: int, device: str,
                 padding_index: int, as_transducer: bool = False) -> None:
        """
        @brief: Initialize the LSTM layer.

        @param input_size: Size of the input sequence.
        @param hidden_size: Size of the hidden layer of the LSTM.
        @param device: Device to use.
        @param padding_index: Index of the padding token.
        @param as_transducer: Use LSTM transducer, otherwise LSTM acceptor.

        @return: None.
        """
        super(LSTM, self).__init__()
        self.device = device
        self.padding_index = padding_index
        self.lstm_cell = torch.nn.LSTMCell(input_size=input_size, hidden_size=hidden_size, device=device)
        self.as_transducer = as_transducer
        self.forward = self._forward_transducer if as_transducer == True else self._forward_acceptor

    def _init_hidden(self, batch_size: int) -> None:
        """
        @brief: Initialize the hidden state.

        @param batch_size: Batch size.

        @return: None.
        """
        self.hidden_state = torch.zeros(batch_size, self.lstm_cell.hidden_size, device=self.device)
        self.cell_state = torch.zeros(batch_size, self.lstm_cell.hidden_size, device=self.device)

    def _forward_acceptor(self, sequence: torch.Tensor, input_length: torch.Tensor) -> torch.Tensor:
        """
        @brief: Forward propagation for LSTM acceptor.

        @param sequence: Input sequence representation.
        @param input_length: Length of the input sequence without padding.

        @return: Last not padded hidden state.
        """
        batch_size, sequence_max_length, _ = sequence.shape

        # Initialize the hidden state:
        self._init_hidden(batch_size)
        last_hidden_state = torch.zeros(batch_size, self.lstm_cell.hidden_size, device=self.device)

        # Run the LSTM:
        for word in range(sequence_max_length):
            # Get next hidden state:
            self.hidden_state, self.cell_state = self.lstm_cell(
                input=sequence[:, word, :], hx=(self.hidden_state, self.cell_state))

            # Save the last hidden state:
            last_hidden_state[word == input_length - 1] = self.hidden_state[word == input_length - 1]

        output = last_hidden_state
        return output

    def _forward_transducer(self, sequence: torch.Tensor, input_length: torch.Tensor) -> torch.Tensor:
        """
        @brief: Forward propagation for LSTM transducer.

        @param sequence: Input sequence representation.
        @param input_length: Length of the input sequence without padding.

        @return: All hidden states.
        """
        batch_size, sequence_max_length, _ = sequence.shape

        # Initialize the hidden state:
        self._init_hidden(batch_size)
        hidden_states = []

        # Run the LSTM:
        for word in range(sequence_max_length):
            # Get next hidden state:
            self.hidden_state, self.cell_state = self.lstm_cell(
                input=sequence[:, word, :], hx=(self.hidden_state, self.cell_state))
            hidden_states.append(self.hidden_state)

        output = torch.stack(hidden_states, dim=1)
        return output


class BiLSTM(torch.nn.Module):
    """
    @brief: Bi-LSTM layer.

    A class to implement Bidirectional LSTM layer using LSTM layer.
    """

    def __init__(self, input_size: int, hidden_size: int, device: str,
                 padding_index: int, as_transducer: bool = False) -> None:
        """
        @brief: Initialize the BiLSTM layer.

        @param input_size: Input size.
        @param hidden_size: Hidden size.
        @param device: Device.
        @param padding_index: Padding index.
        @param as_transducer: Whether to use the transducer.

        @return: None.
        """
        super(BiLSTM, self).__init__()
        self.device = device
        self.padding_index = padding_index
        self.lstm_forward = LSTM(
            input_size=input_size, hidden_size=hidden_size, device=device,
            padding_index=padding_index, as_transducer=as_transducer)
        self.lstm_backward = LSTM(
            input_size=input_size, hidden_size=hidden_size,  device=device,
            padding_index=padding_index, as_transducer=as_transducer)

    def _get_inversed_input(self, xi: torch.Tensor, input_length: torch.LongTensor) -> torch.Tensor:
        """
        @brief: Get the inversed input.

        @param xi: Input sequence representation with shape (batch_size, sequence_length, representation_dim).
        @param input_length: Length of the input sequence without padding.

        @return: Inversed input sequence.
        """
        batch_size, length, representation_dim = xi.shape
        xi = torch.flip(xi, dims=(1,))

        rolled_cyclic_sequence = xi[:, [*range(length), *range(length - 1)], :].clone()
        stride_0, stride_1, stride_2 = rolled_cyclic_sequence.stride()
        result = torch.as_strided(rolled_cyclic_sequence, (batch_size, length, length,
                                  representation_dim), (stride_0, stride_1, stride_1, stride_2))

        inversed_sequence = result[torch.arange(batch_size), (length - input_length) % length, :, :]
        return inversed_sequence

    def forward(self, x: torch.Tensor, input_length: torch.Tensor) -> torch.Tensor:
        """
        @brief: Run forward propagation.

        @param x: Input sequence representation.
        @param input_length: Length of the input sequence without padding.

        @return: The biLSTM output.
        """
        n = x.shape[1]

        # Get the forward input:
        x1_i = x[:, 0:n, :]

        # Get the backward input:
        xn_i = self._get_inversed_input(xi=x, input_length=input_length)

        # Run the forward LSTM:
        lstm_f = self.lstm_forward(x1_i, input_length=input_length)

        # Run the backward LSTM:
        lstm_b = self.lstm_backward(xn_i, input_length=input_length)

        # Concatenate the forward and backward outputs in correct order
        # bi = BiLSTM(x1, ..., xn; i) = LSTM_F (x1, ..., xi) o LSTM_B(xn, ..., xi)
        b = []
        for sequence_index, sequence_i_len in enumerate(input_length.tolist()):
            index_1_i = torch.arange(0, sequence_i_len, 1, device=self.device)
            index_n_i = torch.arange(sequence_i_len - 1, -1, -1, device=self.device)
            x1_i = lstm_f[sequence_index, index_1_i, :]
            xn_i = lstm_b[sequence_index, index_n_i, :]
            xi_padding = torch.zeros(n - x1_i.shape[0], x1_i.shape[1], device=self.device)
            x1_i = torch.cat((x1_i, xi_padding), dim=0)
            xn_i = torch.cat((xn_i, xi_padding), dim=0)
            bi = torch.cat((x1_i, xn_i), dim=1)
            b.append(bi)

        # Concatenate the sequence:
        b = torch.stack(b, dim=0)
        return b


class CharacterLSTM(torch.nn.Module):
    """
    @brief: Character-LSTM layer.
    """

    def __init__(self, embedding_dim: int, hidden_size: int, device: str, padding_char_index: int,
                 char_vocabulary_size: int, word_idx2chars_idx: dict, padding_token_index: int) -> None:
        """
        @brief: Initialize the Character-LSTM layer.

        @param embedding_dim: Embedding dimension.
        @param hidden_size: Hidden size.
        @param device: Device.
        @param padding_char_index: Padding character index.
        @param char_vocabulary_size: Character vocabulary size.
        @param word_idx2chars_idx: Dictionary of word to characters index mapping.
        @param padding_token_index: Padding token index.

        @return: None.
        """
        super(CharacterLSTM, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.device = device
        self.padding_char_index = padding_char_index
        self.char_vocabulary_size = char_vocabulary_size
        self.word_idx2chars_idx = word_idx2chars_idx
        self.padding_token_index = padding_token_index
        self.embedding = torch.nn.Embedding(
            num_embeddings=char_vocabulary_size, embedding_dim=embedding_dim, padding_idx=padding_char_index)
        self.lstm = LSTM(
            input_size=embedding_dim, hidden_size=hidden_size, device=device, padding_index=padding_char_index)
        self.to(device)

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        """
        @brief: Run forward propagation.

        @param sequence: Input sequence of shape (batch_size, sequence_length).

        @return: Character LSTM Representation output.
        """
        # Create character input:
        input_length = torch.sum(sequence != self.padding_token_index, dim=1)
        batch_size, max_sequence_length = sequence.shape

        # Get maximum word length in the batch:
        all_words = sequence.flatten()[sequence.flatten() != self.padding_token_index].unique()
        max_word_length = all_words.cpu().apply_(lambda word: len(
            self.word_idx2chars_idx[word])).to(self.device).max().item()

        # Create character input:
        char_input = torch.full(
            (batch_size, max_sequence_length, max_word_length), fill_value=self.padding_char_index,
            dtype=torch.int64, device=self.device)
        for sequence_index, sequence_i_len in enumerate(input_length.tolist()):
            for word_index, word_i in enumerate(sequence[sequence_index, :sequence_i_len].tolist()):
                if word_i != self.padding_token_index:
                    word_i_chars = self.word_idx2chars_idx[word_i]
                    char_input[sequence_index, word_index, :len(word_i_chars)] = word_i_chars

        # Embed the character input:
        char_input_embedded = self.embedding(char_input)

        # Run the character-LSTM:
        output = torch.zeros(
            batch_size, max_sequence_length, self.hidden_size, dtype=torch.float32, device=self.device)
        for word_index in range(max_sequence_length):
            word_i = char_input_embedded[:, word_index, :]
            word_i_word_len = torch.sum(char_input[:, word_index, :] != self.padding_char_index, dim=1)
            output[:, word_index, :] = self.lstm(word_i, input_length=word_i_word_len)

        return output


class CBOWSubword(torch.nn.Module):
    """
    @brief: CBOW postfix/suffix subword representation layer.
    """

    def __init__(self, device: str = "", num_embeddings: int = -1, embedding_dim: int = -1,
                 padding_token_index: int = -1, word2subword_prefix_idx: dict = {},
                 word2subword_postfix_idx: dict = {}) -> None:
        super(CBOWSubword, self).__init__()
        self.device = device
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.padding_token_index = padding_token_index
        self.word2subword_prefix_idx = word2subword_prefix_idx
        self.word2subword_postfix_idx = word2subword_postfix_idx
        self.embedding = torch.nn.Embedding(
            num_embeddings=num_embeddings, embedding_dim=embedding_dim, padding_idx=padding_token_index)
        self.to(device)

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        """
        @brief: Return CBOW embedding using prefix/postfix sub-words.

        This method uses the summation over word and the sub-words embeddings.

        @param sequence: Input sequence of shape (batch_size, sequence_length).

        @return: Subword CBOW representation output.
        """
        # Get sub-words:
        x_prefix = sequence.clone().cpu().apply_(lambda word: self.word2subword_prefix_idx[word]).to(self.device)
        x_postfix = sequence.clone().cpu().apply_(lambda word: self.word2subword_postfix_idx[word]).to(self.device)

        # Get embeddings:
        word_embedding = self.embedding(sequence)
        x_prefix_embedding = self.embedding(x_prefix)
        x_postfix_embedding = self.embedding(x_postfix)

        # CBOW - Sum the embeddings:
        output = word_embedding + x_prefix_embedding + x_postfix_embedding

        return output


class WordEmbeddingCharacterLSTM(torch.nn.Module):
    """
    @brief: Word embedding concatenated with character LSTM representation layer.

    This layer is used to represent a word using the word embedding and the character-LSTM representation concatenated.
    This representation will be fed to linear layer.
    """

    def __init__(self, device: str, word_embedding_dim: int, char_embedding_dim: int,
                 hidden_size: int, word_vocabulary_size: int, padding_token_index: int,
                 word_idx2chars_idx: dict, padding_char_index: int, char_vocabulary_size: int,
                 dropout_probability: int = 0) -> None:
        """
        @brief: Initialize the WordEmbeddingCharacterLSTM layer.

        @param device: Device.
        @param word_embedding_dim: Word embedding dimension.
        @param char_embedding_dim: Char embedding dimension.
        @param hidden_size: Hidden size.
        @param word_vocabulary_size: Word vocabulary size.
        @param word_idx2chars_idx: Dictionary of word to characters index mapping.
        @param padding_token_index: Padding token index.
        @param padding_char_index: Padding character index.
        @param char_vocabulary_size: Character vocabulary size.
        @param dropout_probability: Dropout probability for the linear layer.

        @return: None.
        """
        super(WordEmbeddingCharacterLSTM, self).__init__()
        self.device = device
        self.word_embedding_dim = word_embedding_dim
        self.char_embedding_dim = char_embedding_dim
        self.hidden_size = hidden_size
        self.word_vocabulary_size = word_vocabulary_size
        self.char_vocabulary_size = char_vocabulary_size
        self.word_idx2chars_idx = word_idx2chars_idx
        self.padding_char_index = padding_char_index
        self.padding_token_index = padding_token_index
        self.dropout_probability = dropout_probability
        self.word_embedding = torch.nn.Embedding(
            num_embeddings=word_vocabulary_size, embedding_dim=word_embedding_dim, padding_idx=padding_token_index)
        self.char_lstm = CharacterLSTM(
            device=device, char_vocabulary_size=char_vocabulary_size, embedding_dim=char_embedding_dim, hidden_size=hidden_size,
            padding_char_index=padding_char_index, word_idx2chars_idx=word_idx2chars_idx, padding_token_index=padding_token_index)
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(in_features=(word_embedding_dim + hidden_size),
                            out_features=(word_embedding_dim + hidden_size) // 2),
            torch.nn.Dropout(p=dropout_probability)
        )

        self.to(device)

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        """
        @brief: Return the word embedding concatenated with character LSTM representation.

        @param sequence: Input sequence of shape (batch_size, sequence_length).

        @return: WordEmbedding o CharacterLSTM Representation output.
        """
        # Embed the word:
        word_embedding = self.word_embedding(sequence)

        # Get the character-LSTM representation:
        char_lstm_representation = self.char_lstm(sequence)

        # Concatenate the word and character-LSTM representations:
        output = torch.cat((word_embedding, char_lstm_representation), dim=2)

        # Fed to the Linear layer:
        output = self.fc(output)

        return output


class ToTensor:
    """
    @brief: Convert a list of indexes into a tensor.
    """

    def __init__(self, device: str) -> None:
        """
        @brief: Initialize the ToTensor class.

        @param device: Device.

        @return: None.
        """
        self.device = device

    def __call__(self, indexes: list) -> torch.Tensor:
        """
        @brief: Convert a list of indexes into a tensor.

        @param indexes: List of indexes.

        @return: The tensor.
        """
        return torch.tensor(indexes, device=self.device, dtype=torch.int64)


class BaseDataset(Dataset):
    """
    @brief: Base dataset class for the different language tasks.
    """

    def __init__(self, device: str, root_dir: str, filename: str, separator: str, transform: object = None) -> None:
        """
        @brief: Initialize the dataset.

        @param device: The device to use.
        @param root_dir: The root directory of the dataset.
        @param filename: The filename of the dataset, if @ref root_dir is empty string,
                         filename will be used as the absolute path to the dataset file.
        @param separator: The separator of between the word to the label.
        @param transform: The transform to apply on the dataset samples.

        @return: None
        """
        super(BaseDataset, self).__init__()
        self.device = device
        self.root_dir = root_dir
        self.filename = filename
        self.separator = separator
        self.transform = transform
        self.data_file_path = os.path.join(self.root_dir, self.filename)
        self.X = []
        self.y = []
        self.vocabulary = set()
        self.vocabulary_size = 0
        self.idx2word = {}
        self.word2idx = {}
        self.labels = set()
        self.idx2label = {}
        self.label2idx = {}
        self.num_of_labels = 0
        self.max_sentence_length = 0
        self.unknown_token_index = -1
        self.padding_token_index = -1
        self.padding_label_index = -1

    def __len__(self) -> int:
        """
        @brief: Returns the length of the dataset.

        @return: The length of the dataset.
        """
        return len(self.X)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        @brief: Returns the data and label at the given index with the context.

        @param index: The index of the data and label to return.

        @return: The data and label at the given index with the context.
        """
        data = self.X[index] if self.transform is None else self.transform(self.X[index])
        label = self.y[index] if self.transform is None else self.transform(self.y[index])

        return data, label

    def initialize(self, metadata: dict = None) -> None:
        """
        @brief: Initialize the dataset.

        @return: None
        """
        self._initialize_metadata(metadata)
        self._transform_data_to_indexes()

    def get_metadata(self) -> dict:
        """
        @brief: Returns dataset metadata.

        @return: The dataset metadata.
        """
        return {
            "vocabulary": self.vocabulary,
            "vocabulary_size": self.vocabulary_size,
            "idx2word": self.idx2word,
            "word2idx": self.word2idx,
            "labels": self.labels,
            "idx2label": self.idx2label,
            "label2idx": self.label2idx,
            "num_of_labels": self.num_of_labels,
            "padding_token_index": self.padding_token_index,
            "padding_label_index": self.padding_label_index,
            "unknown_token_index": self.unknown_token_index
        }

    def _set_metadata(self, metadata: dict) -> None:
        """
        @brief: Sets the metadata.

        @param metadata: The metadata to set.

        @return: None
        """
        self.vocabulary = metadata["vocabulary"]
        self.idx2word = metadata["idx2word"]
        self.word2idx = metadata["word2idx"]
        self.vocabulary_size = metadata["vocabulary_size"]
        self.labels = metadata["labels"]
        self.idx2label = metadata["idx2label"]
        self.label2idx = metadata["label2idx"]
        self.num_of_labels = metadata["num_of_labels"]
        self.padding_token_index = metadata["padding_token_index"]
        self.padding_label_index = metadata["padding_label_index"]
        self.unknown_token_index = metadata["unknown_token_index"]

    def _initialize_words_metadata(self):
        """
        @brief: Initializes words metadata.

        @return: None
        """
        NotImplementedError("This method must be implemented in the child class.")

    def _initialize_labels_metadata(self):
        """
        @brief: Initializes the labels metadata.

        @return: None
        """
        NotImplementedError("This method must be implemented in the child class.")

    def _initialize_metadata(self, metadata: dict = None) -> None:
        """
        @brief: Initializes the metadata.

        The metadata used for dev/test datasets.

        @param metadata: The metadata to initialize from.

        @return: None
        """
        if metadata is None:
            self._initialize_words_metadata()
            self._initialize_labels_metadata()
        else:
            self._set_metadata(metadata)

    def _transform_data_to_indexes(self) -> None:
        """
        @brief: Transforms the data to tensor with indexes.

        @return: None
        """
        NotImplementedError("This method must be implemented in the child class.")


class SequenceClassificationDataset(BaseDataset):
    """
    @brief: Sequence classification dataset.
    """

    def __init__(self, device: str = "", root_dir: str = "", filename: str = "", separator: str = "") -> None:
        """
        @brief: Initialize the dataset.

        @param device: The device to use.
        @param root_dir: The root directory of the dataset.
        @param filename: The filename of the dataset, if @ref root_dir is empty string,
                         filename will be used as the absolute path to the dataset file.
        @param separator: The separator of between the word to the label.

        @return: None
        """
        super(SequenceClassificationDataset, self).__init__(device, root_dir, filename, separator)
        self._initialize_data()

    def _initialize_data(self) -> None:
        """
        @brief: Initializes the data.

        @return: None
        """
        data_dtf = pd.read_table(
            self.data_file_path, sep=self.separator, skip_blank_lines=False, header=None,
            keep_default_na=False, na_filter=True, quoting=3, names=["sentence", "label"])
        self.X = data_dtf.sentence.apply(lambda sentence: list(chain.from_iterable(sentence))).tolist()
        self.y = data_dtf.label.tolist()

        # Remove empty sentences:
        data = [(X, y) for X, y in zip(self.X, self.y) if len(X) > 0]
        self.X, self.y = zip(*data)

        self.max_sentence_length = max([len(sentence) for sentence in self.X])

        assert len(self.X) == len(self.y)

    def _initialize_words_metadata(self):
        self.vocabulary = set(chain.from_iterable(self.X + DEFAULT_CHARACTERS.split()))
        self.idx2word = {index: word for index, word in enumerate(self.vocabulary)}
        self.word2idx = {word: index for index, word in self.idx2word.items()}
        self.vocabulary_size = len(self.idx2word) + 1  # +1 for padding token
        self.padding_token_index = self.vocabulary_size - 1

    def _initialize_labels_metadata(self):
        self.labels = set(self.y)
        self.idx2label = dict(enumerate(self.labels))
        self.label2idx = {label: index for index, label in self.idx2label.items()}
        self.num_of_labels = len(self.idx2label)

    def _transform_data_to_indexes(self) -> None:
        def transform_X_to_indexes(sentence: str) -> int:
            sentence = [self.word2idx.get(word, self.unknown_token_index) for word in sentence]
            padding = [self.padding_token_index] * (self.max_sentence_length - len(sentence))
            return sentence + padding

        def transform_y_to_indexes(label: str) -> int:
            return self.label2idx.get(label)

        X = np.vectorize(transform_X_to_indexes, otypes=[torch.Tensor])(self.X).tolist()
        self.X = torch.tensor(X, dtype=torch.long, device=self.device)
        y = np.vectorize(transform_y_to_indexes, otypes=[int])(self.y).tolist()
        self.y = torch.tensor(y, dtype=torch.long, device=self.device)


class LanguageTaggingDataset(BaseDataset):
    """
    @brief: Language tagging dataset.
    """

    def __init__(self, device: str = "", root_dir: str = "", filename: str = "",
                 separator: str = "", word_representation: str = "") -> None:
        """
        @brief: Initialize the dataset.

        @param device: The device to use.
        @param root_dir: The root directory of the dataset.
        @param filename: The filename of the dataset, if @ref root_dir is empty string,
                         filename will be used as the absolute path to the dataset file.
        @param separator: The separator of between the word to the label.
        @param word_representation: Type of word representation to use, see @ref WORD_REPRESENTATION_TYPES_CLI.

        @return: None
        """
        super(LanguageTaggingDataset, self).__init__(device, root_dir, filename, separator, ToTensor(device))
        self.use_lstm_c_word_representation = True if (
            word_representation in [WORD_REPRESENTATION_TYPES_CLI[1], WORD_REPRESENTATION_TYPES_CLI[3]]) else False
        self.use_cbow_word_representation = True if word_representation == WORD_REPRESENTATION_TYPES_CLI[2] else False
        self._token_processors = [self._is_number, self._is_date]
        self.subword_len = 3
        self.cbow_subwords_prefix = {}
        self.cbow_subwords_postfix = {}
        self.cbow_word2subword_prefix_idx = {}
        self.cbow_word2subword_postfix_idx = {}
        self.char_vocabulary = set()
        self.char2idx = {}
        self.idx2char = {}
        self.word_idx2chars_idx = None
        self.char_vocabulary_size = 0
        self.padding_char_index = -1
        self.unknown_char_index = -1
        self.vocabulary_frequency = Counter()
        self._initialize_data()

    def get_metadata(self) -> dict:
        super_metadata = super(LanguageTaggingDataset, self).get_metadata()
        metadata = {
            "use_cbow_word_representation": self.use_cbow_word_representation,
            "use_lstm_c_word_representation": self.use_lstm_c_word_representation,
            "subword_len": self.subword_len,
            "cbow_subwords_prefix": self.cbow_subwords_prefix,
            "cbow_subwords_postfix": self.cbow_subwords_postfix,
            "cbow_word2subword_prefix_idx": self.cbow_word2subword_prefix_idx,
            "cbow_word2subword_postfix_idx": self.cbow_word2subword_postfix_idx,
            "char_vocabulary": self.char_vocabulary,
            "char_vocabulary_size": self.char_vocabulary_size,
            "char2idx": self.char2idx,
            "idx2char": self.idx2char,
            "word_idx2chars_idx": self.word_idx2chars_idx,
            "padding_char_index": self.padding_char_index,
            "unknown_char_index": self.unknown_char_index,
            "vocabulary_frequency": self.vocabulary_frequency
        }
        return {**super_metadata, **metadata}

    def _set_metadata(self, metadata: dict) -> None:
        super(LanguageTaggingDataset, self)._set_metadata(metadata)
        self.use_cbow_word_representation = metadata["use_cbow_word_representation"]
        self.use_lstm_c_word_representation = metadata["use_lstm_c_word_representation"]
        self.subword_len = metadata["subword_len"]
        self.cbow_subwords_prefix = metadata["cbow_subwords_prefix"]
        self.cbow_subwords_postfix = metadata["cbow_subwords_postfix"]
        self.cbow_word2subword_prefix_idx = metadata["cbow_word2subword_prefix_idx"]
        self.cbow_word2subword_postfix_idx = metadata["cbow_word2subword_postfix_idx"]
        self.char_vocabulary = metadata["char_vocabulary"]
        self.char_vocabulary_size = metadata["char_vocabulary_size"]
        self.char2idx = metadata["char2idx"]
        self.idx2char = metadata["idx2char"]
        self.word_idx2chars_idx = metadata["word_idx2chars_idx"]
        self.padding_char_index = metadata["padding_char_index"]
        self.unknown_char_index = metadata["unknown_char_index"]
        self.vocabulary_frequency = metadata["vocabulary_frequency"]

    def _initialize_data(self) -> None:
        """
        @brief: Initializes the data.

        @return: None
        """
        data_dtf = pd.read_table(self.data_file_path, sep=self.separator, skip_blank_lines=False, header=None,
                                 keep_default_na=False, na_filter=True, quoting=3, names=["word", "label"])
        data_dtf.word = data_dtf.word.apply(self._process_token)

        # Convert to sentences:
        indexes = np.where((data_dtf.word.values == '') == True)[0].tolist()
        sentences = np.split(data_dtf.word.values, indexes)
        sentences = [sentence.tolist()
                     if len(sentence.tolist()) > 0 and sentence.tolist()[0] != ''
                     else sentence.tolist()[1:]
                     for sentence in sentences]
        self.max_sentence_length = max([len(sentence) for sentence in sentences])

        indexes = np.where((data_dtf.word.values == '') == True)[0].tolist()  # In test data, there are no labels
        labels = np.split(data_dtf.label.values, indexes)
        labels = [sentence_labels.tolist()
                  if len(sentence_labels.tolist()) > 0 and sentence_labels.tolist()[0] != ''
                  else sentence_labels.tolist()[1:]
                  for sentence_labels in labels]

        # Remove empty sentences:
        data = [(X, y) for X, y in zip(sentences, labels) if len(X) > 0]
        self.X, self.y = zip(*data)

        assert len(self.X) == len(self.y)

    def _is_number(self, string: str) -> bool:
        """
        @brief: Returns whether the given string is a number.

        @param string: The string to check.

        @return: Whether the given string is a number.
        """
        try:
            float(str(string.replace(",", "")))
            return NUM_TOKEN
        except ValueError:
            return string

    def _is_date(self, string: str) -> bool:
        """
        @brief: Returns whether the given string is a date.

        @param string: The string to check.

        @return: Whether the given string is a date.
        """
        try:
            found_date = re.search(r'(\d+.\d+.\d+)|(\d+/\d+/\d+)', string)
            if found_date is not None:
                return DATE_TOKEN
            else:
                return string
        except ValueError:
            return string

    def _process_token(self, token: str) -> str:
        """
        @brief: Processes the token.

        @param token: The token to preprocess.

        @return: The preprocessed token.
        """
        for processor in self._token_processors:
            token = processor(token)
        return token

    def _get_padded_chars_indexes(self, word: str) -> torch.Tensor:
        """
        @brief: Return padded word characters indexes tensor.

        This method returns characters indexes of the @ref word padded
        with @ref self.padding_index to the maximum word size.

        @param word: The word to get padded characters for.

        @return: The padded word characters indexes tensor.
        """
        padding = [PADDING_CHAR] * (self.max_word_length - len(word))
        padded_word_chars = [char for char in word] + padding
        padded_word_chars_indexes = [self.char2idx[char] for char in padded_word_chars]

        return torch.tensor(padded_word_chars_indexes[:MAX_WORD_SIZE], dtype=torch.int64, device=self.device)

    def _get_chars_indexes(self, word: str) -> torch.Tensor:
        """
        @brief: Return word characters indexes tensor.

        This method returns characters indexes of the @ref word.

        @param word: The word to get characters for.

        @return: The word characters indexes tensor.
        """
        word_chars_indexes = [self.char2idx[char] for char in word]
        return torch.tensor(word_chars_indexes, dtype=torch.int64, device=self.device)

    def _initialize_labels_metadata(self):
        self.labels = set(list(chain.from_iterable(self.y)) + [PADDING_LABEL])
        self.idx2label = dict(enumerate(self.labels))
        self.label2idx = {label: index for index, label in self.idx2label.items()}
        self.num_of_labels = len(self.idx2label)
        self.padding_label_index = self.label2idx[PADDING_LABEL]

    def _initialize_words_metadata(self):
        self.tokens = set(list(chain.from_iterable(self.X)))
        self.vocabulary = set(self.tokens)
        self.vocabulary.update(SPECIAL_TOKENS)
        self.vocabulary_frequency = Counter(list(chain.from_iterable(self.X)) + SPECIAL_TOKENS)

        # Create the metadata needed for CBOW/LSTM_C word representation - part 1:
        if self.use_lstm_c_word_representation is True:
            self.char_vocabulary = set(list(chain.from_iterable(self.vocabulary)) + [PADDING_CHAR, UNKNOWN_CHAR])
            self.char2idx = {character: index for index, character in enumerate(self.char_vocabulary)}
            self.idx2char = {index: character for index, character in enumerate(self.char_vocabulary)}
            self.max_word_length = MAX_WORD_SIZE
            self.char_vocabulary_size = len(self.char2idx)
            self.padding_char_index = self.char2idx[PADDING_CHAR]
            self.unknown_char_index = self.char2idx[UNKNOWN_CHAR]
        elif self.use_cbow_word_representation is True:
            self.cbow_subwords_prefix = {word: word[:self.subword_len] for word in self.vocabulary}
            self.cbow_subwords_postfix = {word: word[-self.subword_len:] for word in self.vocabulary}
            assert len(self.cbow_subwords_prefix) == len(self.cbow_subwords_postfix) == len(self.vocabulary)
            self.vocabulary.update(self.cbow_subwords_prefix.values())
            self.vocabulary.update(self.cbow_subwords_postfix.values())

        self.idx2word = {index: word for index, word in enumerate(self.vocabulary)}
        self.word2idx = {word: index for index, word in self.idx2word.items()}
        self.vocabulary_size = len(self.idx2word)

        # Create the metadata needed for CBOW/LSTM_C word representation - part 2:
        if self.use_lstm_c_word_representation is True:
            self.word_idx2chars_idx = {index: self._get_chars_indexes(word) for index, word in self.idx2word.items()}
        elif self.use_cbow_word_representation is True:
            self.cbow_word2subword_prefix_idx = {self.word2idx[word]: self.word2idx[prefix]
                                                 for word, prefix in self.cbow_subwords_prefix.items()}
            self.cbow_word2subword_postfix_idx = {self.word2idx[word]: self.word2idx[postfix]
                                                  for word, postfix in self.cbow_subwords_postfix.items()}
            subwords_only = self.vocabulary.difference(self.tokens)
            for word in subwords_only:
                word_index = self.word2idx[word]
                self.cbow_word2subword_prefix_idx[word_index] = word_index
                self.cbow_word2subword_postfix_idx[word_index] = word_index

        self.unknown_token_index = self.word2idx[UNKNOWN_TOKEN]
        self.padding_token_index = self.word2idx[PADDING_TOKEN]

    def _transform_data_to_indexes(self) -> None:
        def transform_X_to_indexes(sentence: list) -> list:
            sentence = [self.word2idx.get(word, self.unknown_token_index) for word in sentence]
            return sentence

        def transform_y_to_indexes(labels: list) -> list:
            labels = [self.label2idx.get(label, EMPTY_LABEL_INDEX) for label in labels]
            return labels

        self.X = np.vectorize(transform_X_to_indexes, otypes=[list])(self.X).tolist()
        self.y = np.vectorize(transform_y_to_indexes, otypes=[list])(self.y).tolist()


def collate_batch(batch: List[Tuple[List[int], List[int]]],
                  device: str, word_padding_value: int,
                  label_padding_value: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    @brief: Collates the batch.

    @param batch: The batch to collate.
    @param device: The device to use.
    @param word_padding_value: The padding value for words.
    @param label_padding_value: The padding value for labels.

    @return: The collated batch.
    """
    sentences, labels = zip(*batch)
    padded_sentences = torch.nn.utils.rnn.pad_sequence(
        sequences=sentences, batch_first=True, padding_value=word_padding_value).to(device)
    padded_labels = torch.nn.utils.rnn.pad_sequence(
        sequences=labels, batch_first=True, padding_value=label_padding_value).to(device)

    return padded_sentences, padded_labels


def seed_everything(seed: int) -> None:
    """
    @brief: Set random seed for all random operations.

    @param seed: The seed to use.

    @return: None.
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ":4096:8"
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    GLOBAL_RAND_GEN.manual_seed(SEED)
    torch.use_deterministic_algorithms(False)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int) -> None:
    """
    @brief: Seeds the worker of DataLoader.

    @param worker_id: The id of the worker.

    @return: None.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def random_split(dataset: Dataset, dev_ration: float) -> Tuple[Dataset, Dataset]:
    """
    @brief: Randomly splits the dataset into two parts.

    @param dataset: The dataset to split.
    @param dev_ration: The ratio of the dev set.

    @return: The two splits.
    """
    train_size = int(len(dataset) * (1 - dev_ration))
    dev_size = int(len(dataset) - train_size)
    lengths = [train_size, dev_size]
    temp_train_dataset, temp_dev_dataset = torch.utils.data.random_split(dataset, lengths, generator=GLOBAL_RAND_GEN)
    train_dataset = copy.deepcopy(dataset)
    dev_dataset = copy.deepcopy(dataset)
    train_dataset.X = np.array(train_dataset.X, dtype=object)[temp_train_dataset.indices].tolist()
    train_dataset.y = np.array(train_dataset.y, dtype=object)[temp_train_dataset.indices].tolist()
    dev_dataset.X = np.array(dev_dataset.X, dtype=object)[temp_dev_dataset.indices].tolist()
    dev_dataset.y = np.array(dev_dataset.y, dtype=object)[temp_dev_dataset.indices].tolist()

    return train_dataset, dev_dataset


def load_train_dataset(dataset_type: str, device: str, dataset_filename: str, batch_size: int = 1,
                       num_workers: int = 4, shuffle: bool = True,
                       collate_fn: object = None, dev_ratio: int = 0.1,
                       word_representation: str = "") -> Tuple[DataLoader, DataLoader]:
    """
    @brief: Loads the training dataset.

    @param dataset_type: The type of the dataset to load.
    @param device: The device to load the dataset to.
    @param dataset_filename: The filename of the dataset to load.
    @param batch_size: The batch size of the dataset.
    @param num_workers: The number of workers to use for loading the dataset.
    @param shuffle: Whether to shuffle the train dataset.
    @param collate_fn: The collate batch function to use.
    @param dev_ratio: The ratio of the dev set.
    @param word_representation: The type of word representation to use, see @ref WORD_REPRESENTATION_TYPES_CLI.

    @return: The train and test data loaders.
    """
    print("Loading train dataset...")
    separator = DS_TO_SEPARATOR[dataset_type]
    prefetch_factor = 2
    persistent_workers = True

    if dataset_type == POS_NEG_TAGGING:
        dataset = SequenceClassificationDataset(
            device=device, filename=dataset_filename, separator=separator)
    elif dataset_type in [POS_TAGGING, NER_TAGGING]:
        dataset = LanguageTaggingDataset(
            device=device, filename=dataset_filename, separator=separator, word_representation=word_representation)

    train_dataset, dev_dataset = random_split(dataset, dev_ratio)
    train_dataset.initialize()
    dev_dataset.initialize(metadata=train_dataset.get_metadata())

    if dataset_type in [POS_TAGGING, NER_TAGGING]:
        collate_fn = partial(
            collate_fn, device=device, word_padding_value=train_dataset.padding_token_index,
            label_padding_value=train_dataset.padding_label_index)

    train_data_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, prefetch_factor=prefetch_factor,
        persistent_workers=persistent_workers, worker_init_fn=seed_worker,
        generator=GLOBAL_RAND_GEN, collate_fn=collate_fn)
    dev_data_loader = DataLoader(
        dev_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, prefetch_factor=prefetch_factor,
        persistent_workers=persistent_workers, worker_init_fn=seed_worker,
        generator=GLOBAL_RAND_GEN, collate_fn=collate_fn)

    return train_data_loader, dev_data_loader


def load_test_dataset(dataset_type: str, device: str, test_file: str, batch_size: int = 1,
                      num_workers: int = 4, metadata: dict = None, word_representation: str = "",
                      collate_fn: object = None) -> DataLoader:
    """
    @brief: Loads the test dataset.

    @param dataset_type: The type of the dataset to load.
    @param device: The device to load the dataset to.
    @param test_file: The test file dataset.
    @param batch_size: The batch size of the dataset.
    @param num_workers: The number of workers to load the dataset.
    @param metadata: The metadata of the dataset.
    @param word_representation: The type of word representation to use, see @ref WORD_REPRESENTATION_TYPES_CLI.
    @param collate_fn: The collate batch function to use.

    @return: The test data loader.
    """
    print("Loading test dataset...")

    separator = DS_TO_SEPARATOR[dataset_type]
    prefetch_factor = 2
    persistent_workers = True

    if dataset_type == POS_NEG_TAGGING:
        test_dataset = SequenceClassificationDataset(
            device=device, filename=test_file, separator=separator)
    elif dataset_type in [POS_TAGGING, NER_TAGGING]:
        test_dataset = LanguageTaggingDataset(
            device=device, filename=test_file, separator=separator, word_representation=word_representation)

    test_dataset.initialize(metadata=metadata)

    if dataset_type in [POS_TAGGING, NER_TAGGING]:
        collate_fn = partial(
            collate_fn, device=device, word_padding_value=test_dataset.padding_token_index,
            label_padding_value=test_dataset.padding_label_index)

    test_data_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, prefetch_factor=prefetch_factor,
        persistent_workers=persistent_workers, worker_init_fn=seed_worker,
        generator=GLOBAL_RAND_GEN, collate_fn=collate_fn)

    return test_data_loader


def get_device(log: bool = True) -> str:
    """
    @brief: Returns the device to use.

    @param log: Whether to log the device.

    @return: The device to use.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if log is True:
        print(f"Using {device.upper()} device")

    return device


def train(train_dataloader: DataLoader, dev_dataloader: DataLoader, model: torch.nn.Module,
          optimizer: torch.optim, gradient_clip: int = 0, log: bool = False) -> Tuple[float, float]:
    """
    @brief: Trains the model.

    @param train_dataloader: The train dataloader.
    @param dev_dataloader: The dev dataloader.
    @param model: The model to train.
    @param optimizer: The optimizer to use.
    @param gradient_clip: The gradient clip value.
    @param log: Whether to log the training progress.

    @return: Average train loss and accuracy tuple.
    """
    model.train()
    previous_log_reminder = 0
    samples_log_interval = 500
    predictions_counter = 0
    samples_counter = 0
    total_loss = 0
    correct = 0
    label_padding_index = train_dataloader.dataset.padding_label_index
    dev_losses, dev_accuracies = [], []
    dummy_tensor = torch.ones(1, dtype=torch.float32, requires_grad=True)

    def model_forward(X):
        return checkpoint(model.forward, X, dummy_tensor) if model.device == "cuda" else model(X)

    for X, y in train_dataloader:
        num_of_samples = len(y)

        # Forward pass:
        y_prob = model_forward(X)
        y_pred = model.predict_from_probabilities(y_prob)

        # Compute prediction error:
        loss = model.loss(y_pred=y_prob, y_true=y)
        total_loss += loss.sum().item()
        correct += ((y_pred == y) & (y != label_padding_index)).type(torch.int64).sum().item()

        # Backward pass:
        optimizer.zero_grad()
        loss.backward()
        if gradient_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip)
        optimizer.step()

        predictions_counter += (y != label_padding_index).sum().item()

        # Free memory:
        del X, y, y_prob, y_pred, loss
        torch.cuda.empty_cache()

        # Log progress on dev each samples_log_interval samples:
        current_log_reminder = samples_counter - (samples_counter % samples_log_interval)
        log_iteration = (current_log_reminder % samples_log_interval == 0)
        if log is True and log_iteration is True and previous_log_reminder != current_log_reminder:
            start_timestamp = get_time_now()
            dev_loss, dev_accuracy = test(dev_dataloader, model)
            end_timestamp = get_time_now()
            log_samples = samples_counter - previous_log_reminder
            dev_losses.append((dev_loss, log_samples))
            dev_accuracies.append((dev_accuracy, log_samples))
            test_duration = end_timestamp - start_timestamp
            duration_log = f" | Duration: {test_duration}"
            previous_log_reminder = current_log_reminder
            train_sample_counter_log = 5 * " " + \
                f"[{samples_counter:>5d}/{len(train_dataloader.dataset):>5d}] -------->"
            dev_error_log = f" Dev => Accuracy: {(dev_accuracy):>0.4f} [%] ; Loss: {dev_loss:>6f}"
            log_message = f"{train_sample_counter_log} {dev_error_log} {duration_log}"
            print(log_message)
            model.train()

        samples_counter += num_of_samples

    total_avg_loss = float(total_loss / predictions_counter)
    total_avg_accuracy = float((correct / predictions_counter) * 100)

    return total_avg_loss, total_avg_accuracy, dev_losses, dev_accuracies


def test(dataloader: DataLoader, model: torch.nn.Module) -> Tuple[float, float]:
    """
    @brief: Tests the model.

    @param dataloader: The dataloader to use.
    @param model: The model to test.

    @return: Average test loss and accuracy tuple.
    """
    model.eval()
    predictions_counter = 0
    total_loss = 0
    correct = 0
    label_padding_index = dataloader.dataset.padding_label_index

    with torch.no_grad():
        for X, y in dataloader:
            # Forward pass:
            y_prob = model(X)
            y_pred = model.predict_from_probabilities(y_prob)

            # Compute prediction error:
            loss = model.loss(y_pred=y_prob, y_true=y)
            total_loss += loss.sum().item()

            # Compute accuracy:
            correct += ((y_pred == y) & (y != label_padding_index)).type(torch.int64).sum().item()
            predictions_counter += (y != label_padding_index).sum().item()

            # Free memory:
            del X, y, y_prob, y_pred, loss
            torch.cuda.empty_cache()

    total_avg_loss = float(total_loss / predictions_counter)
    total_avg_accuracy = float((correct / predictions_counter) * 100)

    return total_avg_loss, total_avg_accuracy


def fit(train_data_loader:  DataLoader, dev_data_loader: DataLoader, model: torch.nn.Module,
        optimizer: torch.optim, gradient_clip: int = 0, scheduler: torch.optim.lr_scheduler = None,
        epochs: int = 10, log_train: bool = False,
        log_error: bool = True) -> Tuple[torch.nn.Module, Tuple[list, list, list, list]]:
    """
    @brief: Fits the model.

    @param train_data_loader: The train dataloader to use.
    @param dev_data_loader: The dev dataloader to use.
    @param model: The model to fit.
    @param optimizer: The optimizer to use.
    @param gradient_clip: The gradient clip value.
    @param scheduler: The scheduler of learning rate adjustments to use.
    @param epochs: The number of epochs to fit.
    @param log_train: Whether to log the training progress.
    @param log_error: Whether to log the error of train/dev progress.

    @return: The trained model and the train/dev loss/accuracy tuple.
    """
    print("Training...")

    total_start_timestamp = get_time_now()
    total_train_losses, total_train_accuracies = [], []
    total_dev_losses, total_dev_accuracies = [], []
    total_train_dev_loss, total_train_dev_accuracy = [], []

    if log_train is True:
        print("-" * 130)

    for epoch in range(epochs):
        if log_train is True:
            print(f"Epoch - {epoch + 1}/{epochs} Dev statistics:")
            print("-" * 130)
        start_timestamp = get_time_now()
        train_loss, train_accuracy, dev_loss_list, dev_accuracy_list = train(
            train_data_loader, dev_data_loader, model, optimizer, gradient_clip, log_train)
        dev_loss, dev_accuracy = test(dev_data_loader, model)
        end_timestamp = get_time_now()

        total_train_losses.append(train_loss)
        total_train_accuracies.append(train_accuracy)
        total_dev_losses.append(dev_loss)
        total_dev_accuracies.append(dev_accuracy)
        total_train_dev_loss += dev_loss_list
        total_train_dev_accuracy += dev_accuracy_list
        epoch_duration = end_timestamp - start_timestamp

        log_message = f"Epoch {epoch + 1}/{epochs}"
        if log_error is True:
            train_error_log = f": Train => Accuracy: {(train_accuracy):>0.4f} [%] ; Loss: {train_loss:>6f}"
            dev_error_log = f" | Dev => Accuracy: {(dev_accuracy):>0.4f} [%] ; Loss: {dev_loss:>6f}"
            duration_log = f" | Duration: {epoch_duration}"
            log_message += train_error_log + dev_error_log + duration_log
        print(log_message)

        if log_train is True:
            print("-" * 130)

        if scheduler is not None:
            scheduler.step()

    total_end_timestamp = get_time_now()
    total_duration = total_end_timestamp - total_start_timestamp
    print(f"Done training!\nTotal duration: {total_duration}")

    if log_error is False:
        statistics = (total_train_losses, total_train_accuracies, total_dev_losses, total_dev_accuracies)
    else:
        statistics = (total_train_dev_loss, total_train_dev_accuracy)

    return model, statistics


def predict(tag_task: str, test_data_loader: DataLoader, model: torch.nn.Module, predict_file: str = "") -> None:
    """
    @brief: Predict test dataset on using the model.

    This method is used to evaluate the model on the test/dev set,
    and optionally to save the predictions to a file.

    @param tag_task: The type of the tagging task.
    @param test_data_loader: The test dataloader to use.
    @param model: The model to evaluate.
    @param predict_file: The path to the predictions file.

    @return: None.
    """
    def write_word_predictions(X: torch.Tensor, y_pred: torch.Tensor) -> None:
        with open(predict_file, "a") as file:
            for sentence_prediction in zip(X.tolist(), y_pred.tolist()):
                sentence_predictions_labels = [
                    test_data_loader.dataset.idx2label[prediction_index]
                    for word_index, prediction_index in zip(*sentence_prediction)
                    if word_index != test_data_loader.dataset.padding_token_index]
                sentence_labels = '\n'.join(sentence_predictions_labels) + '\n'
                file.write(f"{sentence_labels}\n")

    def write_sentence_predictions(X: torch.Tensor, y_pred: torch.Tensor) -> None:
        with open(predict_file, "a") as file:
            for prediction_index in y_pred.tolist():
                label = test_data_loader.dataset.idx2label[prediction_index]
                file.write(f"{label}\n")

    def compute_word_correct_predictions(y: torch.Tensor, y_pred: torch.Tensor) -> int:
        return ((y_pred == y) & (y != test_data_loader.dataset.padding_label_index)).type(torch.int64).sum().item()

    def compute_sentence_correct_predictions(y: torch.Tensor, y_pred: torch.Tensor) -> int:
        return ((y_pred == y)).type(torch.int64).sum().item()

    print("Evaluating...")
    if predict_file != "":
        open(predict_file, "w").close()

    model.eval()
    write_prediction_cb = write_sentence_predictions if tag_task == POS_NEG_TAGGING else write_word_predictions
    compute_correct_predictions_cb = compute_sentence_correct_predictions if tag_task == POS_NEG_TAGGING else \
        compute_word_correct_predictions
    correct = 0
    size = 0

    with torch.no_grad():
        for X, y in test_data_loader:
            # Predict:
            y_pred = model.predict(X)

            # Compute prediction error:
            correct += compute_correct_predictions_cb(y, y_pred)

            # Write predictions:
            if predict_file != "":
                write_prediction_cb(X, y_pred)

            size += (y != test_data_loader.dataset.padding_label_index).sum().item()

            # Free memory:
            del X, y, y_pred
            torch.cuda.empty_cache()

    print("Done evaluating!")
    avg_accuracy = float((correct / size) * 100)
    return avg_accuracy


def save_overall_plot_statistics(tag_task: str, statistics: Tuple[list, list, list, list], plot_path: str = ".") -> None:
    """
    @brief: Saves fit statistics plots.

    @param tag_task: The type of the tag.
    @param statistics: The fit statistics to plot.
    @param plot_path: The path to the directory to save the plots.

    @return: None.
    """
    print("Saving plots...")

    train_losses, train_accuracies, dev_losses, dev_accuracies = statistics
    epochs = range(1, len(train_losses) + 1)

    plt.ioff()

    def plot_helper(stat: str, train_stats: list, dev_stats: list) -> None:
        plt.plot(epochs, train_stats, label=f"Train {stat}")
        plt.plot(epochs, dev_stats, label=f"Dev {stat}")
        plt.legend()
        plt.xlabel("Epochs")
        plt.ylabel(f"{stat}")
        plt.title(f"{tag_task.upper()} Training and Dev {stat}")
        plt.savefig(f"{plot_path}/{tag_task}_train_dev_{stat.lower()}.png")
        plt.close()

    plot_helper("Loss", train_losses, dev_losses)
    plot_helper("Accuracy", train_accuracies, dev_accuracies)

    print("Done saving plots!")


def save_model(model_path: str, model: torch.nn.Module, metadata: dict, fit_statistics: dict) -> None:
    """
    @brief: Saves model to file.

    @param model_path: The path to the file to save the model.
    @param model: The model to save.
    @param metadata: The dataset metadata to save.
    @param fit_statistics: The fit statistics to save.

    @return: None.
    """
    print("Saving model...")

    torch.save({
        "model": model.state_dict(),
        "model_non_trainable_parameters": model.get_non_trainable_parameters_state_dict(),
        "dataset_metadata": metadata,
        "fit_statistics": fit_statistics},
        model_path)

    print("Done saving model!")


def load_model(device: str, model_path: str, model_class: type) -> Tuple[torch.nn.Module, dict]:
    """
    @brief: Loads model from file.

    @param device: The device to load the model on.
    @param model_path: The path to the file to load the model from.
    @param model_class: The class of the model to load.

    @return: The loaded model and the dataset metadata.
    """
    print("Loading model...")

    checkpoint = torch.load(model_path, map_location=torch.device(device))
    model = model_class(device=device, **checkpoint["model_non_trainable_parameters"])
    model.load_state_dict(checkpoint["model"])
    dataset_metadata = checkpoint["dataset_metadata"]
    fit_statistics = checkpoint["fit_statistics"]

    print("Done loading model!")
    return model, dataset_metadata, fit_statistics


def parse_cli(description: str = "", bilstm_train: bool = False, bilstm_predict: bool = False) -> argparse.Namespace:
    """
    @brief: Parses the command line arguments.

    @param description: The description of the program.
    @param bilstm_train: bilstm train module usage.
    @param bilstm_predict: bilstm predict module usage.

    @return: The parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(description=description)
    model_fit_group = parser.add_mutually_exclusive_group()

    parser.add_argument("--tag-task", type=str, default=NER_TAGGING, choices=[POS_NEG_TAGGING, NER_TAGGING, POS_TAGGING],
                        help="The language tagging task.")
    parser.add_argument("--num-workers", type=int, default=2,
                        help="The number of workers to use for data loading.")

    if bilstm_predict is False:
        model_fit_group.add_argument("--fit", action="store_true", default=True,
                                     help="Fit the model.")

    model_fit_group.add_argument("--load-model-file", type=str, default="",
                                 help="The path to load a trained model from.")

    if bilstm_predict is False:
        parser.add_argument("--save-model-file", type=str, default="",
                            help="The path to save the trained model to.")

    if bilstm_train is False:
        parser.add_argument("--predict", action="store_true", default=False,
                            help="Predict for the input file.")

    parser.add_argument("--debug", action="store_true", default=False,
                        help="Run in debug mode, set parameters in the module file.")

    if bilstm_predict is False:
        parser.add_argument("--train-file", type=str, default="train",
                            help="The name of the train file contains the entire "
                                 "train dataset, this will be splitted to train/dev.")

    if bilstm_predict is False:
        parser.add_argument("--plot-path", type=str, default="",
                            help="Path to save plots of fitting statistics.")

    if bilstm_train is False:
        parser.add_argument("--test-file", type=str, default="",
                            help="The path to the test dataset file.")
        parser.add_argument("--predict-file", type=str, default="",
                            help="The path to the predictions file.")

    if bilstm_predict is False:
        parser.add_argument("--dev-ratio", type=float, default=0.1,
                            help="The ratio of the dev dataset from the train dataset.")
        parser.add_argument("--epochs", type=int, default=300,
                            help="The number of epochs to fit.")
        parser.add_argument("--embedding-dim", type=int, default=50,
                            help="The size of the word embedding vectors.")

    if bilstm_train is False and bilstm_predict is False:
        parser.add_argument("--mlp-hidden-dim", type=int, default=128,
                            help="The hidden layer dimension of the MLP layer.")

    if bilstm_train is True or bilstm_predict is True:
        parser.add_argument("--word-representation", type=str, default=WORD_REPR_EMBEDDING,
                            choices=WORD_REPRESENTATION_TYPES_CLI,
                            help=f"The word representation type, following types available: {WORD_REPRESENTATION_TYPES}.")

    if bilstm_predict is False:
        parser.add_argument("--lstm-hidden-dim", type=int, default=128,
                            help="The hidden layer dimension of the LSTM layer.")

    parser.add_argument("--batch-size", type=int, default=128,
                        help="The batch size to use.")

    if bilstm_predict is False:
        parser.add_argument("--log-train", action="store_true", default=False,
                            help="Print logs of the training progress.")
        parser.add_argument("--log-error", action="store_true", default=True,
                            help="Print logs of the train/dev progress error.")
        parser.add_argument("--lr", type=float, default=0.003,
                            help="The learning rate to use.")
        parser.add_argument("--sched-step", type=int, default=10,
                            help="The step of the learning rate scheduler to use.")
        parser.add_argument("--sched-gamma", type=float, default=1,
                            help="The gamma of the learning rate scheduler to use.")
        parser.add_argument("--dropout", type=float, default=0,
                            help="The dropout probability for linear/MLP layers to use.")
        parser.add_argument("--weight-decay", type=float, default=0,
                            help="The weight decay to use.")

    arguments, positional_arguments = parser.parse_known_args()

    if bilstm_train is True and len(positional_arguments) == 3:
        word_representation, train_file, model_file = positional_arguments
        arguments.fit = True
        arguments.word_representation = word_representation
        arguments.train_file = train_file
        arguments.save_model_file = model_file
    elif bilstm_predict is True and len(positional_arguments) == 3:
        arguments.predict = True
        word_representation, model_file, input_file = positional_arguments
        arguments.word_representation = word_representation
        arguments.load_model_file = model_file
        arguments.test_file = input_file

    return arguments
