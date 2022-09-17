"""
PyTorch tagging module for language classification tagging assignment.
The module supplies utilities used to the POS (Part of Speech) and the NER (Named Entity Recognition) tagging tasks.

Author: Simon Raviv.
"""
import os
import argparse
import random
import re
import numpy as np
import pandas as pd
import torch
import math

from typing import Tuple
from collections import Counter
from matplotlib import pyplot as plt
from torch.utils.data import Dataset, DataLoader
from itertools import chain

try:
    from sklearn.manifold import TSNE
except ImportError:
    TSNE = None
    print("sklearn not installed, visualizing CNN filters is not supported.")


# Global variables:
SEED = 2022
PADDING_TOKEN = "<PADDING>"
UNKNOWN_TOKEN = "<UNKNOWN>"
PADDING_CHAR = "<PAD>"
UNKNOWN_CHAR = "<UNK>"
NUM_TOKEN = "<NUMBER>"
DATE_TOKEN = "<DATE>"
EMPTY_TOKEN = ""
NER_SEPERATOR = "\t"
POS_SEPERATOR = " "
NER_COMMON_TAG = "O"
NER_TAGGING = "ner"
POS_TAGGING = "pos"
CBOW_SUBWORD_EMBEDDING = "cbow"
CNN_SUBWORD_EMBEDDING = "cnn"
SPECIAL_TOKENS = [PADDING_TOKEN, UNKNOWN_TOKEN, NUM_TOKEN, DATE_TOKEN, EMPTY_TOKEN]
GLOBAL_RAND_GEN = torch.Generator()
MAX_WORD_SIZE = 20


class LanguageTaggingDataset(Dataset):
    """
    @brief: Language tagging dataset.
    """

    def __init__(self, device: str, root_dir: str, filename: str, separator: str, window_size: int = 5,
                 metadata: dict = None, use_pretrained_embedding: bool = False,
                 pretrained_vocabulary: np.ndarray = None, pretrained_embedding: torch.Tensor = None,
                 subword_embedding: str = "") -> None:
        """
        @brief: Initialize the dataset.

        @param device: The device to use.
        @param root_dir: The root directory of the dataset.
        @param filename: The filename of the dataset.
        @param separator: The separator of between the word to the label.
        @param window_size: The window size of the context.
        @param metadata: The metadata of the dataset.
        @param pretrained_vocabulary: The vocabulary of the pretrained embedding.
        @param pretrained_embedding: The pretrained embedding of the dataset.
        @param subword_embedding: Type of sub-word embedding to use, cbow/cnn.

        @return: None
        """
        super(LanguageTaggingDataset).__init__()
        self.device = device
        self.root_dir = root_dir
        self.filename = filename
        self.separator = separator
        self.window_size = window_size
        self.data_file_path = os.path.join(self.root_dir, self.filename)
        self.use_pretrained_embedding = use_pretrained_embedding
        self.pretrained_vocabulary = pretrained_vocabulary
        self.pretrained_embedding = pretrained_embedding
        self.pretrained_embedding_weights = None
        self.use_cbow_subword_embedding = True if subword_embedding == CBOW_SUBWORD_EMBEDDING else False
        self.use_cnn_subword_embedding = True if subword_embedding == CNN_SUBWORD_EMBEDDING else False
        self.X = []
        self.y = []
        self.padded_X = []
        self.padded_y = []
        self.vocabulary = set()
        self.vocabulary_size = 0
        self.idx2word = {}
        self.word2idx = {}
        self.labels = set()
        self.idx2label = {}
        self.label2idx = {}
        self.num_of_labels = 0
        self.original2padded = {}
        self._token_processors = [self._is_number, self._is_date]
        self.subword_len = 3
        self.cbow_subwords_prefix = []
        self.cbow_subwords_postfix = []
        self.cbow_word2subword_prefix_idx = {}
        self.cbow_word2subword_postfix_idx = {}
        self.char_vocabulary = set()
        self.char2idx = {}
        self.idx2char = {}
        self.max_word_length = 0
        self.word_idx2chars_idx = None
        self.char_vocabulary_size = 0
        self.char_padding_index = -1
        self.padding_token_index = -1
        self.vocabulary_frequency = Counter()
        self._initialize_data()
        self._initialize_metadata(metadata)
        self._create_pretrained_embedding_weights()
        self._transform_data_to_indices()
        self.padding_token_index = self.word2idx[PADDING_TOKEN]

    def __len__(self) -> int:
        """
        @brief: Returns the length of the dataset.

        @return: The length of the dataset.
        """
        return len(self.X)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        @brief: Returns the word and label at the given index with the context.

        @param index: The index of the word and label to return.

        @return: The word and label at the given index with the context.
        """
        word = self.X[index]
        label = self.y[index]

        # Get the context of the word:
        padded_index = self.original2padded[index]
        word_with_context = self.padded_X[padded_index - self.window_size//2:padded_index + self.window_size//2 + 1]
        word_with_context[self.window_size//2] = word

        word_with_context = torch.tensor(word_with_context, dtype=torch.long)
        label = torch.tensor(label, dtype=torch.long)

        return word_with_context, label

    def _initialize_data(self) -> None:
        """
        @brief: Initializes the data.

        @return: None
        """

        def create_padded_data(word: str, label: str) -> None:
            nonlocal original_index, padded_index, padding_size
            if word == "":
                word, label = [PADDING_TOKEN] * padding_size, [PADDING_TOKEN] * padding_size
                padded_index += padding_size
            else:
                word, label = [word], [label]
                self.original2padded[original_index] = padded_index
                original_index += 1
                padded_index += 1

            self.padded_X.extend(word)
            self.padded_y.extend(label)

            return None, None

        data_dtf = pd.read_table(self.data_file_path, sep=self.separator,
                                 skip_blank_lines=False, header=None,
                                 keep_default_na=False, na_filter=True,
                                 quoting=3, names=["word", "label"])
        data_dtf.word = data_dtf.word.apply(self._process_token)
        padding_size = self.window_size // 2
        original_index = 0
        padded_index = padding_size
        self.X = data_dtf.word.values
        self.y = data_dtf.label.values
        self.padded_X = [PADDING_TOKEN] * padding_size
        self.padded_y = [PADDING_TOKEN] * padding_size
        np.vectorize(create_padded_data, otypes=[None, None])(self.X, self.y)
        self.padded_X += [PADDING_TOKEN] * padding_size
        self.padded_y += [PADDING_TOKEN] * padding_size
        self.X = data_dtf.word[data_dtf.word != ""].values
        # For test data, the label is empty, so use the word instead.
        self.y = data_dtf.label[data_dtf.word != ""].values

        assert len(self.X) == len(self.y)
        assert len(self.padded_X) == len(self.padded_y)

    def get_metadata(self) -> dict:
        """
        @brief: Returns the metadata of the dataset.

        @return: The metadata dictionary.
        """
        return {"idx2word": self.idx2word,
                "word2idx": self.word2idx,
                "idx2label": self.idx2label,
                "label2idx": self.label2idx,
                "vocabulary": self.vocabulary,
                "vocabulary_size": self.vocabulary_size,
                "num_of_labels": self.num_of_labels}

    def get_vocabulary(self, frequency: int = 5) -> list:
        """
        @brief: Returns the vocabulary of the dataset.

        @param frequency: The minimum frequency of words in the dataset.

        @return: The vocabulary.
        """
        vocabulary = [word for word, count in self.vocabulary_frequency.items() if count >= frequency]
        return vocabulary

    def _create_pretrained_embedding_weights(self) -> None:
        """
        @brief: Create the pretrained embedding weights matrix.

        @return: None
        """
        if self.pretrained_vocabulary is None or self.pretrained_embedding is None:
            return None

        pretrained_idx2word = dict(enumerate(self.pretrained_vocabulary))
        pretrained_word2idx = {value: key for key, value in pretrained_idx2word.items()}
        embedding_dim = self.pretrained_embedding.shape[1]
        self.pretrained_embedding_weights = torch.zeros((self.vocabulary_size, embedding_dim))
        pretrained_vocabulary_set = set(self.pretrained_vocabulary)

        for index, word in enumerate(self.vocabulary):
            if word in pretrained_vocabulary_set:
                self.pretrained_embedding_weights[index] = self.pretrained_embedding[pretrained_word2idx[word]]
            else:
                self.pretrained_embedding_weights[index] = torch.empty(embedding_dim).normal_()

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

    def _filter_rare_tokens(self, tokens: list, min_count: int, replace_by: str) -> set:
        """
        @brief: Filters the rare tokens.

        @param tokens: The tokens to filter.
        @param min_count: The minimum count of the tokens to keep.
        @param replace_by: The token to replace the rare tokens with.

        @return: The filtered tokens.
        """
        return set([replace_by if count < min_count else token for token, count in Counter(tokens).items()])

    def _get_padded_chars_indexes(self, word: str) -> torch.Tensor:
        """
        @brief: Return padded word characters indexes tensor.

        This method returns characters indexes of the @ref word padded
        on the sides with @ref self.padding_index to the maximum
        word size.

        @param word: The word to get padded characters for.

        @return: The padded word characters indexes tensor.
        """
        padded_word_chars = [PADDING_CHAR] * self.max_word_length
        padded_word_chars[self.max_word_length // 2 - len(word) // 2: self.max_word_length // 2 + len(word) // 2] = word
        padded_word_chars_indexes = [self.char2idx[char] for char in padded_word_chars]

        return torch.tensor(padded_word_chars_indexes[:MAX_WORD_SIZE], dtype=torch.long)

    def _initialize_words_metadata(self):
        """
        @brief: Initializes the words and characters metadata.

        @return: None
        """
        self.tokens = set(self.X)
        self.vocabulary = set(self.tokens)
        self.vocabulary.update(SPECIAL_TOKENS)
        self.vocabulary_frequency = Counter(list(self.X) + SPECIAL_TOKENS)

        # Create the metadata needed for CBOW/CNN sub-word embedding - part 1:
        if self.use_cnn_subword_embedding is True:
            self.char_vocabulary = set(chain.from_iterable(self.vocabulary))
            self.char2idx = {character: index for index, character in enumerate(self.char_vocabulary)}
            self.char2idx[PADDING_CHAR] = len(self.char_vocabulary)
            self.char2idx[UNKNOWN_CHAR] = len(self.char_vocabulary) + 1
            self.idx2char = {index: character for index, character in enumerate(self.char_vocabulary)}
            self.max_word_length = min(max(map(len, self.tokens)), MAX_WORD_SIZE)
            self.char_vocabulary_size = len(self.char2idx)
            self.char_padding_index = self.char2idx[PADDING_CHAR]
        elif self.use_cbow_subword_embedding is True:
            self.cbow_subwords_prefix = {word: word[:self.subword_len] for word in self.vocabulary}
            self.cbow_subwords_postfix = {word: word[-self.subword_len:] for word in self.vocabulary}
            assert len(self.cbow_subwords_prefix) == len(self.cbow_subwords_postfix) == len(self.vocabulary)
            self.vocabulary.update(self.cbow_subwords_prefix.values())
            self.vocabulary.update(self.cbow_subwords_postfix.values())

        self.idx2word = {index: word for index, word in enumerate(self.vocabulary)}
        self.word2idx = {word: index for index, word in self.idx2word.items()}
        self.vocabulary_size = len(self.idx2word)

        # Create the metadata needed for CBOW/CNN sub-word embedding - part 2:
        if self.use_cnn_subword_embedding is True:
            self.word_idx2chars_idx = torch.zeros(
                size=(self.vocabulary_size, self.max_word_length), dtype=torch.long).to(self.device)
            for index, word in self.idx2word.items():
                self.word_idx2chars_idx[index] = self._get_padded_chars_indexes(word)
        elif self.use_cbow_subword_embedding is True:
            self.cbow_word2subword_prefix_idx = {self.word2idx[word]: self.word2idx[prefix]
                                                 for word, prefix in self.cbow_subwords_prefix.items()}
            self.cbow_word2subword_postfix_idx = {self.word2idx[word]: self.word2idx[postfix]
                                                  for word, postfix in self.cbow_subwords_postfix.items()}
            subwords_only = self.vocabulary.difference(self.tokens)
            for word in subwords_only:
                word_index = self.word2idx[word]
                self.cbow_word2subword_prefix_idx[word_index] = word_index
                self.cbow_word2subword_postfix_idx[word_index] = word_index

    def _initialize_labels_metadata(self):
        """
        @brief: Initializes the labels metadata.

        @return: None
        """
        self.labels = set(self.y)
        self.idx2label = dict(enumerate(self.labels))
        self.label2idx = {label: index for index, label in self.idx2label.items()}
        self.num_of_labels = len(self.idx2label)

    def _initialize_metadata(self, metadata: dict) -> None:
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

    def _set_metadata(self, metadata: dict) -> None:
        """
        @brief: Sets the metadata of the dataset.

        @param metadata: The metadata to set.

        @return: None
        """
        self.idx2word = metadata["idx2word"]
        self.word2idx = metadata["word2idx"]
        self.idx2label = metadata["idx2label"]
        self.label2idx = metadata["label2idx"]
        self.vocabulary = metadata["vocabulary"]
        self.vocabulary_size = metadata["vocabulary_size"]
        self.num_of_labels = metadata["num_of_labels"]

    def _transform_data_to_indices(self) -> None:
        """
        @brief: Transforms the data to indices.

        @return: None
        """
        def transform_X_fn(word: str) -> int:
            return self.word2idx[word] if word in self.vocabulary else self.word2idx[UNKNOWN_TOKEN]

        def transform_y_fn(label: str) -> int:
            return self.label2idx.get(label, -1)

        self.X = np.vectorize(transform_X_fn, otypes=[int])(self.X)
        self.y = np.vectorize(transform_y_fn, otypes=[int])(self.y)
        self.padded_X = np.vectorize(transform_X_fn, otypes=[int])(self.padded_X)
        self.padded_y = np.vectorize(transform_y_fn, otypes=[int])(self.padded_y)


class VocabularyDataset(Dataset):
    """
    @brief: A language tagging dataset that iterates word by word.

    The dataset will be iterated word by word, without word context.
    """

    def __init__(self, vocabulary: set, word2idx: dict) -> None:
        """
        @brief: Initializes the dataset.

        @param vocabulary: The vocabulary to use.
        @param word2idx: The word to index mapping.

        @return: None
        """
        super(VocabularyDataset).__init__()
        self.vocabulary = list(vocabulary)
        self.word2idx = word2idx

    def __len__(self) -> int:
        """
        @brief: Returns the length of the dataset.

        @return: The length of the dataset.
        """
        return len(self.vocabulary)

    def __getitem__(self, index: int) -> torch.Tensor:
        """
        @brief: Returns the word with index in the vocabulary.

        @param index: The index of the word to return.

        @return: The word tensor at the given index.
        """
        word = self.vocabulary[index]
        word_index = self.word2idx[word]
        word_index_tensor = torch.tensor(word_index, dtype=torch.long)

        return word_index_tensor


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
    torch.use_deterministic_algorithms(True)
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


def get_conv2d_layer_shape(layer: torch.nn.Conv2d,
                           in_shape: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
    """
    @brief: Return the shape of torch.nn.Conv2d layer.

    @param layer: The Conv2d layer.
    @param in_shape: The input shape.

    @return: The shape of the Conv2d layer.
    """
    N, C_in, H_in, W_in = in_shape
    C_out = layer.out_channels

    kernel_size = (layer.kernel_size, layer.kernel_size) if isinstance(layer.kernel_size, int) else layer.kernel_size
    dilation = (layer.dilation, layer.dilation) if isinstance(layer.dilation, int) else layer.dilation
    stride = (layer.stride, layer.stride) if isinstance(layer.stride, int) else layer.stride
    groups = layer.groups

    assert layer.in_channels % groups == 0, "C_in must be a multiple of groups"

    if groups == 1:
        assert C_in == groups * \
            layer.in_channels, f"Input channels must be the same: C_weight: {layer.in_channels}, C_input: {C_in}"

    if isinstance(layer.padding, str):
        if layer.padding == "valid":
            H_out = math.floor((H_in + dilation[0] * (kernel_size[0]-1) - 1) / stride[0] + 1)
            W_out = math.floor((W_in + dilation[1] * (kernel_size[1]-1) - 1) / stride[1] + 1)
        elif layer.padding == "same":
            H_out = H_in
            W_out = W_in
    else:
        padding = (layer.padding, layer.padding) if isinstance(layer.padding, int) else layer.padding

        H_out = math.floor((H_in + 2*padding[0] - dilation[0] * (kernel_size[0]-1) - 1) / stride[0] + 1)
        W_out = math.floor((W_in + 2*padding[1] - dilation[1] * (kernel_size[1]-1) - 1) / stride[1] + 1)

    return (N, C_out, H_out, W_out)


def get_maxpool2d_layer_shape(layer: torch.nn.MaxPool2d,
                              in_shape: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
    """
    @brief: Return the shape of torch.nn.MaxPool2d layer.

    @param layer: The MaxPool2d layer.
    @param in_shape: The input shape.

    @return: The shape of the MaxPool2d layer.
    """
    N, C_in, H_in, W_in = in_shape

    kernel_size = (layer.kernel_size, layer.kernel_size) if isinstance(layer.kernel_size, int) else layer.kernel_size
    padding = (layer.padding, layer.padding) if isinstance(layer.padding, int) else layer.padding

    if hasattr(layer, 'dilation'):
        dilation = (layer.dilation, layer.dilation) if isinstance(layer.dilation, int) else layer.dilation
    else:
        dilation = (1, 1)

    stride = (layer.stride, layer.stride) if isinstance(layer.stride, int) else layer.stride

    round_op = round_op = math.ceil if layer.ceil_mode else math.floor

    if isinstance(layer.padding, str):
        if layer.padding == "valid":
            H_out = round_op((H_in + dilation[0] * (kernel_size[0]-1) - 1) / stride[0] + 1)
            W_out = round_op((W_in + dilation[1] * (kernel_size[1]-1) - 1) / stride[1] + 1)
        elif layer.padding == "same":
            H_out = H_in
            W_out = W_in
    else:
        padding = (layer.padding, layer.padding) if isinstance(layer.padding, int) else layer.padding

        H_out = round_op((H_in + 2*padding[0] - dilation[0] * (kernel_size[0]-1) - 1) / stride[0] + 1)
        W_out = round_op((W_in + 2*padding[1] - dilation[1] * (kernel_size[1]-1) - 1) / stride[1] + 1)

    return (N, C_in, H_out, W_out)


def load_train_dataset(device: str, tag_type: str, data_directory: str,
                       batch_size: int = 1, num_workers: int = 4,
                       shuffle: bool = True, use_pretrained_embedding: bool = False,
                       vocabulary_file: str = "", embedding_file: str = "",
                       subword_embedding: str = "", window_size: int = 5) -> Tuple[DataLoader, DataLoader]:
    """
    @brief: Loads the training dataset.

    @param device: The device to load the dataset to.
    @param tag_type: The type of the tagging task.
    @param data_directory: The directory of the data, containing train/dev datasets.
    @param batch_size: The batch size of the dataset.
    @param num_workers: The number of workers to use for loading the dataset.
    @param shuffle: Whether to shuffle the train dataset.
    @param use_pretrained_embedding: Whether to use pretrained embedding.
    @param vocabulary_file: The file containing the vocabulary for pretrained embedding.
    @param embedding_file: The file containing the pretrained embedding.
    @param subword_embedding: Type of sub-word embedding to use, cbow/cnn.
    @param window_size: The window size for the context.

    @return: The train and test data loaders.
    """
    print("Loading train dataset...")

    separator = NER_SEPERATOR if tag_type == NER_TAGGING else POS_SEPERATOR
    if use_pretrained_embedding:
        pretrained_vocabulary, pretrained_embedding = load_pretrained_embedding(vocabulary_file, embedding_file)
    else:
        pretrained_vocabulary, pretrained_embedding = None, None

    train_dataset = LanguageTaggingDataset(device=device, root_dir=data_directory, filename="train", separator=separator,
                                           window_size=window_size, use_pretrained_embedding=use_pretrained_embedding,
                                           pretrained_vocabulary=pretrained_vocabulary,
                                           pretrained_embedding=pretrained_embedding,
                                           subword_embedding=subword_embedding)
    dev_dataset = LanguageTaggingDataset(device=device, root_dir=data_directory, filename="dev", separator=separator,
                                         window_size=window_size,
                                         use_pretrained_embedding=use_pretrained_embedding,
                                         subword_embedding=subword_embedding,
                                         metadata=train_dataset.get_metadata())
    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle,
                                   num_workers=num_workers, prefetch_factor=num_workers,
                                   persistent_workers=True, worker_init_fn=seed_worker, generator=GLOBAL_RAND_GEN)
    dev_data_loader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False,
                                 num_workers=num_workers, prefetch_factor=num_workers,
                                 persistent_workers=True, worker_init_fn=seed_worker, generator=GLOBAL_RAND_GEN)

    return train_data_loader, dev_data_loader


def load_test_dataset(device: str, tag_type: str, data_directory: str, batch_size: int = 1,
                      num_workers: int = 4, metadata: dict = None,
                      use_pretrained_embedding: bool = False,
                      subword_embedding: str = "", window_size: int = 5) -> DataLoader:
    """
    @brief: Loads the test dataset.

    @param device: The device to load the dataset to.
    @param tag_type: The type of the tagging task.
    @param data_directory: The directory of the data, containing test dataset.
    @param batch_size: The batch size of the dataset.
    @param num_workers: The number of workers to load the dataset.
    @param metadata: The metadata of the dataset.
    @param use_pretrained_embedding: Whether to use CBOW the pretrained embedding.
    @param subword_embedding: Type of sub-word embedding to use, cbow/cnn.
    @param window_size: The window size for the context.

    @return: The test data loader.
    """
    print("Loading test dataset...")
    separator = NER_SEPERATOR if tag_type == NER_TAGGING else POS_SEPERATOR
    test_dataset = LanguageTaggingDataset(device=device, root_dir=data_directory, filename="test",
                                          window_size=window_size, separator=separator, metadata=metadata,
                                          use_pretrained_embedding=use_pretrained_embedding,
                                          subword_embedding=subword_embedding)
    test_data_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                                  num_workers=num_workers, prefetch_factor=num_workers,
                                  persistent_workers=True, worker_init_fn=seed_worker, generator=GLOBAL_RAND_GEN)

    return test_data_loader


def load_pretrained_embedding(vocabulary_file: str, embedding_file: str) -> Tuple[np.ndarray, torch.tensor]:
    """
    @brief: Loads the pretrained embedding.

    @param vocabulary_file: The file containing the vocabulary.
    @param embedding_file: The file containing the embedding.

    @return: The vocabulary and embedding.
    """
    print("Loading pretrained embedding...")

    vocabulary = pd.read_table(vocabulary_file, dtype=str, skip_blank_lines=False, header=None,
                               keep_default_na=False, na_filter=True, quoting=3, names=["word"])
    vocabulary.word = vocabulary.word.apply(lambda word: word.lower())
    vocabulary = vocabulary.word.values
    embedding = torch.tensor(np.loadtxt(embedding_file))

    return vocabulary, embedding


def get_vocabulary_data_loader(dataset: dict, batch_size: int = 1,
                               num_workers: int = 4, word_frequency: int = 10) -> DataLoader:
    """
    @brief: Returns the vocabulary dataloader.

    @param dataset: The train dataset.
    @param batch_size: The batch size of the dataset.
    @param num_workers: The number of workers to load the dataset.
    @param word_frequency: The minimum frequency of the words to be included in the vocabulary.

    @return: The vocabulary data loader.
    """
    dataset = VocabularyDataset(vocabulary=dataset.get_vocabulary(frequency=word_frequency), word2idx=dataset.word2idx)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, prefetch_factor=num_workers,
                            persistent_workers=True, worker_init_fn=seed_worker, generator=GLOBAL_RAND_GEN)

    return dataloader


def get_top_k(word: str, vocabulary: np.ndarray, embedding: torch.tensor, k: int) -> Tuple[list, torch.tensor]:
    """
    @brief: Gets the top k words of the given word.

    Uses the cosine similarity to find the top k words of the given word.

    @param word: The word to get the top k words of.
    @param vocabulary: The vocabulary.
    @param embedding: The embedding vectors.
    @param k: The number of top words to get.

    @return: The top k words and their scores.
    """
    if word not in vocabulary:
        return [], []

    top_k_words = []
    word_index = np.where(vocabulary == word)[0]
    word_embedding = embedding[word_index]
    cosine_similarities = torch.cosine_similarity(word_embedding, embedding, dim=1).sort(descending=True)
    top_k_words = vocabulary[cosine_similarities.indices[1:k + 1]]
    top_k_scores = torch.round(cosine_similarities.values[1:k + 1], decimals=2)

    return top_k_words, top_k_scores


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


def train(dataloader: DataLoader, model: torch.nn.Module, loss_fn,
          optimizer: torch.optim, gradient_clip: int = 0,
          device: str = "cpu", log: bool = False) -> Tuple[float, float]:
    """
    @brief: Trains the model.

    @param dataloader: The dataloader to use.
    @param model: The model to train.
    @param loss_fn: The loss function to use.
    @param optimizer: The optimizer to use.
    @param gradient_clip: The gradient clip value.
    @param device: The device to use.
    @param log: Whether to log the training progress.

    @return: Average train loss and accuracy tuple.
    """
    size = len(dataloader.dataset)
    total_loss, correct = 0, 0

    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        # Forward pass:
        y_prob = model(X)

        # Compute prediction error:
        y_pred = y_prob.argmax(dim=1)
        loss = loss_fn(y_prob, y.long()).type(torch.float64)
        total_loss += loss.item()
        correct += (y_pred == y).type(torch.long).sum().item()

        # Backward pass:
        optimizer.zero_grad()
        loss.backward()
        if gradient_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip)
        optimizer.step()

        # Log progress:
        if log is True and batch % 100 == 0:
            current_loss, current_batch = loss.item(), batch * len(X)
            print(f"Train Loss: {current_loss:>7f}  [{current_batch:>5d}/{size:>5d}]")

    num_of_batches = len(dataloader)
    avg_loss = total_loss / num_of_batches
    avg_accuracy = (correct / size) * 100

    return avg_loss, avg_accuracy


def test(dataloader: DataLoader, model: torch.nn.Module, device: str, loss_fn,
         filter_index: int = None) -> Tuple[float, float]:
    """
    @brief: Tests the model.

    @param dataloader: The dataloader to use.
    @param model: The model to test.
    @param device: The device to use.
    @param loss_fn: The loss function to use.
    @param filter_index: Index of a label not to consider in accuracy calculation.

    @return: Average test loss and accuracy tuple.
    """
    size = len(dataloader.dataset)
    model.eval()
    total_loss, correct = 0, 0
    correct_to_filter = 0

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            # Forward pass:
            y_prob = model(X)
            y_pred = y_prob.argmax(dim=1)

            # Compute prediction error:
            loss = loss_fn(y_prob, y.long()).type(torch.float64).item()
            total_loss += loss
            correct += (y_pred == y).type(torch.long).sum().item()

            # Log progress:
            if filter_index is not None:
                correct_to_filter += ((y == filter_index) & (y_pred == filter_index)).type(torch.long).sum().item()

    num_of_batches = len(dataloader)
    avg_loss = total_loss / num_of_batches
    avg_accuracy = ((correct - correct_to_filter) / (size - correct_to_filter)) * 100

    return avg_loss, avg_accuracy


def fit(train_data_loader:  DataLoader, dev_data_loader: DataLoader, device: str, model: torch.nn.Module,
        loss_fn, optimizer: torch.optim, gradient_clip: int = 0, scheduler: torch.optim.lr_scheduler = None,
        epochs: int = 10, log_train: bool = False, log_error: bool = True,
        filter_index: int = None) -> Tuple[torch.nn.Module, Tuple[list, list, list, list]]:
    """
    @brief: Fits the model.

    @param train_data_loader: The train dataloader to use.
    @param dev_data_loader: The dev dataloader to use.
    @param device: The device to use.
    @param model: The model to fit.
    @param loss_fn: The loss function to use.
    @param optimizer: The optimizer to use.
    @param gradient_clip: The gradient clip value.
    @param scheduler: The scheduler of learning rate adjustments to use.
    @param epochs: The number of epochs to fit.
    @param log_train: Whether to log the training progress.
    @param log_error: Whether to log the error of train/dev progress.
    @param filter_index: Index of a label not to consider in accuracy calculation for dev evaluation.

    @return: The trained model and the train/dev loss/accuracy tuple.
    """
    print("Training...")

    train_losses, train_accuracies = [], []
    dev_losses, dev_accuracies = [], []

    for epoch in range(epochs):
        train_loss, train_accuracy = train(train_data_loader, model, loss_fn,
                                           optimizer, gradient_clip, device, log_train)
        dev_loss, dev_accuracy = test(dev_data_loader, model, device, loss_fn, filter_index)

        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)
        dev_losses.append(dev_loss)
        dev_accuracies.append(dev_accuracy)

        log_message = f"Epoch {epoch + 1}/{epochs}"
        if log_error is True:
            train_error_log = f": Train => Accuracy: {(train_accuracy):>0.4f} [%] ; Loss: {train_loss:>6f}"
            dev_error_log = f" | Dev => Accuracy: {(dev_accuracy):>0.4f} [%] ; Loss: {dev_loss:>6f}"
            log_message += train_error_log + dev_error_log
        print(log_message)

        if scheduler is not None:
            scheduler.step()

    print("Done training!")
    return model, (train_losses, train_accuracies, dev_losses, dev_accuracies)


def evaluate(test_data_loader: DataLoader, device: str,
             model: torch.nn.Module, filter_index: int, predict_path: str = "") -> None:
    """
    @brief: Evaluates the model.

    This method is used to evaluate the model on the test/dev set,
    and optionally to save the predictions to a file.

    @param test_data_loader: The test dataloader to use.
    @param device: The device to use.
    @param model: The model to evaluate.
    @param filter_index: Index of a label not to consider in accuracy calculation.
    @param predict_path: The path to the predictions file.

    @return: None.
    """
    print("Evaluating...")

    def write_single_prediction(prediction_index: int) -> None:
        nonlocal location_in_X
        with open(predict_path, "a") as file:
            prediction_label = test_data_loader.dataset.idx2label[prediction_index]
            log = f"{prediction_label}"

            padded_word_index = test_data_loader.dataset.original2padded[location_in_X]
            padded_next_word_index = test_data_loader.dataset.padded_X[padded_word_index + 1]
            if padded_next_word_index == test_data_loader.dataset.padding_token_index:
                log += "\n"

            file.write(f"{log}\n")
            location_in_X += 1

        return None

    if predict_path != "":
        open(predict_path, "w").close()

    write_batch_predictions = np.vectorize(write_single_prediction, otypes=[None])
    model.eval()
    correct, correct_to_filter = 0, 0
    location_in_X = 0

    with torch.no_grad():
        for X, y in test_data_loader:
            X, y = X.to(device), y.to(device)

            # Predict:
            y_pred = model.predict(X)

            # Compute prediction error:
            correct += (y_pred == y).type(torch.long).sum().item()
            if filter_index is not None:
                correct_to_filter += ((y == filter_index) & (y_pred == filter_index)).type(torch.long).sum().item()

            # Write predictions:
            if predict_path != "":
                write_batch_predictions(y_pred.cpu().numpy())

    size = len(test_data_loader.dataset)
    avg_accuracy = (correct / size) * 100
    filtered_avg_accuracy = ((correct - correct_to_filter) / (size - correct_to_filter)) * 100

    print("Done evaluating!")
    return avg_accuracy, filtered_avg_accuracy


def visualize_cnn_filters(dataset: Dataset, model: torch.nn.Module,
                          word_frequency: int = 1, plot_path: str = "") -> None:
    """
    @brief: Shows the filters explanation.

    The method is supported when sklearn is installed.
    It will save the filters to a plot files.

    @param dataset: The train dataset.
    @param model: The model to use.
    @param word_frequency: The number of words to show.
    @param plot_path: The path to the plot file.

    @return: None.
    """
    if TSNE is None:
        print("Please install sklearn to visualize the filters.")
        return None

    print("Visualizing filters, please wait, this might take a while...")

    # Get the filters
    convolved_filters = []
    vocabulary_data_loader = get_vocabulary_data_loader(
        dataset=dataset, batch_size=model.batch_size * 5, word_frequency=word_frequency)
    for batch in vocabulary_data_loader:
        convolved_batch_words = model.get_convolved_chars(batch)
        convolved_filters.append(convolved_batch_words)

    convolved_filters = torch.cat(convolved_filters, dim=0)
    convolved_filters = convolved_filters.permute(1, 0, 2).detach().cpu().numpy()

    # Reduce filters dimensionality:
    dim_2_filters = []
    for filter_idx in range(convolved_filters.shape[0]):
        dim_2_filter = TSNE(n_components=2, learning_rate='auto',
                            init='random').fit_transform(convolved_filters[filter_idx])
        dim_2_filters.append(dim_2_filter)

    # Save filters plots:
    plt.ioff()
    words = [word for word in vocabulary_data_loader.dataset.vocabulary]
    for filter_idx, _filter in enumerate(dim_2_filters):
        points_2d = [(word[0], word[1]) for word in _filter]
        fig, ax = plt.subplots()
        fig.set_size_inches(80, 80)
        ax.scatter(*zip(*points_2d))
        for index, word in enumerate(words):
            ax.annotate(word, points_2d[index])

        figure_name = f"{plot_path}/{model.tag_type}_filter_{filter_idx+1}"
        fig.savefig(f"{figure_name}.png")
        plt.close("all")


def save_plot_statistics(tag_type: str, statistics: Tuple[list, list, list, list], plot_path: str = ".") -> None:
    """
    @brief: Saves fit statistics plots to script directory.

    @param tag_type: The type of the tag.
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
        plt.title(f"{tag_type.upper()} Training and Dev {stat}")
        plt.savefig(f"{plot_path}/{tag_type}_train_dev_{stat.lower()}.png")
        plt.close()

    plot_helper("Loss", train_losses, dev_losses)
    plot_helper("Accuracy", train_accuracies, dev_accuracies)


def parser_cli(top_k_parser: bool = False) -> argparse.Namespace:
    """
    @brief: Parses the command line arguments.

    @param top_k_parser: Whether to parse the top-k arguments.

    @return: The parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(description="Language tagging")

    parser.add_argument("--fit", action="store_true", default=True,
                        help="Fit the model.")
    parser.add_argument("--debug", action="store_true", default=False,
                        help="Run in debug mode, set parameters in tagger module.")
    parser.add_argument("--dataset-path", type=str, default=os.path.join("..", "data", "ner"),
                        help="The path to the folder with datasets named train/dev.")
    parser.add_argument("--tag-type", type=str, default=NER_TAGGING, choices=[NER_TAGGING, POS_TAGGING],
                        help="The language tagging task.")
    parser.add_argument("--plot-path", type=str, default="",
                        help="Path to save plots of fitting statistics and CNN filters visualization.")
    parser.add_argument("--evaluate", action="store_true", default=False,
                        help="Evaluate the model.")
    parser.add_argument("--test-path", type=str, default="",
                        help="The path to the folder with dataset named test.")
    parser.add_argument("--predict-path", type=str, default="",
                        help="The path to save the predictions.")
    parser.add_argument("--epochs", type=int, default=300,
                        help="The number of epochs to fit.")
    parser.add_argument("--hidden-dim", type=int, default=128,
                        help="The hidden layer dimension of the model.")
    parser.add_argument("--batch-size", type=int, default=128,
                        help="The batch size to use.")
    parser.add_argument("--log-train", action="store_true", default=False,
                        help="Print logs of the training progress.")
    parser.add_argument("--log-error", action="store_true", default=True,
                        help="Print logs of the train/dev progress error.")
    parser.add_argument("--lr", type=float, default=0.003,
                        help="The learning rate to use.")
    parser.add_argument("--dropout", type=float, default=0,
                        help="The dropout to use.")
    parser.add_argument("--weight-decay", type=float, default=0,
                        help="The weight decay to use.")
    parser.add_argument("--sched-step", type=int, default=10,
                        help="The step of the learning rate scheduler to use.")
    parser.add_argument("--sched-gamma", type=float, default=1,
                        help="The gamma of the learning rate scheduler to use.")
    parser.add_argument("--subword-embedding", type=str, default="",
                        choices=[CBOW_SUBWORD_EMBEDDING, CNN_SUBWORD_EMBEDDING],
                        help="Type of sub-word embedding to use.")
    parser.add_argument("--char-embedding-dim", type=int, default=15,
                        help="The character embedding dimension to use, used with CNN sub-word embedding.")
    parser.add_argument("--cnn-char-num-filters", type=int, default=15,
                        help="The number of filters to use for CNN sub-word embedding.")
    parser.add_argument("--cnn-char-kernel-size", type=int, default=3,
                        help="The kernel height to use for CNN sub-word embedding, width is --char-embedding-dim.")
    parser.add_argument("--cnn-char-stride", type=int, default=1,
                        help="The stride height size to use for CNN sub-word embedding.")
    parser.add_argument("--window-size", type=int, default=5,
                        help="The window size to use for the context.")
    parser.add_argument("--visualize-cnn-filters", action="store_true", default=False,
                        help="Visualize CNN filters explanation,"
                             " note this option takes a while to complete.")

    if top_k_parser is True:
        parser = argparse.ArgumentParser(description="Top K")
        parser.add_argument("--debug", action="store_true", default=False,
                            help="Run in debug mode, set parameters in tagger module.")
        parser.add_argument("--top-k", type=int, default=None,
                            help="Get top K similarities.")
        parser.add_argument("--top-k-input", nargs="+", type=int, default=["dog", "england", "john", "explode", "office"],
                            help="Get top K similarities for specific input.")

    parser.add_argument("--vocabulary-file",
                        type=str, default="",
                        help="The path to the vocabulary file.")
    parser.add_argument("--embedding-file",
                        type=str, default="",
                        help="The path to the embedding vectors file.")

    return parser.parse_args()
