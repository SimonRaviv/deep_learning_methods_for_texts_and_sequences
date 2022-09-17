"""
Library module for implementation of paper:
    "A Decomposable Attention Model for Natural Language Inference", 1606.01933v2.

Author: Simon Raviv.
"""
import datetime
import os
import random
import numpy as np
import torch
import h5py
import logging
import operator
import matplotlib.pyplot as plt
import json

from collections import defaultdict, Counter
from typing import List, Tuple


#################################################################################################################
#                                                Global Variables                                               #
#################################################################################################################


SEED = 2022
GLOBAL_RAND_GEN = torch.Generator()
OOV_NUM_TOKENS = 100
BLANK_TOKEN = "<BLANK>"
UNKNOWN_TOKEN = "<UNKNOWN>"
START_TOKEN = "<START>"
END_TOKEN = "</START>"
OOV_TOKEN_FORMAT = "<OOV{index}>"
GLOVE_DIMENSION = 300
DATASETS = ["train", "dev", "test"]
DICTIONARY_SEPARATOR = " "
SNLI_LABELS = set(["neutral", "entailment", "contradiction"])
MAX_SENTENCE_LENGTH = 100
LOGGER_NAME = "datten"
LOGGER = logging.getLogger(LOGGER_NAME)
LOGGER_SEPARATOR_LINE = "-" * 140

TIME_FORMAT = "%H:%M:%S"

#################################################################################################################
#                                               General Utilities                                               #
#################################################################################################################


def get_time_now(time_format: str = TIME_FORMAT) -> str:
    """
    @brief: Returns the current time in the given format.

    @return: The current time.
    """
    return datetime.datetime.strptime(datetime.datetime.now().strftime(time_format), time_format)


def get_duration_in_seconds(start_timestamp: datetime.datetime, end_timestamp: datetime.datetime) -> str:
    """
    @brief: Get duration in seconds between two timestamps.

    @param start_timestamp: Start timestamp.
    @param end_timestamp: End timestamp.

    @return: Duration between two timestamps.
    """
    test_duration = end_timestamp - start_timestamp
    test_duration = datetime.timedelta() - test_duration if test_duration.days < 0 else test_duration
    time_duration_str = f"{test_duration.total_seconds():.2f}"

    return time_duration_str


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


def initialize_logger(log_path: str, console: bool) -> None:
    """
    @brief: Initializes the logger.

    @param log_path: The path to the log file.
    @param console: Whether to log to console.

    @return: None.
    """
    formatter = logging.Formatter(fmt='%(asctime)s %(levelname)-4s %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    if console is True:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        LOGGER.addHandler(console_handler)

    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    LOGGER.addHandler(file_handler)
    LOGGER.setLevel(logging.INFO)
    LOGGER.info("Logger initialized")


def save_model(model_path: str, model: torch.nn.Module, fit_statistics: dict) -> None:
    """
    @brief: Saves model to file.

    @param model_path: The path to the file to save the model.
    @param model: The model to save.
    @param fit_statistics: The fit statistics to save.

    @return: None.
    """
    LOGGER.info("Saving model")

    torch.save({
        "model": model.state_dict(),
        "model_non_trainable_parameters": model.get_non_trainable_parameters_state_dict(),
        "fit_statistics": fit_statistics},
        model_path)

    LOGGER.info("Done saving model")


def load_model(device: str, model_path: str, pretrained_embeddings: torch.Tensor,
               model_class: type) -> Tuple[torch.nn.Module, dict]:
    """
    @brief: Loads model from file.

    @param device: The device to load the model on.
    @param model_path: The path to the file to load the model from.
    @param pretrained_embeddings: The pretrained embeddings to load.
    @param model_class: The class of the model to load.

    @return: The loaded model and the dataset metadata.
    """
    LOGGER.info("Loading model")

    checkpoint = torch.load(model_path, map_location=torch.device(device))
    checkpoint.update({"pretrained_embeddings": pretrained_embeddings})
    model = model_class(device=device, **checkpoint["model_non_trainable_parameters"])
    model.load_state_dict(checkpoint["model"])
    fit_statistics = checkpoint["fit_statistics"]

    LOGGER.info("Done loading model")
    return model, fit_statistics


def get_device() -> str:
    """
    @brief: Returns the device to use.

    @return: The device to use.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    LOGGER.info(f"Using {device.upper()} device")

    return device

#################################################################################################################
#                                                 Data Handling                                                 #
#################################################################################################################


class SNLIDataset(torch.utils.data.Dataset):
    """
    @brief: Stanford Natural Language Inference dataset.
    """

    def __init__(self, device: str, filename: str, max_sentence_length: int = MAX_SENTENCE_LENGTH) -> None:
        """
        @brief: Initialize the dataset.

        @param device: The device to use.
        @param filename: The path to the file to load the HDF5 dataset from.
        @param max_sentence_length: The maximum sentence length to use.

        @return: None.
        """
        super(SNLIDataset, self).__init__()
        self.device = device
        self.filename = filename
        self.max_sentence_length = max_sentence_length
        with h5py.File(self.filename, 'r') as dataset:
            # Reduce by 1 to start from 0:
            self.source = torch.from_numpy(np.array(dataset['source'])).to(device) - 1
            self.target = torch.from_numpy(np.array(dataset['target'])).to(device) - 1
            self.label = torch.from_numpy(np.array(dataset['label'])).to(device) - 1
            self.label_size = torch.from_numpy(np.array(dataset['label_size'])).to(device)
            self.label2idx = json.loads(dataset['label_indexer'][...].tolist().decode())
            self.source_length = torch.from_numpy(np.array(dataset['source_length'])).to(device)
            self.target_length = torch.from_numpy(np.array(dataset['target_length'])).to(device)
            self.batch_index = torch.from_numpy(np.array(dataset['batch_index'])).to(device) - 1
            self.batch_lengths = torch.from_numpy(np.array(dataset['batch_length'])).to(device)
        self.num_of_batches = self.batch_lengths.size(0)
        self.label2idx = {label: index - 1 for label, index in self.label2idx.items()}
        self.idx2label = {index: label for label, index in self.label2idx.items()}
        self.total_per_label = Counter(self.label.tolist())
        self.batches = []
        self.size = 0
        self._initialize_batches()

    def _initialize_batches(self) -> None:
        """
        @brief: Initialize the batches.

        @return: None.
        """
        self.batches = []
        self.size = 0
        for i in range(self.num_of_batches):
            if self.source_length[i] <= self.max_sentence_length and self.target_length[i] <= self.max_sentence_length:
                batch = (
                    self.source[self.batch_index[i]: self.batch_index[i] +
                                self.batch_lengths[i]][:, :self.source_length[i]],
                    self.target[self.batch_index[i]: self.batch_index[i] +
                                self.batch_lengths[i]][:, :self.target_length[i]],
                    self.label[self.batch_index[i]: self.batch_index[i] + self.batch_lengths[i]],
                )
                self.batches.append(batch)
                self.size += self.batch_lengths[i]

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        return self.batches[index]

    def shuffle(self) -> None:
        """
        @brief: Shuffles the dataset.

        @return: None.
        """
        permutation = torch.randperm(self.num_of_batches)
        self.batch_index = self.batch_index[permutation]
        self.batch_lengths = self.batch_lengths[permutation]
        self.batches = operator.itemgetter(*permutation)(self.batches)


class DataIndexer:
    """
    @brief: Indexes the data as a dictionary for serialization.
    """

    def __init__(self, symbols: List[str]) -> None:
        """
        @brief: Initialize the indexer.

        @param symbols: Predefined symbols.

        @return: None.
        """
        super(DataIndexer, self).__init__()
        self.vocabulary = defaultdict(int)
        self.filtered_vocabulary = defaultdict(int)
        self.padding_token, self.unknown_token, self.bos_token, self.eos_token = symbols
        self.dictionary = {self.padding_token: 1, self.unknown_token: 2, self.bos_token: 3, self.eos_token: 4}

    def __len__(self):
        return len(self.dictionary)

    def _word_to_index(self, word: str) -> int:
        """
        @brief: Converts a word to an index.

        @param word: The word to convert.

        @return: The index of the word.
        """
        oov = OOV_TOKEN_FORMAT.format(index=np.random.randint(1, OOV_NUM_TOKENS))
        return self.dictionary[word] if word in self.dictionary else self.dictionary[oov]

    def _get_padded_sequence(self, sentence: List[str], length: int, symbol: str, pad_back=True):
        """
        @brief: Returns padded sentence to a given length.

        @param sentence: The sentence to pad.
        @param length: The length to pad to.
        @param symbol: The symbol to pad with.
        @param pad_back: Whether to pad the sentence back or front.

        @return: The padded sentence.
        """
        if len(sentence) >= length:
            return sentence[:length]

        if pad_back is True:
            return sentence + [symbol] * (length - len(sentence))
        else:
            return [symbol] * (length - len(sentence)) + sentence

    def get_padded_sequence_ndarray(self, sequence: List[str], length: int) -> Tuple[np.ndarray, int]:
        """
        @brief: Converts a sequence to indices.

        @param sequence: The sequence to convert.
        @param length: The length to pad to.

        @return: Tuple of the indices and the not padded length of the sequence.
        """
        padded_sequence = self._get_padded_sequence(sequence, length, self.padding_token)
        padded_sequence_array = np.array([self._word_to_index(word) for word in padded_sequence], dtype=int)
        padding_token_index = self.dictionary[self.padding_token]
        not_padded_length = (padded_sequence_array != padding_token_index).sum()
        return padded_sequence_array, not_padded_length

    def get_sentence(self, sentence: str) -> List[str]:
        """
        @brief: Returns sentence list without special symbols.

        @param sentence: The input sentence.

        @return: List of sentence words.
        """
        return sentence.strip().replace(self.padding_token, "").replace(
            self.bos_token, "").replace(self.eos_token, "").strip().split()

    def filter_vocabulary(self, k: int, filter_by_k: bool = False) -> None:
        """
        @brief: Filters the vocabulary by keeping the k most frequent words.

        @param k: The number of words to keep.
        @param filter_by_k: Whether to filter by k or by the number of words in the vocabulary.

        @return: None.
        """

        vocabulary_list = [(word, frequency) for word, frequency in self.vocabulary.items()]

        if filter_by_k is True:
            self.filtered_vocabulary = {word: frequency for word, frequency in vocabulary_list if frequency > k}
        else:
            vocabulary_list.sort(key=lambda word_frequency: word_frequency[1], reverse=True)
            k = min(k, len(vocabulary_list))
            self.filtered_vocabulary = {word: frequency for word, frequency in vocabulary_list[:k]}

        for word in self.filtered_vocabulary:
            if word not in self.dictionary:
                self.dictionary[word] = len(self.dictionary) + 1


class Word2Vector:
    """
    @brief: Loads the word2vec embeddings.
    """

    def __init__(self, word_vectors_path: str) -> None:
        """
        @brief: Initialize the word2vec embeddings.

        @param word_vectors_path: The path to the word2vec embeddings.

        @return: None.
        """
        super(Word2Vector, self).__init__()
        self.word_vectors_path = word_vectors_path
        with h5py.File(word_vectors_path, 'r') as f:
            self.word_vectors = torch.from_numpy(np.array(f['word_embedding_vectors'])).float()

    def get_word_vectors(self) -> torch.Tensor:
        """
        @brief: Returns the word2vec embeddings.

        @return: The word2vec embeddings.
        """
        return self.word_vectors


def update_vocabulary(dataset: dict, sequence_length: int,
                      word_indexer: DataIndexer, label_indexer: DataIndexer, glove_vocabulary: set) -> int:
    """
    @brief: Update the vocabulary for data indexers based on the dataset.

    @param dataset: The dataset.
    @param sequence_length: The maximum length of the sequence.
    @param word_indexer: The word indexer.
    @param label_indexer: The label indexer.
    @param glove_vocabulary: The glove vocabulary.

    @return: The number of words in the vocabulary.
    """
    sentences_counter = 0
    for source, target, label in zip(dataset["source"], dataset["target"], dataset["label"]):
        source = word_indexer.get_sentence(source)
        target = word_indexer.get_sentence(target)
        label = label_indexer.get_sentence(label)

        if len(target) > sequence_length or len(source) > sequence_length or len(target) < 1 or len(source) < 1:
            continue
        sentences_counter += 1

        for word in source:
            if word in glove_vocabulary:
                word_indexer.vocabulary[word] += 1

        for word in target:
            if word in glove_vocabulary:
                word_indexer.vocabulary[word] += 1

        for word in label:
            label_indexer.vocabulary[word] += 1

    return sentences_counter


def write_hdf5_dataset_file(set_type: str, dataset: dict, sequence_length: int, num_sentences: int,
                            max_sequence_length: int, shuffle: bool, batch_size: int, output_path: str,
                            word_indexer: DataIndexer, label_indexer: DataIndexer,
                            statistics: dict) -> None:
    """
    @brief: Writes the dataset to an HDF5 file.

    @param set_type: The type of the dataset.
    @param dataset: The dataset.
    @param sequence_length: The maximum length of the sequence.
    @param num_sentences: The number of sentences in the dataset.
    @param max_sequence_length: The maximum length of the sequence.
    @param shuffle: Whether to shuffle the dataset.
    @param batch_size: The batch size.
    @param output_path: The output path.
    @param word_indexer: The word indexer.
    @param label_indexer: The label indexer.
    @param statistics: Data processing statistics.

    @return: None.
    """
    new_sequence_length = sequence_length + 1  # Add 1 for BOS token
    targets = np.zeros((num_sentences, new_sequence_length), dtype=int)
    sources = np.zeros((num_sentences, new_sequence_length), dtype=int)
    labels = np.zeros((num_sentences,), dtype=int)
    sources_lengths = np.zeros((num_sentences,), dtype=int)
    targets_lengths = np.zeros((num_sentences,), dtype=int)
    both_lengths = np.zeros(num_sentences, dtype={'names': ['x', 'y'], 'formats': ['i4', 'i4']})
    filtered_sentences = sentence_id = 0

    # Create padded dataset:
    for source, target, label in zip(dataset["source"], dataset["target"], dataset["label"]):
        source = [word_indexer.bos_token] + word_indexer.get_sentence(source)
        target = [word_indexer.bos_token] + word_indexer.get_sentence(target)
        label = label_indexer.get_sentence(label)

        max_sequence_length = max(len(target), len(source), max_sequence_length)
        if len(target) > new_sequence_length or len(source) > new_sequence_length or len(target) < 2 or len(source) < 2:
            filtered_sentences += 1
            continue

        target_length_tuple = word_indexer.get_padded_sequence_ndarray(target, new_sequence_length)
        source_length_tuple = word_indexer.get_padded_sequence_ndarray(source, new_sequence_length)

        targets[sentence_id], targets_lengths[sentence_id] = target_length_tuple
        sources[sentence_id], sources_lengths[sentence_id] = source_length_tuple
        labels[sentence_id] = label_indexer.dictionary[label[0]]
        both_lengths[sentence_id] = (sources_lengths[sentence_id], targets_lengths[sentence_id])

        sentence_id += 1

    # Shuffle the dataset:
    if shuffle is True:
        random_permutation = np.random.permutation(sentence_id)
        targets = targets[random_permutation]
        sources = sources[random_permutation]
        sources_lengths = sources_lengths[random_permutation]
        targets_lengths = targets_lengths[random_permutation]
        labels = labels[random_permutation]
        both_lengths = both_lengths[random_permutation]

    # Order the data by lengths:
    targets_lengths = sources_lengths[:sentence_id]
    sources_lengths = sources_lengths[:sentence_id]
    both_lengths = both_lengths[:sentence_id]
    both_sorted_lengths = np.argsort(both_lengths, order=('x', 'y'))
    sources = sources[both_sorted_lengths]
    targets = targets[both_sorted_lengths]
    labels = labels[both_sorted_lengths]
    target_length = targets_lengths[both_sorted_lengths]
    source_length = sources_lengths[both_sorted_lengths]

    current_source_length = 0
    current_target_length = 0
    lengths_location = []

    for j, i in enumerate(both_sorted_lengths):
        if sources_lengths[i] > current_source_length or targets_lengths[i] > current_target_length:
            current_source_length = sources_lengths[i]
            current_target_length = targets_lengths[i]
            lengths_location.append(j + 1)
    lengths_location.append(len(sources))

    # Build batches:
    current_index = 1
    batch_index = [1]
    batch_length = []
    target_length_new = []
    source_length_new = []
    for i in range(len(lengths_location) - 1):
        while current_index < lengths_location[i + 1]:
            current_index = min(current_index + batch_size, lengths_location[i+1])
            batch_index.append(current_index)

    for i in range(len(batch_index) - 1):
        batch_length.append(batch_index[i + 1] - batch_index[i])
        source_length_new.append(source_length[batch_index[i] - 1])
        target_length_new.append(target_length[batch_index[i] - 1])

    # Write the dataset to the HDF5 file:
    with h5py.File(output_path, "w") as file:
        file["source"] = sources
        file["target"] = targets
        file["target_length"] = np.array(target_length_new, dtype=int)
        file["source_length"] = np.array(source_length_new, dtype=int)
        file["label"] = np.array(labels, dtype=int)
        file["label_size"] = np.array([len(np.unique(np.array(labels, dtype=int)))])
        file["label_indexer"] = json.dumps(label_indexer.dictionary)
        file["batch_length"] = np.array(batch_length, dtype=int)
        file["batch_index"] = np.array(batch_index[:-1], dtype=int)
        file["source_size"] = np.array([len(word_indexer)])
        file["target_size"] = np.array([len(word_indexer)])

    # Save statistics:
    statistics["filtered_sentences"][set_type] = filtered_sentences

    return max_sequence_length


def load_glove_vocabulary(glove_path: str) -> set:
    """
    @brief: Loads the glove vocabulary.

    @param glove_path: Path to the glove vectors.

    @return: A set of the words in the glove vectors.
    """
    with open(glove_path, 'r', encoding='utf-8') as file:
        glove_vocabulary = set([line.split()[0] for line in file.readlines()])
    return glove_vocabulary


def load_glove_vectors(glove_path: str, vocabulary: dict) -> dict:
    """
    @brief: Loads glove vectors for words in vocabulary.

    @param glove_path: Path to glove vectors.
    @param vocabulary: Dictionary of words to load glove vectors for.

    @return: Dictionary of words to glove vectors.
    """
    word_vectors = {}
    with open(glove_path, 'r', encoding='utf-8') as glove_file:
        for line in glove_file:
            line_tuple = line.split(' ')
            word = line_tuple[0]
            vector = line_tuple[1:]
            np_vector = np.asarray(vector, dtype='float32')
            if word in vocabulary:
                word_vectors[word] = np_vector

    return word_vectors


def load_snli_dataset(snli_path: str) -> dict:
    """
    @brief: Loads the SNLI dataset.

    This method loads SNLI dataset into a dictionary.
    The dictionary contains the following keys:
        - 'train': A list of tuples lists (premise, hypothesis, label) for the training set.
        - 'dev': A list of tuples lists (premise, hypothesis, label) for the development set.
        - 'test': A list of tuples lists (premise, hypothesis, label) for the test set.

    @param snli_path: Path to the SNLI dataset.

    @return: A dictionary containing the dataset.
    """
    LOGGER.info("Loading SNLI dataset")
    snli_dataset = {"train": {}, "dev": {}, "test": {}}

    def valid_label(sample):
        return sample[0] in SNLI_LABELS

    for _dataset in DATASETS:
        dataset_file_path = os.path.join(snli_path, f"snli_1.0_{_dataset}.txt")
        with open(dataset_file_path, "r", encoding='utf-8') as dataset_file:
            samples = [line.split("\t") for line in dataset_file]
        snli_dataset[_dataset]["label"] = [sample[0].strip() for sample in samples if valid_label(sample)]
        snli_dataset[_dataset]["source"] = [" ".join(sample[1].replace("(", "").replace(")", "").strip().split())
                                            for sample in samples if valid_label(sample)]
        snli_dataset[_dataset]["target"] = [" ".join(sample[2].replace("(", "").replace(")", "").strip().split())
                                            for sample in samples if valid_label(sample)]

    LOGGER.info("Done loading SNLI dataset")
    return snli_dataset


def preprocess_snli_dataset(snli_dataset: dict, glove_path: str,
                            output_path: str, max_sequence_length: int,
                            shuffle: bool, batch_size: int) -> Tuple[DataIndexer, DataIndexer, dict]:
    """
    @brief: Pre-processes the SNLI dataset.

    This method pre-processes the SNLI dataset.
    It loads the glove vectors for the words in the dataset and saves the dataset to an hdf5 file.
    The hdf5 file contains the following keys:
        - 'source': A list of lists of word indices for the premise sentences.
        - 'target': A list of lists of word indices for the hypothesis sentences.
        - 'label': A list of labels for the sentences.
        - 'source_length': A list of the length of the premise sentences.
        - 'target_length': A list of the length of the hypothesis sentences.
        - 'label_size': A list of the number of labels.
        - 'batch_length': A list of the batch sizes.
        - 'batch_index': A list of the batch indices.
        - 'source_size': A list of the number of words in the source vocabulary.
        - 'target_size': A list of the number of words in the target vocabulary.

    @param snli_dataset: The SNLI dataset.
    @param glove_path: Path to the glove vectors.
    @param output_path: Path to the output folder.
    @param max_sequence_length: The maximum sequence length.

    @return: Tuple of word_indexer, label_indexer, statistics.
    """
    LOGGER.info("Preprocessing dataset")

    statistics = {}
    word_indexer = DataIndexer([BLANK_TOKEN, UNKNOWN_TOKEN, START_TOKEN, END_TOKEN])
    label_indexer = DataIndexer([BLANK_TOKEN, UNKNOWN_TOKEN, START_TOKEN, END_TOKEN])
    label_indexer.dictionary = {}
    glove_vocabulary = load_glove_vocabulary(glove_path)

    for index in range(1, OOV_NUM_TOKENS + 1):
        oov_word = OOV_TOKEN_FORMAT.format(index=index)
        word_indexer.vocabulary[oov_word] += 1

    statistics["num_sentences"] = {"train": 0, "dev": 0, "test": 0}
    for dataset in DATASETS:
        statistics["num_sentences"][dataset] = update_vocabulary(
            dataset=snli_dataset[dataset], sequence_length=max_sequence_length,
            word_indexer=word_indexer, label_indexer=label_indexer, glove_vocabulary=glove_vocabulary)

    word_indexer.filter_vocabulary(0, True)
    label_indexer.filter_vocabulary(len(SNLI_LABELS))

    statistics["num_words"] = len(word_indexer)
    statistics["num_labels"] = len(label_indexer)
    statistics["filtered_sentences"] = {"train": 0, "dev": 0, "test": 0}

    max_sentence_length = 0
    for dataset in DATASETS:
        max_sentence_length = write_hdf5_dataset_file(
            set_type=dataset, dataset=snli_dataset[dataset], sequence_length=max_sequence_length,
            num_sentences=statistics["num_sentences"][dataset], shuffle=shuffle,
            batch_size=batch_size, output_path=os.path.join(output_path, f"{dataset}.hdf5"),
            max_sequence_length=max_sentence_length,
            word_indexer=word_indexer, label_indexer=label_indexer, statistics=statistics)

    statistics["max_sentence_length"] = max_sentence_length
    LOGGER.info("Done Preprocessing dataset")

    return word_indexer, label_indexer, statistics


def preprocess_word_embeddings(word_indexer: DataIndexer, glove_path: str, output_path: str) -> None:
    """
    @brief: Pre-processes the word embeddings.

    This method pre-processes the word embeddings.
    It loads the glove vectors for the words in the dataset and saves the embeddings to an hdf5 file.
    The hdf5 file contains the following keys:
        - 'word_embedding_vectors': A numpy array containing the word embeddings.

    @param word_indexer: The word indexer.
    @param glove_path: Path to the glove vectors file.
    @param output_path: Path to the output folder.

    @return: None.
    """
    LOGGER.info("Preprocessing word embeddings")

    vocabulary = word_indexer.dictionary
    word2idx = {word: index for word, index in vocabulary.items()}
    word2vec_vectors = np.random.normal(size=(len(vocabulary), GLOVE_DIMENSION))
    word2vec = load_glove_vectors(glove_path=glove_path, vocabulary=word2idx)

    for word, vector in word2vec.items():
        word2vec_vectors[word2idx[word] - 1] = vector

    for i in range(len(word2vec_vectors)):
        word2vec_vectors[i] = word2vec_vectors[i] / np.linalg.norm(word2vec_vectors[i])

    glove_file_path = os.path.join(output_path, "glove.hdf5")
    with h5py.File(glove_file_path, "w") as glove_file:
        glove_file["word_embedding_vectors"] = np.array(word2vec_vectors)

    LOGGER.info("Done Preprocessing word embeddings")


def prepare_dataset(snli_path: str, glove_path: str, output_path: str,
                    batch_size: int, shuffle: bool, max_sequence_length: int) -> None:
    """
    @brief: Prepares the dataset.

    It loads the SNLI dataset and pre-processes it.
    It loads the glove vectors for the words and pre-processes the word embeddings.
    Finally, it saves the pre-processed dataset and word embeddings to hdf5 files.

    @param snli_path: The path to the SNLI dataset.
    @param glove_path: The path to the GLOVE word embeddings.
    @param output_path: The path to the output folder.
    @param batch_size: The batch size.
    @param shuffle: Whether to shuffle the dataset.
    @param max_sequence_length: The maximum sequence length, other sequences will be dropped.

    @return: None.
    """
    LOGGER.info("Preparing dataset")

    snli_dataset = load_snli_dataset(snli_path)
    word_indexer, _, statistics = preprocess_snli_dataset(
        snli_dataset, glove_path, output_path, max_sequence_length, shuffle, batch_size)
    preprocess_word_embeddings(word_indexer, glove_path, output_path)

    LOGGER.info("Dataset Statistics:")
    for key, value in statistics.items():
        LOGGER.info(f"{key}: {value}")

    LOGGER.info("Done preparing dataset")


def load_train_dataset(device: str, train_filename: str,
                       dev_filename: str) -> Tuple[SNLIDataset, SNLIDataset]:
    """
    @brief: Loads the training dataset.

    @param device: The device to load the dataset to.
    @param train_filename: The filename of the training dataset.
    @param dev_filename: The filename of the dev dataset.

    @return: The train and test data loaders.
    """
    LOGGER.info("Loading train/dev datasets")
    train_dataset = SNLIDataset(device=device, filename=train_filename)
    dev_dataset = SNLIDataset(device=device, filename=dev_filename)
    LOGGER.info("Done loading train/dev datasets")

    return train_dataset, dev_dataset


def load_test_dataset(device: str, test_dataset_filename: str) -> SNLIDataset:
    """
    @brief: Loads the test dataset.

    @param device: The device to load the dataset to.
    @param test_dataset_filename: The filename of the test dataset.

    @return: The test data loader.
    """
    LOGGER.info("Loading test datasets")
    test_dataset = SNLIDataset(device=device, filename=test_dataset_filename)
    LOGGER.info("Done loading test dataset")

    return test_dataset

#################################################################################################################
#                                                 Model Utilities                                               #
#################################################################################################################


def get_norms(model: torch.nn.Module) -> Tuple[float, float]:
    """
    @brief: Returns the normals of the gradients/parameters of the model.

    @param model: The model.

    @return: The norms of the gradients/parameters.
    """
    gradient_norm = 0.0
    parameter_norm = 0.0
    linear_modules = [module for module in model.modules() if isinstance(module, torch.nn.Linear)]

    for module in linear_modules:
        gradient_norm += module.weight.grad.data.norm() ** 2
        parameter_norm += module.weight.data.norm() ** 2
        if module.bias is not None:
            gradient_norm += module.bias.grad.data.norm() ** 2
            parameter_norm += module.bias.data.norm() ** 2

    return gradient_norm, parameter_norm


def clip_gradients(model: torch.nn.Module, max_grad_norm: int, gradient_norm: float) -> None:
    """
    @brief: Clips the gradients of the model.

    @param model: The model.
    @param max_grad_norm: The gradient clip value.
    @param gradient_norm: The norm of the gradients.
    """
    shrinkage = max_grad_norm / gradient_norm
    if shrinkage >= 1:
        return

    linear_modules = [module for module in model.modules() if isinstance(module, torch.nn.Linear)]
    for module in linear_modules:
        module.weight.grad.data *= shrinkage
        if module.bias is not None:
            module.bias.grad.data *= shrinkage


def train(train_dataset: SNLIDataset, model: torch.nn.Module,
          optimizer: torch.optim.Optimizer, max_grad_norm: int = 0, log: bool = False,
          log_train_interval: int = 1000, log_norms: bool = False) -> Tuple[float, float]:
    """
    @brief: Trains the model.

    @param train_dataset: The train dataset.
    @param model: The model to train.
    @param optimizer: The optimizer to use.
    @param max_grad_norm: The gradient clip value.
    @param log: Whether to log the training progress.
    @param log_train_interval: The interval to log the training progress.
    @param log_norms: Whether to log the norms of the gradients and parameters.

    @return: Average train loss and accuracy tuple.
    """

    model.train()
    total_loss = total_correct = log_loss = log_accuracy = log_num_of_samples = log_batch_counter = samples_counter = 0
    log_interval_counter = 1
    train_losses, train_accuracies = [], []
    sample_counter_log = "({log_interval_counter}) [{samples_counter:>5d}/{train_dataset_len:>5d}] --> "
    error_log = "Accuracy: {train_avg_accuracy:>0.3f} [%] ; Loss: {train_avg_loss:>6f} "
    norm_log = "| Gradient Norm: {gradient_norm:>0.2f} ; Parameter Norm: {parameter_norm:>0.2f} " if log_norms is True else ""
    duration_log = "| Duration: {test_duration} [s]"
    log_message_format = sample_counter_log + error_log + norm_log + duration_log

    start_timestamp = get_time_now("%H:%M:%S.%f")
    for batch, (source, target, label) in enumerate(train_dataset):
        batch_size = len(label)
        log_num_of_samples += batch_size

        # Forward pass:
        optimizer.zero_grad()

        y_prob = model(source, target)
        y_pred = model.predict_from_probabilities(y_prob)

        # Compute prediction error:
        loss = model.loss(y_pred=y_prob, y_true=label)
        current_loss = loss.sum().item() * batch_size
        total_loss += current_loss
        log_loss += current_loss

        current_correct = (y_pred == label).type(torch.int64).sum().item()
        total_correct += current_correct
        log_accuracy += current_correct

        # Backward pass:
        loss.backward()

        # Clip the gradients:
        if max_grad_norm > 0 or log_norms is True:
            gradient_norm, parameter_norm = get_norms(model)
        if max_grad_norm > 0:
            clip_gradients(model, max_grad_norm, gradient_norm)

        optimizer.step()

        # Log progress on dev each samples_log_interval samples:
        if log is True and (batch + 1) % log_train_interval == 0:
            end_timestamp = get_time_now("%H:%M:%S.%f")
            time_duration_str = get_duration_in_seconds(start_timestamp, end_timestamp)

            train_avg_loss = float(log_loss / log_num_of_samples)
            train_avg_accuracy = float((log_accuracy / log_num_of_samples) * 100)
            train_losses.append((train_avg_loss, log_batch_counter))
            train_accuracies.append((train_avg_accuracy, log_batch_counter))

            log_message = log_message_format.format(
                samples_counter=samples_counter, train_dataset_len=len(train_dataset),
                train_avg_accuracy=train_avg_accuracy, train_avg_loss=train_avg_loss, log_interval_counter=log_interval_counter,
                gradient_norm=gradient_norm, parameter_norm=parameter_norm, test_duration=time_duration_str)
            LOGGER.info(log_message)

            log_batch_counter = log_loss = log_accuracy = log_num_of_samples = 0
            log_interval_counter += 1
            start_timestamp = get_time_now("%H:%M:%S.%f")

        samples_counter += batch_size
        log_batch_counter += 1

    total_avg_loss = float(total_loss / samples_counter)
    total_avg_accuracy = float((total_correct / samples_counter) * 100)

    return total_avg_loss, total_avg_accuracy, train_losses, train_accuracies


def test(dataset: SNLIDataset, model: torch.nn.Module, results_break_down: bool = False) -> Tuple[float, float, dict]:
    """
    @brief: Tests the model.

    @param dataset: The dataset to use.
    @param model: The model to test.
    @param results_break_down: Break down the results by class.

    @return: Average test loss and accuracy tuple, and the results break down(optional).
    """
    model.eval()
    predictions_counter = total_loss = correct = 0
    predictions_per_label_index = Counter({index: 0 for index in dataset.label2idx.values()})

    with torch.no_grad():
        for _, (source, target, label) in enumerate(dataset):
            # Forward pass:
            y_prob = model(source, target)
            y_pred = model.predict_from_probabilities(y_prob)

            # Compute prediction error:
            loss = model.loss(y_pred=y_prob, y_true=label)
            total_loss += loss.sum().item()

            # Compute accuracy:
            correct_mask = (y_pred == label)
            correct += correct_mask.type(torch.int64).sum().item()
            predictions_counter += len(label)

            # Break down the results by class:
            if results_break_down is True:
                predictions_per_label_index.update(y_pred[correct_mask].tolist())

    # Compute total average loss and accuracy:
    total_avg_loss = float(total_loss / predictions_counter)
    total_avg_accuracy = float((correct / predictions_counter) * 100)

    # Compute break down results:
    break_down_results = None
    if results_break_down is True:
        break_down_results = {
            label:
            round((predictions_per_label_index[index] / dataset.total_per_label[index]) * 100, 2)
            for label, index in dataset.label2idx.items()}

    return total_avg_loss, total_avg_accuracy, break_down_results


def fit(train_dataset: SNLIDataset, dev_dataset: SNLIDataset, model: torch.nn.Module,
        optimizer: torch.optim.Optimizer, max_grad_norm: int = 0, epochs: int = 250, log_train: bool = False,
        log_epoch: bool = True, log_train_interval: int = 1000, log_norms: bool = False,
        shuffle: bool = False, model_path: str = "") -> Tuple[torch.nn.Module, Tuple[list, list, list, list]]:
    """
    @brief: Fits the model.

    @param train_dataset: The train dataset to use.
    @param dev_dataset: The dev dataset to use.
    @param model: The model to fit.
    @param optimizer: The optimizer to use.
    @param max_grad_norm: The maximum gradient norm.
    @param epochs: The number of epochs to fit.
    @param log_train: Whether to log the training progress.
    @param log_epoch: Whether to log the error of train/dev progress.
    @param log_train_interval: The interval to log the training progress.
    @param log_norms: Whether to log the gradient and parameter norms.
    @param shuffle: Whether to shuffle the dataset before each epoch.
    @param model_path: The path to save the model.

    @return: A tuple of the trained model and the training history.
    """
    LOGGER.info("Start training")

    total_start_timestamp = get_time_now()
    total_train_losses, total_train_accuracies = [], []
    total_dev_losses, total_dev_accuracies = [], []
    best_models = []
    train_error_log = ": Train => Accuracy: {avg_train_accuracy:>0.4f} [%] ; Loss: {avg_train_loss:>6f}"
    dev_error_log = " | Dev => Accuracy: {dev_accuracy:>0.4f} [%] ; Loss: {dev_loss:>6f}"
    duration_log = " | Duration: {epoch_duration}"
    log_epoch_format = f"{train_error_log} {dev_error_log} {duration_log}"

    for epoch in range(1, epochs + 1):
        # Mark start of epoch:
        start_timestamp = get_time_now()

        # Shuffle the dataset:
        if shuffle is True:
            train_dataset.shuffle()

        # Train:
        if log_train is True:
            LOGGER.info(LOGGER_SEPARATOR_LINE)
            LOGGER.info(f"Epoch - {epoch}/{epochs} Train statistics:")
            LOGGER.info(LOGGER_SEPARATOR_LINE)

        avg_train_loss, avg_train_accuracy, _, _ = train(
            train_dataset=train_dataset, model=model, optimizer=optimizer, max_grad_norm=max_grad_norm,
            log=log_train, log_train_interval=log_train_interval, log_norms=log_norms)

        # Evaluate:
        avg_dev_loss, avg_dev_accuracy, _ = test(dataset=dev_dataset, model=model)

        end_timestamp = get_time_now()

        # Save fit statistics:
        total_train_losses.append(avg_train_loss)
        total_train_accuracies.append(avg_train_accuracy)
        total_dev_losses.append(avg_dev_loss)
        total_dev_accuracies.append(avg_dev_accuracy)
        epoch_duration = end_timestamp - start_timestamp

        # Log the epoch statistics:
        log_message = f"Epoch {epoch}/{epochs}"
        if log_epoch is True:
            log_message += log_epoch_format.format(
                avg_train_accuracy=avg_train_accuracy, avg_train_loss=avg_train_loss,
                dev_accuracy=avg_dev_accuracy, dev_loss=avg_dev_loss, epoch_duration=epoch_duration)
        LOGGER.info(LOGGER_SEPARATOR_LINE)
        LOGGER.info(log_message)

        if log_train is True:
            LOGGER.info(LOGGER_SEPARATOR_LINE)

        # Save best model:
        if avg_dev_accuracy >= max(total_dev_accuracies):
            model_filename = f"{model.__class__.__name__}_epoch_{epoch}_dev_accuracy_{avg_dev_accuracy:.2f}.pt"
            model_full_path = os.path.join(model_path, model_filename)
            statistics = (total_train_losses, total_train_accuracies, total_dev_losses, total_dev_accuracies)
            save_model(model_path=model_full_path, model=model, fit_statistics=statistics)
            best_models.append((epoch, avg_dev_accuracy, model_filename))
            LOGGER.info(f"Saved current best model: {model_full_path}")
            best_till_now = " | ".join([f"{epoch} - {accuracy:.2f}" for epoch, accuracy, _ in best_models])
            LOGGER.info(f"Best models till now - (epoch - dev_accuracy): {best_till_now}")

    statistics = (total_train_losses, total_train_accuracies, total_dev_losses, total_dev_accuracies)
    total_end_timestamp = get_time_now()
    total_duration = total_end_timestamp - total_start_timestamp

    final_best_model = best_models[-1]
    LOGGER.info("Final best model - Epoch: {0} | Dev Accuracy: {1:.2f} | Path: {2}".format(*final_best_model))
    LOGGER.info(f"Done training - Total duration: {total_duration}")

    return model, statistics


def save_plot_statistics(statistics: Tuple[list, list, list, list], plot_path: str = "") -> None:
    """
    @brief: Saves fit statistics plots.

    @param statistics: The fit statistics to plot (train loss, train accuracy, dev loss, dev accuracy).
    @param plot_path: The path to the directory to save the plots.

    @return: None.
    """
    LOGGER.info("Saving plots")

    train_losses, train_accuracies, dev_losses, dev_accuracies = statistics

    def plot_helper(stat: str, train_stats: list, dev_stats: list) -> None:
        _, ax = plt.subplots()
        ax.plot(train_stats, label=f"Train {stat}")
        ax.plot(dev_stats, label=f"Dev {stat}")
        ax.legend()
        ax.set_xlabel("Epoch")
        ax.set_ylabel(f"{stat}")
        ax.set_title(f"Training and Dev {stat}")
        plt.savefig(f"{plot_path}/train_dev_{stat.lower()}.png")
        plt.close()

    plot_helper("Loss", train_losses, dev_losses)
    plot_helper("Accuracy", train_accuracies, dev_accuracies)

    LOGGER.info("Done saving plots")
