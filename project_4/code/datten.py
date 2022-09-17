"""
Application module for implementation of paper:
    "A Decomposable Attention Model for Natural Language Inference", 1606.01933v2.

Author: Simon Raviv.
"""
import datetime
import os
import sys
import argparse
import torch
import functools

from typing import Tuple

import lib_datten as lib

# Global variables:
MODULE_ROOT_DIR = os.path.dirname(__file__)
DATA_PARSER = "data"
MODEL_PARSER = "model"


class MLP(torch.nn.Module):
    """
    @brief: A multi-layer perceptron.
    """

    def __init__(self, device: str, input_dim: int, output_dim: int,
                 dropout: float, parameters_gaussian_std_init: float) -> None:
        """
        @brief: Initialize the class.

        @param device: Device to use.
        @param input_dim: Input dimension.
        @param output_dim: Output dimension.
        @param dropout: Dropout probability.
        @param parameters_gaussian_std_init: Standard deviation of the gaussian distribution used
                                             to initialize the parameters of the linear layer.

        @return: None.
        """
        super(MLP, self).__init__()
        self.device = device
        self.parameters_gaussian_init = (0, parameters_gaussian_std_init)
        self.fc = torch.nn.Sequential(
            torch.nn.Dropout(p=dropout),
            torch.nn.Linear(in_features=input_dim, out_features=output_dim, bias=True),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=dropout),
            torch.nn.Linear(in_features=output_dim, out_features=output_dim, bias=True),
            torch.nn.ReLU()
        )
        self._initialize_parameters()
        self.to(device)

    def _initialize_parameters(self) -> None:
        """
        @brief: Initialize the parameters.

        @return: None.
        """
        linear_modules = [module for module in self.modules() if isinstance(module, torch.nn.Linear)]
        for module in linear_modules:
            module.weight.data.normal_(*self.parameters_gaussian_init)
            module.bias.data.normal_(*self.parameters_gaussian_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        @brief: Forward pass.

        @param x: Input tensor.

        @return: Output tensor.
        """
        return self.fc(x)


class InputEncoder(torch.nn.Module):
    """
    @brief: Encoder model for input data.
    """

    def __init__(self, device: str, vocabulary_size: int, embedding_dim: int,
                 hidden_dim: int, parameters_gaussian_std_init: float, pretrained_embeddings: torch.Tensor) -> None:
        """
        @brief: Initialize the class.

        @param device: Device to use.
        @param vocabulary_size: Size of the vocabulary.
        @param embedding_dim: Dimension of the embedding.
        @param hidden_dim: Dimension of the hidden layer.
        @param parameters_gaussian_std_init: Standard deviation of the gaussian distribution used
                                             to initialize the parameters of the linear layer.
        @param pretrained_embeddings: Pretrained embeddings.

        @return: None.
        """
        super(InputEncoder, self).__init__()
        self.device = device
        self.vocabulary_size = vocabulary_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.parameters_gaussian_init = (0, parameters_gaussian_std_init)
        self.embedding = self._get_embeddings(pretrained_embeddings=pretrained_embeddings, freeze_pretrained=True)
        self.input_linear = torch.nn.Linear(in_features=self.embedding_dim, out_features=self.hidden_dim, bias=False)
        self._initialize_parameters()
        self.to(self.device)

    def _initialize_parameters(self) -> None:
        """
        @brief: Initialize the parameters.

        @return: None.
        """
        self.input_linear.weight.data.normal_(*self.parameters_gaussian_init)

    def _get_embeddings(self, pretrained_embeddings: torch.Tensor, freeze_pretrained: bool = True) -> torch.nn.Embedding:
        """
        @brief: Get the embeddings.

        @param pretrained_embeddings: Pretrained embeddings.
        @param freeze_pretrained: Freeze the pretrained embeddings.

        @return: The embeddings.
        """
        embedding = torch.nn.Embedding(num_embeddings=self.vocabulary_size, embedding_dim=self.embedding_dim)
        if pretrained_embeddings is not None:
            embedding.weight.data.copy_(pretrained_embeddings)
            embedding.weight.requires_grad = not freeze_pretrained

        return embedding

    def forward(self, sentence_1: torch.Tensor, sentence_2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        @brief: Forward pass of the model.

        @param sentence_1: Sentence 1.
        @param sentence_2: Sentence 2.

        @return: A tuple of the encoded sentences.
        """
        # Embed the input:
        embedded_sentence_1 = self.embedding(sentence_1).view(-1, self.embedding_dim)
        embedded_sentence_2 = self.embedding(sentence_2).view(-1, self.embedding_dim)

        # Linear transform:
        batch_size = sentence_1.size(0)
        encoded_sentence_1 = self.input_linear(embedded_sentence_1).view(batch_size, -1, self.hidden_dim)
        encoded_sentence_2 = self.input_linear(embedded_sentence_2).view(batch_size, -1, self.hidden_dim)

        return encoded_sentence_1, encoded_sentence_2


class Attention(torch.nn.Module):
    """
    @brief: Attention model.
    """

    def __init__(self, device: str, hidden_dim: int, num_of_labels: int, dropout: float,
                 parameters_gaussian_std_init: float) -> None:
        """
        @brief: Initialize the class.

        @param device: Device to use.
        @param hidden_dim: Dimension of the hidden layer.
        @param num_of_labels: Number of labels.
        @param dropout: Dropout probability.
        @param parameters_gaussian_std_init: Standard deviation of the gaussian distribution used
                                             to initialize the parameters of the linear layers.

        @return: None.
        """
        super(Attention, self).__init__()
        self.device = device
        self.hidden_dim = hidden_dim
        self.num_of_labels = num_of_labels
        self.dropout = dropout
        self.parameters_gaussian_init = (0, parameters_gaussian_std_init)
        self.f_mlp = MLP(device=device, input_dim=hidden_dim, output_dim=hidden_dim, dropout=dropout,
                         parameters_gaussian_std_init=parameters_gaussian_std_init)
        self.g_mlp = MLP(device=device, input_dim=2 * hidden_dim, output_dim=hidden_dim, dropout=dropout,
                         parameters_gaussian_std_init=parameters_gaussian_std_init)
        self.h_mlp = MLP(device=device, input_dim=2 * hidden_dim, output_dim=hidden_dim, dropout=dropout,
                         parameters_gaussian_std_init=parameters_gaussian_std_init)
        self.linear_output = torch.nn.Linear(in_features=hidden_dim, out_features=num_of_labels, bias=True)
        self.softmax = torch.nn.LogSoftmax(dim=1)
        self._initialize_parameters()
        self.to(self.device)

    def _initialize_parameters(self) -> None:
        """
        @brief: Initialize the parameters.

        @return: None.
        """
        linear_modules = [module for module in self.modules() if isinstance(module, torch.nn.Linear)]
        for module in linear_modules:
            module.weight.data.normal_(*self.parameters_gaussian_init)
            module.bias.data.normal_(*self.parameters_gaussian_init)

    def forward(self, encoded_sentence_1: torch.Tensor, encoded_sentence_2: torch.Tensor) -> torch.Tensor:
        """
        @brief: Forward pass of the model.

        @param encoded_sentence_1: Encoded sentence 1.
        @param encoded_sentence_2: Encoded sentence 2.

        @return: The probabilities of the labels.
        """
        length_1 = encoded_sentence_1.size(1)
        length_2 = encoded_sentence_2.size(1)

        # Attend:
        f1 = self.f_mlp(encoded_sentence_1).view(-1, length_1, self.hidden_dim)
        f2 = self.f_mlp(encoded_sentence_2).view(-1, length_2, self.hidden_dim)

        score_1 = torch.bmm(f1, f2.transpose(1, 2))
        probability_1 = torch.nn.functional.softmax(score_1.view(-1, length_2), dim=1).view(-1, length_1, length_2)

        score_2 = torch.transpose(score_1.contiguous(), 1, 2).contiguous()
        probability_2 = torch.nn.functional.softmax(score_2.view(-1, length_1), dim=1).view(-1, length_2, length_1)

        sentence_1_combined = torch.cat((encoded_sentence_1, torch.bmm(probability_1, encoded_sentence_2)), dim=2)
        sentence_2_combined = torch.cat((encoded_sentence_2, torch.bmm(probability_2, encoded_sentence_1)), dim=2)

        # Compare:
        g1 = self.g_mlp(sentence_1_combined.view(-1, 2 * self.hidden_dim)).view(-1, length_1, self.hidden_dim)
        g2 = self.g_mlp(sentence_2_combined.view(-1, 2 * self.hidden_dim)).view(-1, length_2, self.hidden_dim)

        sentence_1_output = g1.sum(dim=1).squeeze(1)
        sentence_2_output = g2.sum(dim=1).squeeze(1)

        # Aggregate:
        input_combined = torch.cat((sentence_1_output, sentence_2_output), dim=1)
        h = self.h_mlp(input_combined)

        # Predict:
        logits = self.linear_output(h)
        probabilities = self.softmax(logits)

        return probabilities


class DecomposableAttentionModel(torch.nn.Module):
    """
    @brief: A Decomposable Attention Model for Natural Language Inference.
    """

    def __init__(self, device: str, vocabulary_size: int, embedding_dim: int,
                 hidden_dim: int, parameters_gaussian_std_init: float,
                 pretrained_embeddings: torch.Tensor = None, num_of_labels: int = 3, dropout: float = 0) -> None:
        """
        @brief: Initialize the class.

        @param device: Device to use.
        @param vocabulary_size: Vocabulary size.
        @param embedding_dim: Embedding dimension.
        @param hidden_dim: Hidden dimension.
        @param parameters_gaussian_std_init: Standard deviation of the gaussian distribution used
                                             to initialize the parameters of the linear layers.
        @param pretrained_embeddings: Pretrained embeddings.
        @param num_of_labels: Number of labels.
        @param dropout: Dropout probability.

        @return: None.
        """
        super(DecomposableAttentionModel, self).__init__()
        self.device = device
        self.vocabulary_size = vocabulary_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.parameters_gaussian_std_init = parameters_gaussian_std_init
        self.parameters_gaussian_init = (0, parameters_gaussian_std_init)
        self.num_of_labels = num_of_labels
        self.dropout = dropout
        self.input_encoder = InputEncoder(
            device=device, vocabulary_size=vocabulary_size, embedding_dim=embedding_dim,
            hidden_dim=hidden_dim, parameters_gaussian_std_init=parameters_gaussian_std_init,
            pretrained_embeddings=pretrained_embeddings)
        self.intra_attention = Attention(
            device=device, hidden_dim=hidden_dim, num_of_labels=num_of_labels,
            dropout=dropout, parameters_gaussian_std_init=parameters_gaussian_std_init)
        self.to(device)

    def forward(self, sentence_1: torch.Tensor, sentence_2: torch.Tensor) -> torch.Tensor:
        """
        @brief: Forward propagation.

        @param sentence_1: Sentence 1.
        @param sentence_2: Sentence 2.

        @return: Output probabilities of the network.
        """
        # Encode:
        encoded_sentence_1, encoded_sentence_2 = self.input_encoder(sentence_1, sentence_2)

        # Attend:
        intra_attention_output = self.intra_attention(encoded_sentence_1, encoded_sentence_2)

        # Output:
        probabilities = intra_attention_output
        return probabilities

    def predict(self, sentence_1: torch.Tensor, sentence_2: torch.Tensor) -> torch.Tensor:
        """
        @brief: Predict the output.

        @param sentence_1: Input sentence 1.
        @param sentence_2: Input sentence 2.

        @return: The output prediction.
        """
        probabilities = self.forward(sentence_1, sentence_2)
        prediction = probabilities.argmax(dim=1)
        return prediction

    def predict_from_probabilities(self, y_prob: torch.Tensor) -> int:
        """
        @brief: Get the prediction.

        @param y_prob: Output probabilities.

        @return: The prediction.
        """
        prediction = y_prob.argmax(dim=1)
        return prediction

    def loss(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """
        @brief: Calculate the loss.

        @param y_pred: Output probabilities.
        @param y_true: True labels.

        @return: The loss.
        """
        loss = torch.functional.F.nll_loss(input=y_pred, target=y_true.long(), reduction='mean')
        return loss

    def get_non_trainable_parameters_state_dict(self) -> dict:
        """
        @brief: Get the state dict of the non trainable parameters.

        @return: The state dict of the parameters.
        """
        return {
            "vocabulary_size": self.vocabulary_size,
            "embedding_dim": self.embedding_dim,
            "hidden_dim": self.hidden_dim,
            "parameters_gaussian_std_init": self.parameters_gaussian_std_init,
            "num_of_labels": self.num_of_labels,
            "dropout": self.dropout
        }


class Optimizer:
    """
    @brief: A class for the optimizer.

    It will optimize the two modules of the main module in the same way.
    It will use 2 different optimizers for the two modules. As the optimizer
    used is Adagrad, the overall sum of the parameters affects the parameters
    at each update.
    """

    def __init__(self, model: DecomposableAttentionModel, lr: float, weight_decay: float,
                 initial_accumulator_value: float = 0) -> None:
        """
        @brief: Initialize the class.

        @param model: Model to optimize.
        @param lr: Learning rate.
        @param weight_decay: Weight decay.
        @param initial_accumulator_value: Initial value of the accumulator.

        @return: None.
        """
        super(Optimizer, self).__init__()
        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay
        self.initial_accumulator_value = initial_accumulator_value
        self.optimizers = []
        input_encoder_parameters = filter(lambda p: p.requires_grad, self.model.input_encoder.parameters())
        input_encoder_optimizer = torch.optim.Adagrad(
            input_encoder_parameters, lr=lr, weight_decay=weight_decay, initial_accumulator_value=initial_accumulator_value)
        self.optimizers.append(input_encoder_optimizer)
        intra_attention_parameters = self.model.intra_attention.parameters()
        intra_attention_optimizer = torch.optim.Adagrad(
            intra_attention_parameters, lr=lr, weight_decay=weight_decay, initial_accumulator_value=initial_accumulator_value)
        self.optimizers.append(intra_attention_optimizer)

    def zero_grad(self) -> None:
        """
        @brief: Zero the gradients.

        @return: None.
        """
        for optimizer in self.optimizers:
            optimizer.zero_grad()

    def step(self) -> None:
        """
        @brief: Update the parameters.

        @return: None.
        """
        for optimizer in self.optimizers:
            optimizer.step()


def parse_cli(description: str = "") -> argparse.Namespace:
    """
    @brief: Parses the command line arguments.

    @param description: The description of the program.

    @return: The parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(description=description)
    subparsers = parser.add_subparsers(title='subcommands', dest="subparser_name")
    data_parser = subparsers.add_parser(DATA_PARSER, help="Data processing command.")
    model_parser = subparsers.add_parser(MODEL_PARSER, help="Model command.")

    # Data processing related arguments:
    data_parser.add_argument("--debug", action="store_true", default=False,
                             help="Run in debug mode, set parameters in the module file.")
    data_parser.add_argument("--log-file", type=str, default="datten.log",
                             help="The file path to log the output to.")
    data_parser.add_argument("--no-log-console", action="store_true", default=False,
                             help="Don't to log to console.")
    data_parser.add_argument("--snli-path", type=str, default="",
                             help="The path to the unzipped SNLI snli_1.0 dataset folder.")
    data_parser.add_argument("--glove-path", type=str, default="",
                             help="The path to the unzipped GLOVE 840B.300d word embeddings file.")
    data_parser.add_argument("--output-path", type=str, default="",
                             help="The path to the output folder.")
    data_parser.add_argument("--batch-size", type=int, default=32,
                             help="The batch size to use.")
    data_parser.add_argument("--shuffle", action="store_true", default=False,
                             help="Shuffle the data.")
    data_parser.add_argument("--max-sequence-length", type=int, default=100,
                             help="The maximum sequence length to use, other sequences will be dropped.")

    # Model fitting related arguments:
    model_fit_group = model_parser.add_mutually_exclusive_group()
    model_parser.add_argument("--debug", action="store_true", default=False,
                              help="Run in debug mode, set parameters in the module file.")
    model_parser.add_argument("--log-file", type=str, default="datten.log",
                              help="The file path to log the output to.")
    model_parser.add_argument("--no-log-console", action="store_true", default=False,
                              help="Don't to log to console.")
    model_parser.add_argument("--log-train", action="store_true", default=False,
                              help="Print logs of the fitting progress.")
    model_parser.add_argument("--log-train-interval", type=int, default=1000,
                              help="The interval to print the logs of the fitting progress during training.")
    model_parser.add_argument("--log-epoch", action="store_true", default=False,
                              help="Print logs of the fitting train/dev progress.")
    model_parser.add_argument("--log-norms", action="store_true", default=False,
                              help="Print logs of the parameters and gradients normals.")
    model_fit_group.add_argument("--fit", action="store_true", default=False,
                                 help="Fit the model.")
    model_fit_group.add_argument("--load-model-path", type=str, default="",
                                 help="The path to the model to load.")
    model_parser.add_argument("--save-model-path", type=str, default="",
                              help="The path to save the best models to.")
    model_parser.add_argument("--evaluate", action="store_true", default=False,
                              help="Evaluate on test file.")
    model_parser.add_argument("--train-file", type=str, default="train",
                              help="The name of the train file contains the train dataset.")
    model_parser.add_argument("--dev-file", type=str, default="train",
                              help="The name of the dev file contains the dev dataset.")
    model_parser.add_argument("--test-file", type=str, default="train",
                              help="The name of the test file contains the test dataset.")
    model_parser.add_argument("--results-break-down", action="store_true", default=False,
                              help="Print accuracy per label during test time.")
    model_parser.add_argument("--word-embeddings-file", type=str, default="",
                              help="The path to GLOVE word embeddings file.")
    model_parser.add_argument("--plot-path", type=str, default="",
                              help="Path to save plots of fitting statistics.")
    model_parser.add_argument("--gaussian-std", type=float, default=0.01,
                              help="The standard deviation of the gaussian initialization, mean is 0.")
    model_parser.add_argument("--initial-accumulator-value", type=float, default=0,
                              help="The initial accumulator value for the Adagrad optimizer.")
    model_parser.add_argument("--shuffle", action="store_true", default=False,
                              help="Shuffle the data before each epoch.")
    model_parser.add_argument("--epochs", type=int, default=250,
                              help="The number of epochs to fit.")
    model_parser.add_argument("--embedding-dim", type=int, default=300,
                              help="The size of the word embedding vectors.")
    model_parser.add_argument("--mlp-hidden-dim", type=int, default=128,
                              help="The hidden layer dimension of the MLP layer.")
    model_parser.add_argument("--lr", type=float, default=0.003,
                              help="The learning rate to use.")
    model_parser.add_argument("--dropout", type=float, default=0,
                              help="The dropout probability to use.")
    model_parser.add_argument("--weight-decay", type=float, default=0,
                              help="The weight decay to use.")
    model_parser.add_argument("--max-grad-norm", type=float, default=5,
                              help="The maximum gradient norm to use, gradients over this value will be clipped.")

    arguments = parser.parse_args()
    return arguments


def main(cli_args) -> int:
    """
    @brief: Main function for the module.

    @return: Exit code.
    """
    # Set the seed for reproducibility:
    lib.seed_everything(lib.SEED)

    # Initialize the logger:
    lib.initialize_logger(log_path=cli_args.log_file, console=not cli_args.no_log_console)

    # Set the device to run on:
    device = lib.get_device()

    # Log arguments:
    lib.LOGGER.info(lib.LOGGER_SEPARATOR_LINE)
    lib.LOGGER.info(f"CLI Arguments:")
    lib.LOGGER.info(lib.LOGGER_SEPARATOR_LINE)
    args = "\n" + "\n".join([f"{argument}: {value}" for argument, value in vars(cli_args).items()])
    lib.LOGGER.info(args)
    lib.LOGGER.info(lib.LOGGER_SEPARATOR_LINE)

    # Handle dataset:
    if cli_args.subparser_name == DATA_PARSER:
        lib.prepare_dataset(snli_path=cli_args.snli_path, glove_path=cli_args.glove_path,
                            output_path=cli_args.output_path, batch_size=cli_args.batch_size,
                            shuffle=cli_args.shuffle, max_sequence_length=cli_args.max_sequence_length)
    elif cli_args.subparser_name == MODEL_PARSER:
        # Load the word embeddings:
        word_embeddings = lib.Word2Vector(cli_args.word_embeddings_file).get_word_vectors()
        lib.LOGGER.info(f"Loaded {len(word_embeddings)} word embeddings.")

        # Train the model:
        if cli_args.fit is True:
            # Load the dataset:
            train_dataset, dev_dataset = lib.load_train_dataset(
                device=device, train_filename=cli_args.train_file, dev_filename=cli_args.dev_file)
            lib.LOGGER.info(f"Loaded {len(train_dataset)} train samples")
            lib.LOGGER.info(f"Loaded {len(dev_dataset)} dev samples")

            # Create neural network:
            model = DecomposableAttentionModel(
                device=device,
                vocabulary_size=len(word_embeddings),
                embedding_dim=cli_args.embedding_dim,
                hidden_dim=cli_args.mlp_hidden_dim,
                parameters_gaussian_std_init=cli_args.gaussian_std,
                pretrained_embeddings=word_embeddings,
                num_of_labels=len(lib.SNLI_LABELS),
                dropout=cli_args.dropout
            )
            lib.LOGGER.info(lib.LOGGER_SEPARATOR_LINE)
            lib.LOGGER.info(f"\n{model}")
            lib.LOGGER.info(lib.LOGGER_SEPARATOR_LINE)

            # fit the model:
            optimizer = Optimizer(model=model, lr=cli_args.lr, weight_decay=cli_args.weight_decay)

            model, fit_statistics = lib.fit(
                train_dataset=train_dataset, dev_dataset=dev_dataset, model=model,
                optimizer=optimizer, max_grad_norm=cli_args.max_grad_norm, epochs=cli_args.epochs,
                log_train=cli_args.log_train, log_epoch=cli_args.log_epoch,
                log_train_interval=cli_args.log_train_interval, log_norms=cli_args.log_norms,
                shuffle=cli_args.shuffle, model_path=cli_args.save_model_path,
            )

            # Save plot of training history:
            if cli_args.plot_path != "":
                lib.save_plot_statistics(statistics=fit_statistics, plot_path=cli_args.plot_path)
        elif cli_args.load_model_path != "":
            # Load the model:
            model, _ = lib.load_model(
                device=device, model_path=cli_args.load_model_path,
                pretrained_embeddings=word_embeddings, model_class=DecomposableAttentionModel)
            lib.LOGGER.info(lib.LOGGER_SEPARATOR_LINE)
            lib.LOGGER.info(f"\n{model}")
            lib.LOGGER.info(lib.LOGGER_SEPARATOR_LINE)

        # Evaluate the model:
        if (cli_args.fit is True or cli_args.load_model_path != "") and (cli_args.evaluate is True and cli_args.test_file != ""):
            test_dataset = lib.load_test_dataset(device=device, test_dataset_filename=cli_args.test_file)
            lib.LOGGER.info(f"Loaded {len(test_dataset)} test samples")
            _, avg_test_accuracy, accuracy_per_label = lib.test(
                dataset=test_dataset, model=model, results_break_down=cli_args.results_break_down)
            lib.LOGGER.info(lib.LOGGER_SEPARATOR_LINE)
            lib.LOGGER.info(f"Test --> Accuracy: {avg_test_accuracy:.4f} [%]")
            if cli_args.results_break_down is True:
                lib.LOGGER.info(f"Test --> Accuracy [%] per label: {accuracy_per_label}")
            lib.LOGGER.info(lib.LOGGER_SEPARATOR_LINE)

    return 0


def set_debug_cli_args(cli_args: argparse.Namespace) -> argparse.Namespace:
    """
    @brief: Sets debug CLI arguments.

    @param cli_args: The CLI arguments.

    @return: The CLI arguments.
    """
    file_path = functools.partial(os.path.join, MODULE_ROOT_DIR, "..")
    TIME_FORMAT = "%d_%m_%Y_%H_%M_%S"
    timestamp = datetime.datetime.now().strftime(TIME_FORMAT)
    output_debug_dir = os.path.join("output", "debug", timestamp, "")
    os.makedirs(output_debug_dir)

    if cli_args.subparser_name == DATA_PARSER:
        cli_args.debug = True
        cli_args.log_file = file_path(output_debug_dir, "datten.log")
        cli_args.no_log_console = False
        cli_args.snli_path = file_path("data", "snli", "snli_1.0")
        cli_args.glove_path = file_path("data", "glove", "glove.840B.300d.txt")
        cli_args.output_path = file_path("data", "processed_dataset")
        cli_args.batch_size = 32
        cli_args.shuffle = False
        cli_args.max_sequence_length = 100
    elif cli_args.subparser_name == MODEL_PARSER:
        cli_args.debug = True
        cli_args.log_file = file_path(output_debug_dir, "datten.log")
        cli_args.no_log_console = False
        cli_args.log_train = True
        cli_args.log_train_interval = 2000
        cli_args.log_epoch = True
        cli_args.log_norms = False
        cli_args.fit = True
        cli_args.load_model_path = ""
        cli_args.save_model_path = file_path(output_debug_dir)
        cli_args.evaluate = True
        cli_args.train_file = file_path("data", "processed_dataset", "train.hdf5")
        cli_args.dev_file = file_path("data", "processed_dataset", "dev.hdf5")
        cli_args.test_file = file_path("data", "processed_dataset", "test.hdf5")
        cli_args.results_break_down = True
        cli_args.word_embeddings_file = file_path("data", "processed_dataset", "glove.hdf5")
        cli_args.plot_path = file_path(output_debug_dir)
        cli_args.gaussian_std = 0.01
        cli_args.initial_accumulator_value = 0
        cli_args.shuffle = True
        cli_args.epochs = 250
        cli_args.embedding_dim = 300
        cli_args.mlp_hidden_dim = 300
        cli_args.lr = 0.05
        cli_args.dropout = 0.2
        cli_args.weight_decay = 0.00001
        cli_args.max_grad_norm = 5

    return cli_args


if __name__ == "__main__":
    try:
        exit_code = 0
        cli_args = parse_cli(description="A Decomposable Attention Model for Natural Language Inference")

        # Run in production mode:
        if cli_args.subparser_name is not None and cli_args.debug is False:
            exit_code = main(cli_args)
            sys.exit(exit_code)
    except Exception as error:
        lib.LOGGER.error(error)
        exit_code = 1
        sys.exit(exit_code)

    # Run in debug mode, ignore all other CLI arguments:
    cli_args = set_debug_cli_args(cli_args)
    lib.LOGGER.info(cli_args)
    exit_code = main(cli_args)
    sys.exit(exit_code)
