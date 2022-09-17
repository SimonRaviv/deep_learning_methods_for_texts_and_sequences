"""
A module to train a BiLSTM model.

For an input sequence w1, ..., wn, represent each item as a vector xi = repr(wi).
Then, feed these representations through a biLSTM, resulting in n vectors b1, ..., bn where:
bi = biLSTM(x1, ..., xn; i) = LSTM_F (x1, ..., xi) ◦ LSTM_B(xn, ..., xi)
These will be fed to another layer of biLSTM, resulting in vectors b'1, ..., b'n where b'i = biLSTM(b1, ..., bn; i).
Each vector b'i will be fed into a linear layer followed by a softmax for predicting the label yi.
Use the cross-entropy loss.

Each word will be represented in one of the following options:
(a) an embedding vector: repr(wi) = E[wi]
(b) a character-level LSTM: repr(wi) = repr(c1; c2, ..., cmi) = LSTM_C(E[c1], ..., E[cmi ]).
(c) the embeddings+subword representation used in assignment 2.
(d) a concatenation of (a) and (b) followed by a linear layer.

Author:
Simon Raviv.
"""
import os
import sys
import argparse
import torch

import lib_rnn as lib

# Global variables:
MODULE_ROOT_DIR = os.path.dirname(__file__)


class BiLSTMTagger(torch.nn.Module):
    """
    @brief: BiLSTM sequence tagger model.
    """

    def __init__(self, device: str = "", word_vocabulary_size: int = -1, embedding_dim: int = -1,
                 lstm_hidden_size: int = -1, fc_output_size: int = -1, padding_token_index: int = -1,
                 padding_label_index: int = -1, word_representation_type: str = "",
                 cbow_word2subword_prefix_idx: dict = {}, cbow_word2subword_postfix_idx: dict = {},
                 char_vocabulary_size: int = -1, padding_char_index: int = -1, word_idx2chars_idx: dict = {},
                 dropout_probability: int = 0) -> None:
        """
        @brief: Initialize the class.

        @param device: Device to use.
        @param word_vocabulary_size: Size of the word vocabulary.
        @param embedding_dim: Size of the input sequence.
        @param lstm_hidden_size: Size of the hidden layer of the LSTM.
        @param fc_output_size: Size of the output layer of the linear layer.
        @param padding_token_index: Index of the padding token.
        @param padding_label_index: Index of the padding label.
        @param word_representation_type: Type of word representation, see @ref lib.WORD_REPRESENTATION_TYPES_CLI.
        @param cbow_word2subword_prefix_idx: Dictionary of word to subword prefix mapping.
        @param cbow_word2subword_postfix_idx: Dictionary of word to subword postfix mapping.
        @param char_vocabulary_size: Size of the character vocabulary.
        @param padding_char_index: Index of the character padding token.
        @param word_idx2chars_idx: Dictionary of word to characters mapping.
        @param dropout_probability: Dropout probability for linear/MLP layers.

        @return: None.
        """
        super(BiLSTMTagger, self).__init__()
        self.device = device
        self.word_vocabulary_size = word_vocabulary_size
        self.embedding_dim = embedding_dim
        self.lstm_hidden_size = lstm_hidden_size
        self.fc_output_size = fc_output_size
        self.padding_token_index = padding_token_index
        self.padding_label_index = padding_label_index
        self.word_representation_type = word_representation_type
        self.cbow_word2subword_prefix_idx = cbow_word2subword_prefix_idx
        self.cbow_word2subword_postfix_idx = cbow_word2subword_postfix_idx
        self.char_vocabulary_size = char_vocabulary_size
        self.padding_char_index = padding_char_index
        self.word_idx2chars_idx = word_idx2chars_idx
        self.dropout_probability = dropout_probability
        self._initialize_word_representation_factory()
        repr_layer, repr_kwargs, repr_dim = self.representation_factory[self.word_representation_type]
        self.word_representation = repr_layer(**repr_kwargs)
        self.bi_lstm_layer_1 = lib.BiLSTM(
            input_size=repr_dim, hidden_size=lstm_hidden_size,
            device=device, padding_index=padding_token_index, as_transducer=True)
        self.bi_lstm_layer_2 = lib.BiLSTM(
            input_size=(2 * lstm_hidden_size), hidden_size=lstm_hidden_size,
            device=device, padding_index=padding_token_index, as_transducer=True)
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(in_features=(2 * lstm_hidden_size), out_features=fc_output_size),
            torch.nn.Dropout(p=dropout_probability),
        )
        self.softmax = torch.nn.Softmax(dim=2)
        self.to(device)

    def _initialize_word_representation_factory(self) -> None:
        """
        @brief: Initialize word representation factory.

        This method initializes word representation factory according to the word representation
        option chosen.

        @return: None.
        """
        self.representation_factory = {
            lib.WORD_REPRESENTATION_TYPES_CLI[0]: (
                torch.nn.Embedding,
                {"num_embeddings": self.word_vocabulary_size, "embedding_dim": self.embedding_dim,
                 "padding_idx": self.padding_token_index},
                self.embedding_dim),
            lib.WORD_REPRESENTATION_TYPES_CLI[1]: (
                lib.CharacterLSTM,
                {"embedding_dim": self.embedding_dim, "hidden_size": self.lstm_hidden_size, "device": self.device,
                 "padding_char_index": self.padding_char_index, "char_vocabulary_size": self.char_vocabulary_size,
                 "word_idx2chars_idx": self.word_idx2chars_idx, "padding_token_index": self.padding_token_index},
                self.lstm_hidden_size),
            lib.WORD_REPRESENTATION_TYPES_CLI[2]: (
                lib.CBOWSubword,
                {"device": self.device, "num_embeddings": self.word_vocabulary_size, "embedding_dim": self.embedding_dim,
                 "padding_token_index": self.padding_token_index, "word2subword_prefix_idx": self.cbow_word2subword_prefix_idx,
                 "word2subword_postfix_idx": self.cbow_word2subword_postfix_idx},
                self.embedding_dim),
            lib.WORD_REPRESENTATION_TYPES_CLI[3]: (
                lib.WordEmbeddingCharacterLSTM,
                {"device": self.device, "word_embedding_dim": self.embedding_dim, "char_embedding_dim": self.embedding_dim // 2,
                 "hidden_size": self.lstm_hidden_size, "word_vocabulary_size": self.word_vocabulary_size,
                 "padding_token_index": self.padding_token_index, "word_idx2chars_idx": self.word_idx2chars_idx,
                 "padding_char_index": self.padding_char_index, "char_vocabulary_size": self.char_vocabulary_size,
                 "dropout_probability": self.dropout_probability},
                ((self.embedding_dim + self.lstm_hidden_size) // 2))
        }

    def forward(self, sequence: torch.Tensor, dummy_tensor: torch.Tensor = None) -> torch.Tensor:
        """
        @brief: Forward propagation.

        @param sequence: Input sequence, should be padded with @ref self.padding_token_index.
        @param dummy_tensor: Dummy tensor, used for torch.checkpoint.

        @return: Output probabilities of the network.
        """
        # Retrieve input word representation:
        sequence_representation = self.word_representation(sequence)

        # Forward propagate the sequence through the BiLSTM layers:
        input_length = torch.sum(sequence != self.padding_token_index, dim=1)
        bi = self.bi_lstm_layer_1(x=sequence_representation, input_length=input_length)
        bi_tag = self.bi_lstm_layer_2(x=bi, input_length=input_length)

        # Feed the sequence through the linear layer:
        logits = self.fc(bi_tag)

        # Mask the padding:
        mask = (sequence != self.padding_token_index).float().unsqueeze(dim=2)
        logits = logits * mask

        # Create output probabilities:
        probabilities = self.softmax(logits)

        return probabilities

    def predict(self, sequence: torch.Tensor) -> torch.Tensor:
        """
        @brief: Predict the output.

        @param sequence: Input sequence.

        @return: The output prediction.
        """
        probabilities = self.forward(sequence)
        predictions = probabilities.argmax(dim=2)
        return predictions

    def predict_from_probabilities(self, y_prob: torch.Tensor) -> int:
        """
        @brief: Get the prediction.

        @param y_prob: Output probabilities.

        @return: The prediction.
        """
        prediction = y_prob.argmax(dim=2)
        return prediction

    def loss(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """
        @brief: Calculate the loss.

        The loss is calculated without taking into account the padding label.

        @param y_pred: Output probabilities.
        @param y_true: True labels.

        @return: The loss.
        """
        batch_size, max_sequence_length, num_of_labels = y_pred.shape
        input = y_pred.view(batch_size * max_sequence_length, num_of_labels)
        target = y_true.contiguous().view(batch_size * max_sequence_length)
        loss = torch.functional.F.cross_entropy(
            input=input, target=target, reduction='none', ignore_index=self.padding_label_index)
        mask = (target != self.padding_label_index).view(-1).type(torch.FloatTensor).to(self.device)
        mask /= mask.shape[0]
        loss = loss.dot(mask) / mask.sum()

        return loss

    def get_non_trainable_parameters_state_dict(self) -> dict:
        """
        @brief: Get the state dict of the non trainable parameters.

        @return: The state dict of the parameters.
        """
        return {
            "word_vocabulary_size": self.word_vocabulary_size,
            "embedding_dim": self.embedding_dim,
            "lstm_hidden_size": self.lstm_hidden_size,
            "fc_output_size": self.fc_output_size,
            "padding_token_index": self.padding_token_index,
            "padding_label_index": self.padding_label_index,
            "word_representation_type": self.word_representation_type,
            "cbow_word2subword_prefix_idx": self.cbow_word2subword_prefix_idx,
            "cbow_word2subword_postfix_idx": self.cbow_word2subword_postfix_idx,
            "char_vocabulary_size": self.char_vocabulary_size,
            "padding_char_index": self.padding_char_index,
            "word_idx2chars_idx": self.word_idx2chars_idx,
            "dropout_probability": self.dropout_probability,
        }


def main(cli_args) -> None:
    """
    @brief: Main function for the module.

    @return: Exit code.
    """
    # Set the seed for reproducibility:
    lib.seed_everything(lib.SEED)

    # Needed to support CUDA:
    torch.multiprocessing.set_start_method('spawn')

    # Set the device to run on:
    device = lib.get_device()

    # Set num of workers for data parallelism:
    num_workers = cli_args.num_workers

    # Train the model:
    if cli_args.fit is True:
        # Load dataset:
        shuffle = True
        train_data_loader, dev_data_loader = lib.load_train_dataset(
            dataset_type=cli_args.tag_task, device=device, dataset_filename=cli_args.train_file,
            num_workers=num_workers, batch_size=cli_args.batch_size, shuffle=shuffle, collate_fn=lib.collate_batch,
            word_representation=cli_args.word_representation, dev_ratio=cli_args.dev_ratio)

        # Create neural network:
        model = BiLSTMTagger(
            device=device, word_vocabulary_size=train_data_loader.dataset.vocabulary_size,
            embedding_dim=cli_args.embedding_dim,
            lstm_hidden_size=cli_args.lstm_hidden_dim,
            fc_output_size=train_data_loader.dataset.num_of_labels,
            padding_token_index=train_data_loader.dataset.padding_token_index,
            padding_label_index=train_data_loader.dataset.padding_label_index,
            word_representation_type=cli_args.word_representation,
            cbow_word2subword_prefix_idx=train_data_loader.dataset.cbow_word2subword_prefix_idx,
            cbow_word2subword_postfix_idx=train_data_loader.dataset.cbow_word2subword_postfix_idx,
            char_vocabulary_size=train_data_loader.dataset.char_vocabulary_size,
            padding_char_index=train_data_loader.dataset.padding_char_index,
            word_idx2chars_idx=train_data_loader.dataset.word_idx2chars_idx,
            dropout_probability=cli_args.dropout)
        print(model)

        # Fit the model:
        optimizer = torch.optim.Adam(
            params=model.parameters(), lr=cli_args.lr, weight_decay=cli_args.weight_decay)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=cli_args.sched_step, gamma=cli_args.sched_gamma)

        model, fit_statistics = lib.fit(
            model=model, optimizer=optimizer,
            train_data_loader=train_data_loader, dev_data_loader=dev_data_loader,
            epochs=cli_args.epochs, scheduler=scheduler,
            log_error=cli_args.log_error, log_train=cli_args.log_train)

        # Save the model:
        if cli_args.save_model_file != "":
            lib.save_model(cli_args.save_model_file, model, train_data_loader.dataset.get_metadata(), fit_statistics)

        # Plot learning statistics:
        if cli_args.plot_path != "":
            if cli_args.log_train is True:
                lib.save_train_plot_statistics(cli_args.tag_task, fit_statistics, cli_args.plot_path)
            else:
                lib.save_overall_plot_statistics(cli_args.tag_task, fit_statistics, cli_args.plot_path)

    return 0


if __name__ == "__main__":
    try:
        exit_code = 0
        cli_args = lib.parse_cli(description="BiLSTM-Tagger for POS/NEG tagging", bilstm_train=True)

        # Run in production mode:
        if cli_args.debug is False:
            exit_code = main(cli_args)
            sys.exit(exit_code)
    except Exception as error:
        print(error)
        exit_code = 1
        sys.exit(exit_code)

    # Run in debug mode, ignore all other CLI arguments.
    tag_task = lib.NER_TAGGING
    system = "laptop"
    cli_args = argparse.Namespace(
        debug=True,
        tag_task=tag_task,
        log_error=True,
        log_train=True,
        fit=True,
        predict=True,
        train_file=os.path.join(MODULE_ROOT_DIR, "..", "data", tag_task, "train_debug"),
        test_file=os.path.join(MODULE_ROOT_DIR, "..", "data", tag_task, "test"),
        predict_file=os.path.join(MODULE_ROOT_DIR, "..", "output", "debug", system, f"{tag_task}.pred"),
        plot_path=os.path.join(MODULE_ROOT_DIR, "..", "output", "debug", system),
        word_representation=lib.WORD_REPRESENTATION_TYPES_CLI[1],
        save_model_file=os.path.join(MODULE_ROOT_DIR, "..", "output", "debug", system, f"{tag_task}_model.pt"),
        load_model_file=os.path.join(MODULE_ROOT_DIR, "..", "output", "debug", system, f"{tag_task}_model.pt"),
        num_workers=1,
        dev_ratio=0.1,
        epochs=1,
        embedding_dim=64,
        lstm_hidden_dim=64,
        batch_size=50,
        lr=0.02,
        sched_step=2,
        sched_gamma=0.5,
        dropout=0,
        weight_decay=0)

    print(cli_args)
    exit_code = main(cli_args)
    sys.exit(exit_code)
