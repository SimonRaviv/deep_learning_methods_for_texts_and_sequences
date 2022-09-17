"""
A module of an neural network that uses the Long-Short-Term-Memory (LSTM) kind of RNN, followed by
an MLP with one hidden layer, in order to perform binary classification of input sequences.
The network learns to distinguish good sequences from bad ones.

Positive sentences in the form:
[1-9]+a+[1-9]+b+[1-9]+c+[1-9]+d+[1-9]+

Negative sentences in the form:
[1-9]+a+[1-9]+c+[1-9]+b+[1-9]+d+[1-9]+

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


class LSTMAcceptor(torch.nn.Module):
    """
    @brief: LSTM acceptor sequence classifier.

    A class of an neural network that uses the Long-Short-Term-Memory (LSTM), followed by
    an MLP with one hidden layer, in order to perform binary classification of input sequences.
    The network learns to distinguish good sequences from bad ones.
    """

    def __init__(self, device: str = "", vocabulary_size: int = -1, lstm_input_size: int = -1, lstm_hidden_size: int = -1,
                 fc_hidden_size: int = -1, fc_output_size: int = -1, padding_index: int = -1, dropout_probability: int = 0) -> None:
        """
        @brief: Initialize the class.

        @param device: Device to use.
        @param vocabulary_size: Size of the vocabulary.
        @param lstm_input_size: Size of the input sequence.
        @param lstm_hidden_size: Size of the hidden layer of the LSTM.
        @param fc_hidden_size: Size of the hidden layer of the MLP.
        @param fc_output_size: Size of the output layer of the MLP.
        @param padding_index: Index of the padding token.
        @param dropout_probability: Probability of dropout to the MLP.

        @return: None.
        """
        super(LSTMAcceptor, self).__init__()
        self.device = device
        self.vocabulary_size = vocabulary_size
        self.lstm_input_size = lstm_input_size
        self.lstm_hidden_size = lstm_hidden_size
        self.fc_hidden_size = fc_hidden_size
        self.fc_output_size = fc_output_size
        self.padding_index = padding_index
        self.dropout_probability = dropout_probability
        self.embedding = torch.nn.Embedding(num_embeddings=vocabulary_size,
                                            embedding_dim=lstm_input_size, padding_idx=padding_index)
        self.lstm = lib.LSTM(input_size=lstm_input_size, hidden_size=lstm_hidden_size,
                             device=device, padding_index=padding_index)
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(in_features=lstm_hidden_size, out_features=fc_hidden_size),
            torch.nn.Tanh(),
            torch.nn.Dropout(p=dropout_probability),
            torch.nn.Linear(in_features=fc_hidden_size, out_features=fc_output_size),
            torch.nn.Softmax(dim=1)
        )
        self.to(device)

    def forward(self, sequence: torch.Tensor, dummy_tensor: torch.Tensor = None) -> torch.Tensor:
        """
        @brief: Forward propagation.

        @param sequence: Input sequence, should be padded with @ref self.padding_index.
        @param dummy_tensor: Dummy tensor, used for torch.checkpoint.

        @return: Output probabilities of the network.
        """
        # Calculate input length:
        input_length = torch.sum(sequence != self.padding_index, dim=1)

        # Embed the sequence:
        embedded_sequence = self.embedding(sequence)

        # Pass the sequence through the LSTM:
        lstm_output = self.lstm(sequence=embedded_sequence, input_length=input_length)

        # Classify the sequence:
        output = self.fc(lstm_output)

        return output

    def predict(self, sequence: torch.Tensor) -> torch.Tensor:
        """
        @brief: Predict the output.

        @param sequence: Input sequence.

        @return: The output prediction.
        """
        probabilities = self.forward(sequence)
        predictions = probabilities.argmax(dim=1)
        return predictions

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
        loss = torch.functional.F.cross_entropy(input=y_pred, target=y_true, reduction='sum')
        return loss

    def get_non_trainable_parameters_state_dict(self) -> dict:
        """
        @brief: Get the state dict of the non trainable parameters.

        @return: The state dict of the parameters.
        """
        return {
            "vocabulary_size": self.vocabulary_size,
            "lstm_input_size": self.lstm_input_size,
            "lstm_hidden_size": self.lstm_hidden_size,
            "fc_hidden_size": self.fc_hidden_size,
            "fc_output_size": self.fc_output_size,
            "padding_index": self.padding_index,
        }


def main(cli_args) -> None:
    """
    @brief: Main function for the module.

    @return: Exit code.
    """
    # Set the seed for reproducibility:
    lib.seed_everything(lib.SEED)

    # Set the device to run on:
    device = lib.get_device()

    # Set num of workers for data parallelism:
    num_workers = cli_args.num_workers

    # Train the model:
    if cli_args.fit is True:
        # Load dataset:
        shuffle = True
        train_data_loader, dev_data_loader = lib.load_train_dataset(
            dataset_type=lib.POS_NEG_TAGGING, device=device, dataset_filename=cli_args.train_file,
            num_workers=num_workers, batch_size=cli_args.batch_size, shuffle=shuffle)
        dataset_metadata = train_data_loader.dataset.get_metadata()

        # Create neural network:
        model = LSTMAcceptor(
            device=device,
            vocabulary_size=train_data_loader.dataset.vocabulary_size,
            lstm_input_size=cli_args.embedding_dim,
            lstm_hidden_size=cli_args.lstm_hidden_dim,
            fc_hidden_size=cli_args.mlp_hidden_dim,
            fc_output_size=train_data_loader.dataset.num_of_labels,
            padding_index=train_data_loader.dataset.padding_token_index,
            dropout_probability=cli_args.dropout)
        print(model)

        # fit the model:
        optimizer = torch.optim.Adam(params=model.parameters(), lr=cli_args.lr, weight_decay=cli_args.weight_decay)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=cli_args.sched_step, gamma=cli_args.sched_gamma)

        model, fit_statistics = lib.fit(
            model=model, optimizer=optimizer,
            train_data_loader=train_data_loader, dev_data_loader=dev_data_loader,
            epochs=cli_args.epochs, scheduler=scheduler,
            log_error=cli_args.log_error, log_train=cli_args.log_train)

        # Save the model:
        if cli_args.save_model_file != "":
            lib.save_model(cli_args.save_model_file, model, dataset_metadata, fit_statistics)

        # Plot learning statistics:
        if cli_args.plot_path != "":
            if cli_args.log_train is True:
                lib.save_train_plot_statistics("pos_neg", fit_statistics, cli_args.plot_path)
            else:
                lib.save_overall_plot_statistics("pos_neg", fit_statistics, cli_args.plot_path)

    elif cli_args.load_model_file != "":
        model, dataset_metadata, fit_statistics = lib.load_model(device, cli_args.load_model_file, LSTMAcceptor)
        print(model)

    # Evaluate the model:
    if cli_args.predict is True and cli_args.test_file != "" and cli_args.predict_file != "":
        if cli_args.debug is True and cli_args.load_model_file == "":
            dev_accuracy = lib.predict(lib.POS_NEG_TAGGING, dev_data_loader, model)
            log = f"Dev accuracy: {dev_accuracy:.4f}"
            print(log)

        test_data_loader = lib.load_test_dataset(
            dataset_type=lib.POS_NEG_TAGGING, device=device, test_file=cli_args.test_file, num_workers=0,
            batch_size=cli_args.batch_size * 10, metadata=dataset_metadata)
        lib.predict(lib.POS_NEG_TAGGING, test_data_loader, model, cli_args.predict_file)

    return 0


if __name__ == "__main__":
    try:
        exit_code = 0
        cli_args = lib.parse_cli(description="Positive and negative tagging sequence classifier")

        # Run in production mode:
        if cli_args.debug is False:
            exit_code = main(cli_args)
            sys.exit(exit_code)
    except Exception as error:
        print(error)
        exit_code = 1
        sys.exit(exit_code)

    # Run in debug mode, ignore all other CLI arguments.
    cli_args = argparse.Namespace(
        debug=True,
        log_error=True,
        log_train=False,
        fit=True,
        predict=True,
        train_file=os.path.join(MODULE_ROOT_DIR, "..", "data", "pos_neg", "train"),
        test_file=os.path.join(MODULE_ROOT_DIR, "..", "data", "pos_neg", "test"),
        predict_file=os.path.join(MODULE_ROOT_DIR, "..", "output", "debug", "laptop", "pos_neg.pred"),
        plot_path=os.path.join(MODULE_ROOT_DIR, "..", "output", "debug", "laptop"),
        save_model_file=os.path.join(MODULE_ROOT_DIR, "..", "output", "debug", "laptop", "pos_neg_model.pt"),
        load_model_file=os.path.join(MODULE_ROOT_DIR, "..", "output", "debug", "laptop", "pos_neg_model.pt"),
        num_workers=2,
        dev_ratio=0.1,
        epochs=15,
        embedding_dim=20,
        mlp_hidden_dim=16,
        lstm_hidden_dim=32,
        batch_size=32,
        lr=0.0012,
        sched_step=10,
        sched_gamma=1,
        dropout=0.5,
        weight_decay=0)

    print(cli_args)
    exit_code = main(cli_args)
    sys.exit(exit_code)
