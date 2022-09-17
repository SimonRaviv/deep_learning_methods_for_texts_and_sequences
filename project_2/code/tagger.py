"""
PyTorch tagging neural network module for language classification tagging assignment.
The module supports the POS (Part of Speech) and the NER (Named Entity Recognition) tagging tasks.

Author: Simon Raviv.
"""
import argparse
import torch
import os
import sys
import numpy as np

import lib_tagger as lib

MODULE_ROOT_DIR = os.path.dirname(__file__)


class Tagger(torch.nn.Module):
    """
    @brief: Tagger module for language classification tagging assignment.

    The module supports the POS (Part of Speech) and the NER (Named Entity Recognition) tagging tasks.
    """

    def __init__(self, tag_type: str, device: str = "cpu", batch_size: int = 128,
                 vocabulary_size: int = 250, hidden_dim: int = 100, num_of_labels: int = 5,
                 embedding_dim: int = 50, window_size: int = 5, dropout_prob: float = 0,
                 pretrained_embedding_weights: bool = None, freeze_pretrained: bool = False,
                 subword_embedding: str = "", cbow_word2subword_prefix: dict = {}, cbow_word2subword_postfix: dict = {},
                 char_vocabulary_size: int = 0, char_embedding_dim: int = 25, char_padding_index: int = 0,
                 word_idx2chars_idx: dict = {}, cnn_char_num_filters: int = 50,
                 cnn_char_kernel_size: int = 3, cnn_char_padding_size: int = 2, cnn_char_stride: int = 1) -> None:
        """
        @brief: Initialize the tagger.

        @param tag_type: The type of the tag.
        @param device: The device to use: "cpu" or "cuda".
        @param batch_size: The main batch size.
        @param vocabulary_size: The number of features.
        @param hidden_dim: The number of hidden units.
        @param num_of_labels: The number of labels.
        @param embedding_dim: The embedding dimension.
        @param window_size: The window size.
        @param dropout_prob: The dropout probability.
        @param pretrained_embedding_weights: The pretrained embedding layer.
        @param freeze_pretrained: If True, the pretrained embedding layer is frozen when used.
        @param subword_embedding: The subword embedding type, "cnn" or "cbow".
        @param cbow_word2subword_prefix: The word to prefix CBOW sub-word mapping dictionary.
        @param cbow_word2subword_postfix: The word to postfix CBOW sub-word mapping dictionary.
        @param char_vocabulary_size: The number of characters in the vocabulary.
        @param char_embedding_dim: The character embedding dimension.
        @param char_padding_index: The padding character index.
        @param word_idx2chars_idx: The word to word characters indexes mapping dictionary for CNN sub-word embedding.
        @param cnn_char_num_filters: The CNN character number of filters.
        @param cnn_char_kernel_size: The CNN character kernel size.
        @param cnn_char_padding_size: The CNN character padding size.
        @param cnn_char_stride: The CNN character stride.

        @return: None
        """
        super(Tagger, self).__init__()
        self.tag_type = tag_type
        self.device = device
        self.batch_size = batch_size
        self.embedding_dim = embedding_dim
        self.window_size = window_size
        self.num_of_labels = num_of_labels
        self.vocabulary_size = vocabulary_size
        self.hidden_dim = hidden_dim
        self.dropout_prob = dropout_prob
        self.pretrained_embedding_weights = pretrained_embedding_weights
        self.cbow_word2subword_prefix = cbow_word2subword_prefix
        self.cbow_word2subword_postfix = cbow_word2subword_postfix
        self.use_cbow_subword_embedding = subword_embedding == lib.CBOW_SUBWORD_EMBEDDING
        self.use_cnn_subword_embedding = subword_embedding == lib.CNN_SUBWORD_EMBEDDING
        self.char_vocabulary_size = char_vocabulary_size
        self.char_embedding_dim = char_embedding_dim
        self.char_padding_index = char_padding_index
        self.word_idx2chars_idx = word_idx2chars_idx
        self.cnn_char_num_filters = cnn_char_num_filters
        self.cnn_char_kernel_size = cnn_char_kernel_size
        self.cnn_char_padding_size = cnn_char_padding_size
        self.cnn_char_stride = cnn_char_stride
        self.char_embedding = None
        self.char_cnn_layer = None
        self.char_maxpool_layer = None
        self.conv2d_layer_shape = None
        self.maxpool2d_layer_shape = None

        # Model structure:
        self.fc_input_dim = self.window_size * self.embedding_dim
        linear_dropout_multiplier = 1

        if self.use_cnn_subword_embedding is True:
            self.char_embedding = self._get_char_embedding_layer()
            self.char_cnn_layer = torch.nn.Conv2d(
                in_channels=1, out_channels=self.cnn_char_num_filters,
                kernel_size=(self.cnn_char_kernel_size, self.char_embedding_dim),
                stride=(self.cnn_char_stride, 1), padding=(self.cnn_char_padding_size, 0))
            self.conv2d_layer_shape = lib.get_conv2d_layer_shape(
                in_shape=(self.batch_size, 1, lib.MAX_WORD_SIZE, self.char_embedding_dim),
                layer=self.char_cnn_layer)

            self.char_maxpool_layer = torch.nn.MaxPool2d(kernel_size=(self.conv2d_layer_shape[2], 1))
            self.maxpool2d_layer_shape = lib.get_maxpool2d_layer_shape(
                in_shape=(self.batch_size, self.cnn_char_num_filters,
                          self.conv2d_layer_shape[2], self.conv2d_layer_shape[3]),
                layer=self.char_maxpool_layer)
            self.dropout_embedding_layer = torch.nn.Dropout(p=dropout_prob)
            self.fc_input_dim = (self.window_size *
                                 (self.embedding_dim +
                                  self.cnn_char_num_filters *
                                  self.maxpool2d_layer_shape[2] *
                                  self.maxpool2d_layer_shape[3]))
            linear_dropout_multiplier = 0.5

        self.word_embedding = self._get_word_embedding_layer(freeze_pretrained=freeze_pretrained)
        self.linear_layer_1 = torch.nn.Linear(in_features=self.fc_input_dim, out_features=self.hidden_dim)
        self.activation_layer = torch.nn.Tanh()
        self.dropout_linear_layer = torch.nn.Dropout(p=dropout_prob * linear_dropout_multiplier)
        self.linear_layer_2 = torch.nn.Linear(in_features=self.hidden_dim, out_features=num_of_labels)
        self.softmax_layer = torch.nn.Softmax(dim=1)
        self._initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        @brief: Forward pass of the neural network.

        @param x: The input.

        @return: The output prediction.
        """
        batch_size = x.shape[0]
        embedding = self.word_embedding(x)

        if self.use_cbow_subword_embedding is True:
            embedding = self._get_cbow_subword_embedding(x, batch_size, embedding)
        elif self.use_cnn_subword_embedding is True:
            embedding = self._get_cnn_subword_embedding(x, batch_size, embedding)

        embedding = embedding.view(batch_size, -1)
        Z1 = self.linear_layer_1(embedding)
        D1 = self.dropout_linear_layer(Z1)
        A1 = self.activation_layer(D1)
        Z2 = self.linear_layer_2(A1)
        A2 = self.softmax_layer(Z2)

        probabilities = A2
        return probabilities

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        @brief: Predict the output.

        @param x: The input.

        @return: The output prediction.
        """
        probabilities = self.forward(x)
        predictions = torch.argmax(probabilities, dim=1)

        return predictions

    def _initialize_weights(self):
        """
        @brief: Initialize the weights of the model.

        @return: None
        """
        if self.char_cnn_layer is None:
            return

        if self.pretrained_embedding_weights is None:
            bias = np.sqrt(3.0 / self.word_embedding.weight.size(1))
            torch.nn.init.uniform_(self.word_embedding.weight.data, -bias, bias)

        if self.char_embedding is not None:
            bias = np.sqrt(3.0 / self.char_embedding.weight.size(1))
            torch.nn.init.uniform_(self.char_embedding.weight.data, -bias, bias)

        torch.nn.init.xavier_uniform_(self.char_cnn_layer.weight.data)
        if self.char_cnn_layer.bias is not None:
            self.char_cnn_layer.bias.data.zero_()

        torch.nn.init.xavier_uniform_(self.linear_layer_1.weight.data, gain=torch.nn.init.calculate_gain('tanh'))
        if self.linear_layer_1.bias is not None:
            self.linear_layer_1.bias.data.zero_()

        torch.nn.init.xavier_uniform_(self.linear_layer_2.weight.data)
        if self.linear_layer_2.bias is not None:
            self.linear_layer_2.bias.data.zero_()

    def _get_word_embedding_layer(self, freeze_pretrained: bool = False) -> None:
        """
        @brief: Returns word embedding layer.

        This creates and returns the embedding layer.
        The layer can be pretrained or not, based on the input parameters.
        This embedding layer will be used for words and sub-words in the case of CBOW.

        @param freeze_pretrained: If True, the pretrained embedding layer is frozen when used.

        @return: The embedding layer.
        """
        if self.pretrained_embedding_weights is None:
            word_embedding = torch.nn.Embedding(
                num_embeddings=self.vocabulary_size, embedding_dim=self.embedding_dim)
        else:
            word_embedding = torch.nn.Embedding.from_pretrained(
                embeddings=self.pretrained_embedding_weights, freeze=freeze_pretrained)

        return word_embedding

    def _get_char_embedding_layer(self):
        """
        @brief: Returns char embedding layer.

        This creates and returns the embedding layer for characters.
        This embedding layer will be used for sub-words in the case of CNN.

        @## Note: This method supported when @ref self.use_cnn_subword_embedding is True.

        @return: The embedding layer.
        """
        char_embedding = torch.nn.Embedding(
            num_embeddings=self.char_vocabulary_size,
            embedding_dim=self.char_embedding_dim, padding_idx=self.char_padding_index)

        return char_embedding

    def _get_cbow_subword_embedding(self, x, batch_size, embedding):
        """
        @brief: Return CBOW embedding supporting sub-words.

        This method uses the summation over word and the sub-words embeddings,
        when supported, otherwise the original embedding.

        @param x: The input.
        @param batch_size: Batch size.
        @param embedding: X embedding.

        @return: Sub-word supported embedding.
        """
        x_prefix = x.clone().cpu().apply_(
            lambda word_index: self.cbow_word2subword_prefix[word_index]).to(self.device)
        x_postfix = x.clone().cpu().apply_(
            lambda word_index: self.cbow_word2subword_postfix[word_index]).to(self.device)
        embedding = embedding.view(batch_size, -1)
        x_prefix_embedding = self.word_embedding(x_prefix).view(batch_size, -1)
        x_postfix_embedding = self.word_embedding(x_postfix).view(batch_size, -1)
        embedding = embedding + x_prefix_embedding + x_postfix_embedding

        return embedding

    def _get_cnn_subword_embedding(self, x, batch_size, embedding):
        """
        @brief: Return CNN embedding for sub-words.

        This method uses the word characters indexes to get the sub-word embedding.

        @param x: The input.
        @param batch_size: Batch size.
        @param embedding: Word embedding.

        @return: CNN sub-word supported embedding.
        """
        word2chars_idx = self.word_idx2chars_idx[x]
        cnn_char_features = []
        for word_index in range(self.window_size):
            word_chars = word2chars_idx[:, word_index, :].unsqueeze(1)
            chars_embedding = self.char_embedding(word_chars)
            chars_embedding = self.dropout_embedding_layer(chars_embedding)
            convolved_chars = self.char_cnn_layer(chars_embedding)
            pooled_chars = self.char_maxpool_layer(convolved_chars).view(batch_size, -1)
            cnn_char_features.append(pooled_chars.unsqueeze(1))

        cnn_char_features = torch.cat(cnn_char_features, dim=1)
        output_embedding = torch.cat((embedding, cnn_char_features), dim=2)

        return output_embedding

    def get_convolved_chars(self, words: torch.Tensor) -> torch.Tensor:
        """
        @brief: Return convolved chars.

        This method returns for each word the convolved chars.

        @param x: The input.

        @return: Convolved chars.
        """
        batch_size = words.shape[0]
        words2chars = self.word_idx2chars_idx[words]
        chars_embedding = self.char_embedding(words2chars).unsqueeze(1)
        convolved_chars = self.char_cnn_layer(chars_embedding).view(batch_size, self.conv2d_layer_shape[1], -1)

        return convolved_chars


def main(cli_args: argparse.Namespace) -> int:
    """
    @brief: Main function.

    @param cli_args: The command line arguments.

    @return: Exit code.
    """
    # Set the seed for reproducibility:
    lib.seed_everything(lib.SEED)

    # Set the device to run on:
    device = lib.get_device()

    # Load dataset:
    shuffle = True
    use_pretrained_embedding = cli_args.vocabulary_file != cli_args.embedding_file != ""
    train_data_loader, dev_data_loader = lib.load_train_dataset(
        device=device, tag_type=cli_args.tag_type, data_directory=cli_args.dataset_path,
        batch_size=cli_args.batch_size, shuffle=shuffle,
        use_pretrained_embedding=use_pretrained_embedding,
        vocabulary_file=cli_args.vocabulary_file, embedding_file=cli_args.embedding_file,
        subword_embedding=cli_args.subword_embedding, window_size=cli_args.window_size)

    # Create neural network:
    embedding_dim = 50
    freeze_pretrained = False
    cnn_char_padding_size = 2

    model = Tagger(
        tag_type=cli_args.tag_type,
        device=device,
        batch_size=cli_args.batch_size,
        embedding_dim=embedding_dim,
        window_size=cli_args.window_size,
        vocabulary_size=train_data_loader.dataset.vocabulary_size,
        hidden_dim=cli_args.hidden_dim,
        num_of_labels=train_data_loader.dataset.num_of_labels,
        dropout_prob=cli_args.dropout,
        pretrained_embedding_weights=train_data_loader.dataset.pretrained_embedding_weights,
        freeze_pretrained=freeze_pretrained,
        subword_embedding=cli_args.subword_embedding,
        cbow_word2subword_prefix=train_data_loader.dataset.cbow_word2subword_prefix_idx,
        cbow_word2subword_postfix=train_data_loader.dataset.cbow_word2subword_postfix_idx,
        char_vocabulary_size=train_data_loader.dataset.char_vocabulary_size,
        char_padding_index=train_data_loader.dataset.char_padding_index,
        char_embedding_dim=cli_args.char_embedding_dim,
        cnn_char_num_filters=cli_args.cnn_char_num_filters,
        cnn_char_kernel_size=cli_args.cnn_char_kernel_size,
        cnn_char_stride=cli_args.cnn_char_stride,
        cnn_char_padding_size=cnn_char_padding_size,
        word_idx2chars_idx=train_data_loader.dataset.word_idx2chars_idx).to(device)
    print(model)

    # Initialize hyperparameters:
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=cli_args.lr, weight_decay=cli_args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=cli_args.sched_step, gamma=cli_args.sched_gamma)
    filter_index = train_data_loader.dataset.label2idx[lib.NER_COMMON_TAG] if cli_args.tag_type == lib.NER_TAGGING else None
    gradient_clip = 0.5 if cli_args.subword_embedding == lib.CNN_SUBWORD_EMBEDDING else 0

    # Train the model:
    if cli_args.fit is True:
        model, fit_statistics = lib.fit(train_data_loader, dev_data_loader, device, model,
                                        loss_fn, optimizer, gradient_clip, scheduler, cli_args.epochs,
                                        cli_args.log_train, cli_args.log_error, filter_index)

        # Plot learning statistics:
        if cli_args.plot_path != "":
            lib.save_plot_statistics(cli_args.tag_type, fit_statistics, cli_args.plot_path)

        # Run CNN filters explanation method:
        if cli_args.visualize_cnn_filters is True:
            lib.visualize_cnn_filters(dataset=train_data_loader.dataset, model=model,
                                      word_frequency=10, plot_path=cli_args.plot_path)

    # Evaluate the model:
    if cli_args.evaluate is True and cli_args.test_path != "" and cli_args.predict_path != "":
        if cli_args.debug is True:
            dev_accuracy, filtered_dev_accuracy = lib.evaluate(dev_data_loader, device, model, filter_index)
            log = f"Dev accuracy: {dev_accuracy:.4f}"
            if cli_args.tag_type == lib.NER_TAGGING:
                log += f"\nFiltered dev accuracy: {filtered_dev_accuracy:.4f}"
            print(log)

        test_data_loader = lib.load_test_dataset(
            device=device, tag_type=cli_args.tag_type, data_directory=cli_args.test_path,
            batch_size=cli_args.batch_size * 10, metadata=train_data_loader.dataset.get_metadata(),
            use_pretrained_embedding=use_pretrained_embedding,
            subword_embedding=cli_args.subword_embedding, window_size=cli_args.window_size)
        lib.evaluate(test_data_loader, device, model, filter_index, cli_args.predict_path)

    return 0


if __name__ == '__main__':
    try:
        exit_code = 1
        cli_args = lib.parser_cli()

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
        fit=True,
        evaluate=True,
        tag_type=lib.NER_TAGGING,
        dataset_path=os.path.join(MODULE_ROOT_DIR, "..", "data", "ner"),
        test_path=os.path.join(MODULE_ROOT_DIR, "..", "data", "ner"),
        predict_path=os.path.join(MODULE_ROOT_DIR, "..", "output", "debug", "test1_debug.ner"),
        plot_path=os.path.join(MODULE_ROOT_DIR, "..", "output", "debug"),
        vocabulary_file=os.path.join(MODULE_ROOT_DIR, "..", "data", "vocab.txt"),
        embedding_file=os.path.join(MODULE_ROOT_DIR, "..", "data", "wordVectors.txt"),
        subword_embedding=lib.CNN_SUBWORD_EMBEDDING,
        epochs=1,
        hidden_dim=64,
        batch_size=64,
        lr=0.001,
        dropout=0,
        weight_decay=0,
        sched_step=10,
        sched_gamma=1,
        char_embedding_dim=30,
        cnn_char_num_filters=5,
        cnn_char_kernel_size=3,
        cnn_char_stride=1,
        window_size=5,
        visualize_cnn_filters=True,
        log_error=True,
        log_train=False)

    print(cli_args)
    exit_code = main(cli_args)
    sys.exit(exit_code)
