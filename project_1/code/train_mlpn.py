import random
import numpy as np

import utils
import mlpn

from train_loglin import feats_to_vec

# Global variables:
STUDENT = {'name': 'Simon Raviv'}
SEED = 2022


def accuracy_on_dataset(dataset, params,
                        features_to_vector_cb=feats_to_vec,
                        label_to_number_cb=(lambda label: utils.L2I[label])):
    """
    @brief: Calculates prediction accuracy on the dataset.

    @param dataset: The dataset.
    @param params: Network parameters.
    @param features_to_vector_cb: A callback function to convert features to Numpy vector.
    @param label_to_number_cb: A callback function to convert label to number.

    @return: Accuracy of the prediction on the given dataset.
    """
    good = bad = 0.0
    for label, features in dataset:
        # Create Numpy features X vector:
        X = features_to_vector_cb(features)

        # Classify the input:
        Y_hat = mlpn.classifier_output(X, params)
        y_prediction = np.argmax(Y_hat)

        # Count the prediction as success/failure:
        y_label = label_to_number_cb(label)
        if y_prediction == y_label:
            good += 1
        else:
            bad += 1

    # Compute total accuracy:
    accuracy = good / (good + bad)
    return accuracy


def train_classifier(train_data, dev_data, num_iterations, learning_rate, params,
                     features_to_vector_cb=feats_to_vec,
                     label_to_number_cb=(lambda label: utils.L2I[label])):
    """
    @brief: Create and train a classifier, and return the parameters.

    @param train_data: A list of (label, feature) pairs.
    @param dev_data: A list of (label, feature) pairs.
    @param num_iterations: The maximal number of training iterations.
    @param learning_rate: The learning rate to use.
    @param params: List of parameters (initial values).
    @param features_to_vector_cb: A callback function to convert features to Numpy vector.
    @param label_to_number_cb: A callback function to convert label to number.

    @return: The trained network parameters.
    """
    lr_decay_factor = 1.0
    lr_decay_interval = 10

    for I in range(num_iterations):
        cum_loss = 0.0  # Total loss in this iteration.
        random.shuffle(train_data)
        for label, features in train_data:
            x = features_to_vector_cb(features)  # Convert features to a vector.
            y = label_to_number_cb(label)  # Convert the label to number if needed.
            loss, grads = mlpn.loss_and_gradients(x, y, params)
            cum_loss += loss

            # Update the parameters according to the gradients:
            for i in range(len(params)):
                params[i] = params[i] - learning_rate * grads[i]

        train_loss = cum_loss / len(train_data)
        train_accuracy = accuracy_on_dataset(train_data, params, features_to_vector_cb, label_to_number_cb)
        dev_accuracy = accuracy_on_dataset(dev_data, params, features_to_vector_cb, label_to_number_cb)
        print(f"I: {I+1} train_loss: {train_loss}, train_accuracy: {train_accuracy}, dev_accuracy: {dev_accuracy}")

        if I % lr_decay_interval == 0:
            learning_rate *= lr_decay_factor

    return params


def fit():
    """
    @brief: Trains the network on the given dataset.

    The function demonstrates the fiting on the language dataset and
    on the xor dataset. Set the @ref xor_data to True to run the
    fitting on the xor dataset.

    @return: The original and trained parameters tuple.
    """
    # Set random state seed:
    random.seed(SEED)
    np.random.seed(SEED)

    xor_data = False
    if xor_data is False:
        # Set data variables:
        train_data = utils.TRAIN
        dev_data = utils.DEV

        # Initialize parameters dimensions:
        in_dim = len(utils.vocab)
        out_dim = len(utils.L2I)

        # Initialize network architecture:
        dimensions = [in_dim, 15, 15, out_dim]

        # Callback functions:
        features_to_vector_cb = feats_to_vec

        def label_to_number_cb(label):
            return utils.L2I[label]

        # Set optimizer parameters:
        num_iterations = 25
        learning_rate = 0.0009
    else:
        # Set data variables:
        train_data = dev_data = [(1, [0, 0]),
                                 (0, [0, 1]),
                                 (0, [1, 0]),
                                 (1, [1, 1])]

        # Initialize parameters dimensions:
        in_dim = 2
        out_dim = 2

        # Initialize network architecture:
        dimensions = [in_dim, 5, out_dim]

        # Callback functions:

        def features_to_vector_cb(features):
            return np.array(features)

        def label_to_number_cb(label):
            return label

        # Set optimizer parameters:
        num_iterations = 25
        learning_rate = 0.1

    # Create classifier and train the network:
    params = mlpn.create_classifier(dimensions)
    trained_params = train_classifier(train_data, dev_data, num_iterations, learning_rate,
                                      params, features_to_vector_cb, label_to_number_cb)

    return params, trained_params


if __name__ == '__main__':
    params, trained_params = fit()
