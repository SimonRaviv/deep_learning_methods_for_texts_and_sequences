import random
import numpy as np

import utils
import loglinear as ll

# Global variables:
STUDENT = {'name': 'Simon Raviv'}
SEED = 2022


def feats_to_vec(features):
    """
    @brief: Converts bigram features to vector.

    @param features: Bigram features.

    @return: Vector of bigram features.
    """
    counts = [0] * len(utils.F2I)
    for feature in features:
        if feature not in utils.F2I:
            continue
        feature_index = utils.F2I[feature]
        counts[feature_index] += 1

    np_features = np.array(counts)
    return np_features


def accuracy_on_dataset(dataset, params):
    """
    @brief: Calculates prediction accuracy on the dataset.

    @param dataset: The dataset.
    @param params: Network parameters.

    @return: Accuracy of the prediction on the given dataset.
    """
    good = bad = 0.0
    for label, features in dataset:
        # Create Numpy features X vector:
        X = feats_to_vec(features)

        # Classify the input:
        Y_hat = ll.classifier_output(X, params)
        y_prediction = np.argmax(Y_hat)

        # Count the prediction as success/failure:
        y_label = utils.L2I[label]
        if y_prediction == y_label:
            good += 1
        else:
            bad += 1

    # Compute total accuracy:
    accuracy = good / (good + bad)
    return accuracy


def train_classifier(train_data, dev_data, num_iterations, learning_rate, params):
    """
    @brief: Trains the classifier.

    @param train_data: A list of (label, feature) pairs.
    @param dev_data: A list of (label, feature) pairs.
    @param num_iterations: The maximal number of training iterations.
    @param learning_rate: The learning rate to use.
    @param params: List of parameters (initial values).

    @return: The trained network parameters.
    """
    for I in range(num_iterations):
        cum_loss = 0.0  # Total loss in this iteration.
        random.shuffle(train_data)
        for label, features in train_data:
            x = feats_to_vec(features)  # Convert features to a vector.
            y = utils.L2I[label]  # Convert the label to number.
            loss, grads = ll.loss_and_gradients(x, y, params)
            cum_loss += loss

            # Initialize parameters:
            W, b = params
            gW, gb = grads

            # Do update rule:
            W = W - learning_rate * gW
            b = b - learning_rate * gb

            # Update parameters:
            params = W, b

        # Compute the accuracy on the datasets:
        train_loss = cum_loss / len(train_data)
        train_accuracy = accuracy_on_dataset(train_data, params)
        dev_accuracy = accuracy_on_dataset(dev_data, params)
        print(I, train_loss, train_accuracy, dev_accuracy)

    return params


def fit():
    """
    @brief: Trains the network on the given dataset.

    @return: The original and trained parameters tuple.
    """
    # Set random state seed:
    random.seed(SEED)
    np.random.seed(SEED)

    # Set data variables:
    train_data = utils.TRAIN
    dev_data = utils.DEV

    # Set optimizer parameters:
    num_iterations = 10
    learning_rate = 0.01

    # Initialize input/output parameters:
    in_dim = len(utils.vocab)
    out_dim = len(utils.L2I)

    # Create classifier and train the network:
    params = ll.create_classifier(in_dim, out_dim)
    trained_params = train_classifier(train_data, dev_data, num_iterations, learning_rate, params)

    return params, trained_params


if __name__ == '__main__':
    params, trained_params = fit()
