"""
@brief: A module for training and evaluating a multi-layer perceptron.
"""

import train_mlp1 as classifier
import mlp1 as mlp
import utils


def predict_on_test(test_data, model):
    """
    @brief: Predict on test data.

    @param test_data: A list of (x,y) pairs.
    @param model: A list of the form [(Wi, bi)].

    @return: A list of predictions.
    """
    predictions = []
    for _, x in test_data:
        x = classifier.feats_to_vec(x)
        prediction = mlp.predict(x, model)
        label = utils.I2L[prediction]
        predictions.append(label)
    return predictions


def write_predictions(predictions, filename):
    """
    @brief: Write predictions to file.

    @param predictions: A list of predictions.
    @param filename: The file to write to.
    """
    with open(filename, "w") as file:
        for prediction in predictions:
            file.write(str(prediction) + "\n")


def main():
    """
    @brief: Main function.
    """
    _, model = classifier.fit()
    test_data = utils.TEST
    predictions = predict_on_test(test_data, model)
    write_predictions(predictions, "test.pred")


if __name__ == '__main__':
    main()
