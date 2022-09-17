import numpy as np

from grad_check import gradient_check
from loglinear import softmax, cross_entropy_loss, encode_one_hot
from mlp1 import tanh_derivative

STUDENT = {'name': 'Simon Raviv'}


def classifier_output(x, params):
    """
    @brief: Computes the output of a N layers MLP classifier.

    @param x: A n-dim vector (numpy array).
    @param params: A list of the form [W1, b1, W2, b2, ..., WN, bN] where each is a numpy arrays.

    @returns: A n-dim vector (numpy array) of the N layers MLP classifier output.
    """
    num_layers = len(params) // 2
    activation_function = np.tanh
    A = [x]
    Z = []

    for i in range(1, num_layers + 1):
        Wi = params[(i - 1) * 2]
        bi = params[i * 2 - 1].reshape(-1,)
        Zi = Wi.T.dot(A[i - 1]) + bi
        Ai = activation_function(Zi)
        A.append(Ai)
        Z.append(Zi)

    An = softmax(Z[-1])
    Y_hat = An

    return Y_hat


def predict(x, params):
    """
    @brief: Returns the prediction (highest scoring class) for a single sample x.

    @param x: A n-dim vector (numpy array).
    @param params: A list of the form [W1, b1, W2, b2, ..., WN, bN] where each is a numpy arrays.

    @returns: The index of the highest scoring class.
    """
    return np.argmax(classifier_output(x, params))


def loss_and_gradients(x, y, params):
    """
    @brief: Computes the loss and gradients of the loss function.

    @param x: A n-dim vector (numpy array).
    @param y: A scalar.
    @param params: A list of the form [W1, b1, W2, b2, ..., WN, bN] where each is a numpy arrays.

    @returns: A tuple of (loss, [gW1, gb1, gW2, gb2, ..., gWN, gbN]).
    """
    ### Forward pass ###
    x = np.array(x)
    num_layers = len(params) // 2
    activation_function = np.tanh
    activation_derivative_function = tanh_derivative
    W = []
    A = [x]
    Z = []

    for i in range(1, num_layers):
        Wip1 = params[(i - 1) * 2]
        bi = params[i * 2 - 1].reshape(-1,)
        Zi = Wip1.T.dot(A[i - 1]) + bi.reshape(-1,)
        Ai = activation_function(Zi)
        W.append(Wip1)
        A.append(Ai)
        Z.append(Zi)

    Wn, bn = params[-2:]
    Zn = Wn.T.dot(A[-1]) + bn.reshape(-1,)
    An = softmax(Zn)
    Y_hat = An

    ### Backward pass ###

    Y = encode_one_hot(y, Wn.shape[1])
    loss = cross_entropy_loss(Y, Y_hat)

    # Compute gradients:

    # Layer n gradients:
    gZn = Y_hat - Y
    gWn = A[-1].reshape(-1, 1).dot(gZn.reshape(1, -1))
    gbn = gZn
    gZim1 = gZn
    gradients = [gbn, gWn]

    # Layers n - 1 gradients:
    for i in range(num_layers - 1, 0, -1):
        # Initialize parameters:
        Wip1 = params[i * 2]

        # Compute gradients:
        gZi = Wip1.dot(gZim1) * activation_derivative_function(Z[i - 1])
        gWi = A[i - 1].reshape(-1, 1).dot(gZi.reshape(1, -1))
        gbi = gZi

        gZim1 = gZi

        # Save gradients:
        gradients.append(gbi)
        gradients.append(gWi)

    gradients.reverse()
    return loss, gradients


def create_classifier(dims):
    """
    @brief: Creates a N layers MLP classifier with specified architecture.

    The initialization of the weights and biases is done using the Glorot Xavier
    initialization method.

    @param dims: A list of integers where each element corresponds to the number of
                 neurons in a layer. The format is [input_dim, L1_dim, ..., LN_dim, output_dim].

    @returns: A list of parameters where each element corresponds to the weights
              and biases for a single layer. The format is [W1, b1, W2, b2, ..., WN, bN].
    """
    parameters = list()
    for layer in range(len(dims[:-1])):
        epsilon = np.sqrt(6.0) / np.sqrt(dims[layer] + dims[layer + 1])
        Wi = np.random.uniform(-epsilon, epsilon, size=(dims[layer], dims[layer + 1]))
        bi = np.zeros((dims[layer + 1],))
        parameters.append(Wi)
        parameters.append(bi)

    return parameters


if __name__ == '__main__':
    test_mlpn = [3, 6, 4, 5]
    W1, b1, W2, b2, W3, b3 = create_classifier(test_mlpn)

    def _loss_and_W1_grad(W1):
        global b1
        loss, grads = loss_and_gradients([1, 2, 3], 0, [W1, b1, W2, b2])
        return loss, grads[0]

    def _loss_and_b1_grad(b1):
        global W1
        loss, grads = loss_and_gradients([1, 2, 3], 0, [W1, b1, W2, b2])
        return loss, grads[1]

    def _loss_and_W2_grad(W2):
        global b2
        loss, grads = loss_and_gradients([1, 2, 3], 0, [W1, b1, W2, b2])
        return loss, grads[2]

    def _loss_and_b2_grad(b2):
        global W2
        loss, grads = loss_and_gradients([1, 2, 3], 0, [W1, b1, W2, b2])
        return loss, grads[3]

    def _loss_and_W3_grad(W3):
        global b3
        loss, grads = loss_and_gradients([1, 2, 3], 0, [W1, b1, W2, b2, W3, b3])
        return loss, grads[4]

    def _loss_and_b3_grad(b3):
        global W3
        loss, grads = loss_and_gradients([1, 2, 3], 0, [W1, b1, W2, b2, W3, b3])
        return loss, grads[5]

    for _ in range(10):
        W1 = np.random.randn(*W1.shape)
        b1 = np.random.randn(*b1.shape)
        W2 = np.random.randn(*W2.shape)
        b2 = np.random.randn(*b2.shape)
        W3 = np.random.randn(*W3.shape)
        b3 = np.random.randn(*b3.shape)

        gradient_check(_loss_and_W1_grad, W1)
        gradient_check(_loss_and_b1_grad, b1)
        gradient_check(_loss_and_W2_grad, W2)
        gradient_check(_loss_and_b2_grad, b2)
        gradient_check(_loss_and_W3_grad, W3)
        gradient_check(_loss_and_b3_grad, b3)

    # Extra test:
    x = np.array([1, 2])
    W1 = np.array([[1, 2, 3], [5, 6, 7]])
    b1 = np.array([-1, -2, -3])
    W2 = np.array([[1, 2, 3], [5, 6, 7], [8, 9, 10]])
    b2 = np.array([-1, -2, -3])
    params = [W1, b1, W2, b2]

    loss, [dW1, db1, dW2, db2] = loss_and_gradients(x, 1, [W1, b1, W2, b2])
