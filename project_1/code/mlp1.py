import numpy as np

from loglinear import softmax, cross_entropy_loss, encode_one_hot

STUDENT = {'name': 'Simon Raviv'}


def tanh_derivative(z):
    """
    @brief: Computes the derivative of tanh(z).

    @param z: The input to the tanh function.

    @return: The derivative of tanh(z).
    """
    output = 1 - np.power(np.tanh(z), 2)
    return output


def classifier_output(x, params):
    """
    @brief: Computes the output of a log-linear classifier.

    @param x: A n-dim vector (numpy array).
    @param params: A list of (numpy arrays) of the form [W, b, U, b_tag].

    @returns: A n-dim vector (numpy array) of log-linear classifier output.
    """
    W, b, U, b_tag = params

    # Layer 1:
    Z1 = W.T.dot(x) + b
    A1 = np.tanh(Z1)

    # Layer 2:
    Z2 = U.T.dot(A1) + b_tag
    A2 = softmax(Z2)

    Y_hat = A2
    return Y_hat


def predict(x, params):
    """
    @brief: Returns the prediction (highest scoring class) for a single sample x.

    @param x: A n-dim vector (numpy array).
    @param params: A list of (numpy arrays) of the form [W, b, U, b_tag].

    @returns: The index of the highest scoring class.
    """
    return np.argmax(classifier_output(x, params))


def loss_and_gradients(x, y, params):
    """
    @brief: Computes the loss and gradients of the loss function.

    Compute the loss and the gradients at point x with given parameters.
    y is a scalar indicating the correct label.

    @param x: A n-dim vector (numpy array) of input data.
    @param y: A scalar value of the correct label.
    @param params: A list of (numpy arrays) of the form [W, b, U, b_tag].

    @returns: A tuple of (loss, [gW, gb, gU, gb_tag]).
    """
    x = np.array(x)
    W, b, U, b_tag = params

    ### Forward pass ###

    # Layer 1:
    Z1 = W.T.dot(x) + b
    A1 = np.tanh(Z1)

    # Layer 2:
    Z2 = U.T.dot(A1) + b_tag
    A2 = softmax(Z2)
    Y_hat = A2

    ### Backward pass ###

    Y = encode_one_hot(y, U.shape[1])
    loss = cross_entropy_loss(Y, Y_hat)

    # Compute gradients:

    # Layer 2 gradients:
    gZ2 = Y_hat - Y
    gU = A1.reshape(-1, 1).dot(gZ2.reshape(1, -1))
    gb_tag = gZ2

    # Layer 1 gradients:
    gZ1 = U.dot(gZ2) * tanh_derivative(Z1)
    gW = x.reshape(-1, 1).dot(gZ1.reshape(1, -1))
    gb = gZ1

    return loss, [gW, gb, gU, gb_tag]


def create_classifier(in_dim, hid_dim, out_dim):
    """
    @brief: Creates a 1 hidden layer classifier.

    @param in_dim: The size of the input layer.
    @param hid_dim: The size of the hidden layer.
    @param out_dim: The size of the output layer.

    @returns: A list of (numpy arrays) of the form [W, b, U, b_tag].
    """
    epsilon = np.sqrt(6.0) / np.sqrt(hid_dim + in_dim)
    W = np.random.uniform(-epsilon, epsilon, size=(in_dim, hid_dim))
    epsilon = np.sqrt(6.0) / np.sqrt(hid_dim + 1)
    b = np.zeros((hid_dim,))
    epsilon = np.sqrt(6.0) / np.sqrt(out_dim + hid_dim)
    U = np.random.uniform(-epsilon, epsilon, size=(hid_dim, out_dim))
    epsilon = np.sqrt(6.0) / np.sqrt(out_dim + 1)
    b_tag = np.zeros((out_dim,))

    return [W, b, U, b_tag]


if __name__ == '__main__':
    test1 = softmax(np.array([1, 2]))
    print(test1)
    assert np.amax(np.fabs(test1 - np.array([0.26894142,  0.73105858]))) <= 1e-6

    test2 = softmax(np.array([1001, 1002]))
    print(test2)
    assert np.amax(np.fabs(test2 - np.array([0.26894142, 0.73105858]))) <= 1e-6

    test3 = softmax(np.array([-1001, -1002]))
    print(test3)
    assert np.amax(np.fabs(test3 - np.array([0.73105858, 0.26894142]))) <= 1e-6

    from grad_check import gradient_check

    x = np.array([1, 2, 3])
    W, b, U, b_tag = create_classifier(3, 4, 4)

    def _loss_and_W_grad(W):
        global b
        loss, grads = loss_and_gradients(x, 0, [W, b, U, b_tag])
        return loss, grads[0]

    def _loss_and_b_grad(b):
        global W
        loss, grads = loss_and_gradients(x, 0, [W, b, U, b_tag])
        return loss, grads[1]

    def _loss_and_U_grad(U):
        global b_tag
        loss, grads = loss_and_gradients(x, 0, [W, b, U, b_tag])
        return loss, grads[2]

    def _loss_and_b_tag_grad(b_tag):
        global U
        loss, grads = loss_and_gradients(x, 0, [W, b, U, b_tag])
        return loss, grads[3]

    for _ in range(10):
        W = np.random.randn(*W.shape)
        b = np.random.randn(*b.shape)
        U = np.random.randn(*U.shape)
        b_tag = np.random.randn(*b_tag.shape)
        gradient_check(_loss_and_b_grad, b)
        gradient_check(_loss_and_W_grad, W)
        gradient_check(_loss_and_b_tag_grad, b_tag)
        gradient_check(_loss_and_U_grad, U)

    # Extra test:
    x = np.array([1, 2])
    W1 = np.array([[1, 2, 3], [5, 6, 7]])
    b1 = np.array([-1, -2, -3])
    W2 = np.array([[1, 2, 3], [5, 6, 7], [8, 9, 10]])
    b2 = np.array([-1, -2, -3])
    params = [W1, b1, W2, b2]

    loss, grads = loss_and_gradients(x, 1, params)
