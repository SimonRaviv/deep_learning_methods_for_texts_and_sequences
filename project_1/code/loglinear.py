import numpy as np

STUDENT = {'name': 'Simon Raviv'}


def cross_entropy_loss(Y, Y_hat):
    """
    @brief: Computes the cross entropy loss.

    @param Y: A n-dim vector (numpy array) of target values.
    @param Y_hat: A n-dim vector (numpy array) of predicted values.

    @returns: A scalar value of the cross entropy loss.
    """
    ce_loss = -np.sum(Y * np.log(Y_hat + np.finfo(float).eps))
    return ce_loss


def encode_one_hot(y, output_dimension):
    """
    @brief: Encodes the given scalar value as a one-hot vector.

    @param y: Scalar value to be encoded.
    @param output_dimension: Number of classes.

    @returns: A one-hot vector of length output_dimension.
    """
    Y = np.zeros((output_dimension,))
    Y[y] = 1
    return Y


def softmax(z):
    """
    @brief: Computes the softmax vector.

    This is a numerically stable implementation of the softmax.

    @param x: A n-dim vector (numpy array).

    @returns: A n-dim vector (numpy array) of softmax values.
    """
    max_element = np.max(z)
    z = np.exp(z - max_element) / np.sum(np.exp(z - max_element))
    return z


def classifier_output(x, params):
    """
    @brief: Computes the output of a log-linear classifier.

    @param x: A n-dim vector (numpy array).
    @param params: A list of (numpy arrays) of the form [W, b].

    @returns: A n-dim vector (numpy array) of log-linear classifier output.
    """
    W, b = params

    Z = W.T.dot(x) + b
    A = softmax(Z)
    Y_hat = A

    return Y_hat


def predict(x, params):
    """
    @brief: Returns the prediction (highest scoring class) for a single sample x.

    Returns the prediction (highest scoring class id) of a
    a log-linear classifier with given parameters on input x.

    @param x: A n-dim vector (numpy array) of input data.
    @param params: A list of (numpy arrays) of the form [W, b].

    @returns: The prediction (highest scoring class id).
    """
    return np.argmax(classifier_output(x, params))


def loss_and_gradients(x, y, params):
    """
    @brief: Computes the loss and gradients of the loss function.

    Compute the loss and the gradients at point x with given parameters.
    y is a scalar indicating the correct label.

    @param x: A n-dim vector (numpy array) of input data.
    @param y: A scalar value of the correct label.
    @param params: A list of (numpy arrays) of the form [W, b].

    @returns: A tuple of (loss, [gW, gb]).
    """
    W, b = params
    x = np.array(x)

    ### Forward pass ###
    Z = W.T.dot(x) + b
    A = softmax(Z)
    Y_hat = A

    ### Backward pass ###
    Y = encode_one_hot(y, W.shape[1])
    loss = cross_entropy_loss(Y, Y_hat)

    # Compute gradients:
    gZ = Y_hat - Y
    gW = gZ.reshape(-1, 1).dot(x.reshape(-1, 1).T).T
    gb = gZ

    return loss, [gW, gb]


def create_classifier(in_dim, out_dim):
    """
    @brief: Creates a log-linear classifier.

    Creates a log-linear classifier with input dimension in_dim
    and output dimension out_dim, with W and b zero initialized.

    @param in_dim: Input dimension.
    @param out_dim: Output dimension.

    @returns: A list of (numpy arrays) of the form [W, b].
    """
    W = np.zeros((in_dim, out_dim))
    b = np.zeros(out_dim)
    return [W, b]


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

    W, b = create_classifier(3, 4)

    def _loss_and_W_grad(W):
        global b
        loss, grads = loss_and_gradients([1, 2, 3], 0, [W, b])
        return loss, grads[0]

    def _loss_and_b_grad(b):
        global W
        loss, grads = loss_and_gradients([1, 2, 3], 0, [W, b])
        return loss, grads[1]

    for _ in range(10):
        W = np.random.randn(W.shape[0], W.shape[1])
        b = np.random.randn(b.shape[0])
        gradient_check(_loss_and_b_grad, b)
        gradient_check(_loss_and_W_grad, W)
