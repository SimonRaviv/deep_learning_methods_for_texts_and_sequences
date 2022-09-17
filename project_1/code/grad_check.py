import numpy as np

STUDENT = {'name': 'Simon Raviv'}


def gradient_check(f, x):
    """
    @brief: Compute the gradient of a function at a particular point.

    @param f: A function that takes a single argument and outputs the cost and its gradients.
    @param x: The point (numpy array) to check the gradient at.

    @return: The function value and the gradient (as a list).
    """
    fx, grad = f(x)  # Evaluate function value at original point
    h = 1e-4

    # Iterate over all indexes in x:
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        ix = it.multi_index

        # Copy the parameters x, as they are modified:
        params = np.copy(x)

        # Evaluate the function at the current parameter plus epsilon:
        params[ix] += h
        theta_plus, _ = f(params)

        # Evaluate the function at the current parameter minus epsilon:
        params[ix] -= 2 * h
        theta_minus, _ = f(params)

        # Compute numeric gradient:
        numeric_gradient = (theta_plus - theta_minus) / (2 * h)

        # Compare gradients:
        reldiff = abs(numeric_gradient - grad[ix]) / max(1, abs(numeric_gradient), abs(grad[ix]))
        if reldiff > 1e-5:
            print("Gradient check failed.")
            print("First gradient error found at index %s" % str(ix))
            print("Your gradient: %f \t Numerical gradient: %f" % (grad[ix], numeric_gradient))
            return

        it.iternext()  # Step to next index

    print("Gradient check passed!")


def sanity_check():
    """
    @brief: Some basic sanity checks.
    """
    def quad(x):
        return (np.sum(x ** 2), x * 2)

    print("Running sanity checks...")
    gradient_check(quad, np.array(123.456))       # scalar test
    gradient_check(quad, np.random.randn(3,))     # 1-D test
    gradient_check(quad, np.random.randn(4, 5))   # 2-D test
    print()


if __name__ == '__main__':
    sanity_check()
