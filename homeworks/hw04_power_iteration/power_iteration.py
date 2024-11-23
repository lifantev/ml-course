import numpy as np


def get_dominant_eigenvalue_and_eigenvector(data, num_steps):
    """
    data: np.ndarray - symmetric diagonalizable real-valued matrix
    num_steps: int - number of power method steps

    Returns:
    eigenvalue: float - dominant eigenvalue estimation after `num_steps` steps
    eigenvector: np.ndarray - corresponding eigenvector estimation
    """
    ### YOUR CODE HERE
    assert num_steps > 0
    assert data.shape[0] == data.shape[1]

    rk = np.random.uniform(-1, 1, data.shape[1])
    rk /= np.linalg.norm(rk)
    sk = 0.0
    for _ in range(num_steps):
        rk = data @ rk
        rk /= np.linalg.norm(rk)
        sk = rk @ data @ rk.T

    return float(sk), rk
