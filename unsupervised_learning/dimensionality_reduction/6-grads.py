#!/usr/bin/env python3
""" Calculates the gradients for the t-SNE algorithm. """


import numpy as np

# Import Q_affinities from the appropriate module
Q_affinities = __import__('5-Q_affinities').Q_affinities


def grads(Y, P):
    """
    Calculates the gradients of Y for the t-SNE algorithm.
    """
    n, ndim = Y.shape
    # Calculate Q affinities
    Q, _ = Q_affinities(Y)
    # Initialize the gradient dY
    dY = np.zeros_like(Y)
    # Calculate the gradient
    for i in range(n):
        for j in range(n):
            if P[i, j] > 0:  # Only consider pairs with non-zero affinity
                # Contribution to the gradient
                dY[i] += (P[i, j] - 
                Q[i, j]) * (Y[i] - Y[j]) 
    # Normalize the gradient by the number of points
    dY /= n
    return dY, Q
