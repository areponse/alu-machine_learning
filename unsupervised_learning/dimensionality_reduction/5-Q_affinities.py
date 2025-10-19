#!/usr/bin/env python3
"""
Calculates the Q affinities for the t-SNE algorithm.
"""


import numpy as np


def Q_affinities(Y):
    """
    Calculates the Q affinities for the t-SNE algorithm.
    """
    n, _ = Y.shape
    # Calculate the squared pairwise distances in the low-dimensional space
    sum_Y = np.sum(Y**2, axis=1, keepdims=True)
    D = sum_Y + sum_Y.T - 2 * np.dot(Y, Y.T)
    # Compute the numerator of the Q affinities
    num = np.exp(-D)  # Exponential of the negative distances
    # Set the diagonal to zero to avoid self affinities
    np.fill_diagonal(num, 0)
    # Calculate the Q affinities
    Q = num / np.sum(num, axis=1, keepdims=True)
    # Handle any potential division by zero in Q
    Q[np.isnan(Q)] = 0  # Set NaN Q values to zero
    return Q, num
