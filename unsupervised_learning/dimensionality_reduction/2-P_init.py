#!/usr/bin/env python3
"""
Initializes variables required to calculate the P affinities in t-SNE
"""


import numpy as np


def P_init(X, perplexity):
    """
    Initializes variables for calculating P affinities in t-SNE
    """
    n, d = X.shape
    # Calculate squared pairwise distances
    D = (np.sum(X**2, axis=1).reshape(n, 1) +
         np.sum(X**2, axis=1) -
         2 * np.dot(X, X.T))
    np.fill_diagonal(D, 0)  # Set diagonal to 0
    # Initialize P affinities and beta values
    P = np.zeros((n, n))
    betas = np.ones((n, 1))
    # Calculate Shannon entropy for the specified perplexity
    H = np.log2(perplexity)
    return D, P, betas, H
