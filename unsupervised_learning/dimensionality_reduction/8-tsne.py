#!/usr/bin/env python3
""" Performs t-SNE on a dataset. """


import numpy as np

# Import the required functions
pca = __import__('1-pca').pca
P_affinities = __import__('4-P_affinities').P_affinities
grads = __import__('6-grads').grads
cost = __import__('7-cost').cost


def tsne(X, ndims=2, idims=50, perplexity=30.0, iterations=1000, lr=500):
    """
    Performs t-SNE transformation on the dataset X.
    """
    n, d = X.shape
    # Step 1: Perform PCA to reduce dimensionality to idims
    X_pca = pca(X, idims)
    # Step 2: Initialize Y randomly
    Y = np.random.rand(n, ndims)
    # Step 3: Early exaggeration parameters
    exaggeration = 4.0
    a_t = 0.5  # Initial exaggeration factor
    for iteration in range(iterations):
        # Step 4: Compute P affinities
        if iteration < 100:
            P = P_affinities(X_pca, perplexity) * exaggeration
        else:
            P = P_affinities(X_pca, perplexity)
        # Step 5: Calculate Q affinities and gradients
        dY, Q = grads(Y, P)
        # Step 6: Update Y
        Y += lr * dY
        # Step 7: Re-center Y
        Y -= np.mean(Y, axis=0)
        # Step 8: Print cost every 100 iterations
        if (iteration + 1) % 100 == 0:
            current_cost = cost(P, Q)
            print(f"Cost at iteration {iteration + 1}: {current_cost:.4f}")
        # Step 9: Adjust the exaggeration factor
        if iteration == 20:
            a_t = 0.8  # Change exaggeration factor after 20 iterations
    return Y