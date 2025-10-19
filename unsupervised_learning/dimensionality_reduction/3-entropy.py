#!/usr/bin/env python3
"""
Calculates the Shannon entropy and P affinities relative to a data point
"""


import numpy as np


def HP(Di, beta):
    """
    Calculates the Shannon entropy and P affinities for a data point
    """
    # Calculate the P affinities
    Pi = np.exp(-Di * beta)
    # Normalize P affinities to sum to 1
    Pi /= np.sum(Pi)
    # Calculate the Shannon entropy
    Hi = -np.sum(Pi * np.log2(Pi + 1e-12))  # Avoid log(0)
    return Hi, Pi
