#!/usr/bin/env python3
"""Forward algorithm for Hidden Markov Model"""


import numpy as np


def forward(Observation, Emission, Transition, Initial):
    """Forward algorithm for HMM
    Return: P, F
    """
    if type(Observation) is not np.ndarray or len(Observation.shape) != 1:
        return None, None
    T = Observation.shape[0]
    if type(Emission) is not np.ndarray or len(Emission.shape) != 2:
        return None, None
    N, M = Emission.shape
    if (type(Transition) is not np.ndarray or len(Transition.shape) != 2 or
            Transition.shape[0] != N or Transition.shape[1] != N):
        return None, None
    if (type(Initial) is not np.ndarray or len(Initial.shape) != 2 or
            Initial.shape[0] != N or Initial.shape[1] != 1):
        return None, None
    F = np.zeros((N, T))
    F[:, 0] = Initial.reshape(-1) * Emission[:, Observation[0]]
    for t in range(1, T):
        for s in range(N):
            F[s, t] = np.sum(F[:, t-1] * Transition[:, s]) * \
                      Emission[s, Observation[t]]
    P = np.sum(F[:, -1])
    return P, F
