#!/usr/bin/env python3
'''
    function def absorbing(P): that
    determines if a markov chain is absorbing
'''


import numpy as np


def absorbing(P):
    '''
    Determines if a markov chain is absorbing.
    A Markov chain is absorbing if:
    1. It has at least one absorbing state
       absorbing state

    Args:
        P: numpy.ndarray transition matrix
    Returns: boolean
    '''
    if not isinstance(P, np.ndarray) or len(P.shape) != 2:
        return None
    n1, n2 = P.shape
    if n1 != n2:
        return None
    if not np.allclose(np.sum(P, axis=1), 1):
        return None
    absorbing_states = np.where(np.diag(P) == 1)[0]
    if len(absorbing_states) == 0:
        return False
    non_absorbing = np.where(np.diag(P) != 1)[0]
    if len(non_absorbing) == 0:
        return True
    R = P[non_absorbing][:, absorbing_states]
    Q = P[non_absorbing][:, non_absorbing]
    identity_matrix = np.eye(len(non_absorbing))
    try:
        fundamental_matrix = np.linalg.inv(identity_matrix - Q)
        reachability = np.dot(fundamental_matrix, R)
        return np.all(reachability.sum(axis=1) > 0)
    except np.linalg.LinAlgError:
        return False
