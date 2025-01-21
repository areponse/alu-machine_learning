#!/usr/bin/env python3

"""
Matrix addition
"""
# import numpy as np

def shape(matrix):
    """ Calculates the shape of a matrix """
    shape = []
    while isinstance(matrix, list):
        shape.append(len(matrix))
        matrix = matrix[0]
    return shape

def add_matrices(mat1, mat2):
    """Add the two matrices with any number of dimensions

    Args:
        mat1 (_type_): _description_
        mat2 (_type_): _description_

    Returns:
        _type_: _description_
    """
    # do it with loops
    if shape(mat1) != shape(mat2):
        return None
    matrix = []
    if not isinstance(mat1[0], list):
        return [mat1[i] + mat2[i] for i in range(len(mat1))]
    for i in range(len(mat1)):
        matrix.append(add_matrices(mat1[i], mat2[i]))
    return matrix
