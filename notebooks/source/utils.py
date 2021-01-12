def ij_to_list_index(i, j, n):
    """
    Assuming you have a symetric nxn matrix M and take the entries of the upper triangular including the
    diagonal and then ravel it to transform it into a list. This function will transform a matrix location
    given row i and column j into the proper list index.
    :param i: row index of the matrix M
    :param j: column index of matrix M
    :param n: total number of rows / colums of n
    :return: The correct index of the lost
    """
    assert j >= i, "We only consider the upper triangular part..."

    index = 0
    for k in range(i):
        index += n - k - 1
    return index + j