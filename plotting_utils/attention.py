import torch
import numpy as np


def flat_to_matrix(flat_arr: np.array, pattern: str, n_x: int, n_y: int) -> np.array:
    """Convert from flat array to matrix using snake pattern either through rows or columns

    :param flat_arr: flattened array
    :param pattern: snake pattern with which the matrix was originally flattened
    :param n_x: first dimension of original matrix
    :param n_y: second dimension of original matrix
    :return: resulting matrix after reconstruction
    """

    new_arr = np.zeros((n_x, n_y))
    k = 1
    if pattern == "columns":
        for i in range(n_y):
            new_arr[:, i] = flat_arr[i*n_x: i*n_x + n_x][::k]
            k *= -1
    if pattern == "rows":
        for i in range(n_x):
            new_arr[i, :] = flat_arr[i*n_y: i*n_y + n_y][::k]
            k *= -1

    return new_arr


def matrix_to_flat_idx(row_idx: int, col_idx: int, pattern: str, n_x: int, n_y: int) -> int:
    """Get index in flattened array from matrix indices

    :param row_idx: row index in matrix
    :param col_idx: column index in matrix
    :param pattern: snake pattern used to flatten matrix
    :param n_x: first dimension of original matrix
    :param n_y: second dimension of original matrix
    :return: index in flattened array
    """

    if pattern == "columns":
        if (col_idx % 2) == 0:
            return col_idx*n_x + row_idx
        else:
            return col_idx*n_y + n_y - row_idx - 1

    if pattern == "rows":
        if (row_idx % 2) == 0:
            return row_idx*n_x + col_idx
        else:
            return row_idx*n_x + n_x - col_idx - 1


def attention_2D_viz(attn_weights, i, j, n_x=8, n_y=8):

    # -1 due to missing first element in sequence
    rows_idx = matrix_to_flat_idx(i, j, "rows", n_x, n_y) - 1
    cols_idx = matrix_to_flat_idx(i, j, "columns", n_x, n_y) - 1

    batch_sum_rows = torch.sum(attn_weights[0][:, rows_idx, :], axis=0).detach().numpy()
    batch_sum_cols = torch.sum(attn_weights[1][:, cols_idx, :], axis=0).detach().numpy()

    complete_rows = np.pad(batch_sum_rows, (1, 0), constant_values=5)
    complete_cols = np.pad(batch_sum_cols, (1, 0), constant_values=5)

    matrix_rows = flat_to_matrix(complete_rows, "rows", n_x, n_y)
    matrix_cols = flat_to_matrix(complete_cols, "columns", n_x, n_y)

    return matrix_rows + matrix_cols


def attention_1D_viz(attn_weights, i, j, n_x=8, n_y=8):

    # -1 due to missing first element in sequence
    rows_idx = matrix_to_flat_idx(i, j, "rows", n_x, n_y) - 1

    batch_sum_rows = torch.sum(attn_weights[:, rows_idx, :], axis=0).detach().numpy()

    complete_rows = np.pad(batch_sum_rows, (1, 0), constant_values=5)

    matrix_rows = flat_to_matrix(complete_rows, "rows", n_x, n_y)

    return matrix_rows
