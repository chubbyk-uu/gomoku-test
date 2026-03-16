"""Cython kernels for Gomoku threat analysis hotspots."""

import cython
cimport numpy as cnp

ctypedef cnp.int8_t grid_t


cdef inline int _line_tactical_score(int length, int open_ends):
    if length >= 5:
        return 200000
    if length == 4:
        if open_ends == 2:
            return 80000
        if open_ends == 1:
            return 30000
        return 0
    if length == 3:
        if open_ends == 2:
            return 8000
        if open_ends == 1:
            return 2000
        return 0
    if length == 2:
        if open_ends == 2:
            return 500
        if open_ends == 1:
            return 100
        return 0
    if length == 1:
        if open_ends == 2:
            return 30
        if open_ends == 1:
            return 5
        return 0
    return 0


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef tuple quick_pattern_summary(
    cnp.ndarray[grid_t, ndim=2] grid,
    int row,
    int col,
    int player,
):
    cdef int board_size = grid.shape[0]
    cdef int r
    cdef int c
    cdef int dr
    cdef int dc
    cdef int left_len
    cdef int right_len
    cdef bint left_open
    cdef bint right_open
    cdef int total_len
    cdef int open_ends
    cdef int i
    cdef bint has_immediate_threat = False
    cdef bint has_potential = False
    cdef int promising_directions = 0
    cdef int dr_values[4]
    cdef int dc_values[4]

    dr_values[0], dc_values[0] = 1, 0
    dr_values[1], dc_values[1] = 0, 1
    dr_values[2], dc_values[2] = 1, 1
    dr_values[3], dc_values[3] = 1, -1

    if grid[row, col] != 0:
        return False, False, False

    for i in range(4):
        dr = dr_values[i]
        dc = dc_values[i]
        left_len = 0
        r = row - dr
        c = col - dc
        while 0 <= r < board_size and 0 <= c < board_size and grid[r, c] == player:
            left_len += 1
            r -= dr
            c -= dc
        left_open = 0 <= r < board_size and 0 <= c < board_size and grid[r, c] == 0

        right_len = 0
        r = row + dr
        c = col + dc
        while 0 <= r < board_size and 0 <= c < board_size and grid[r, c] == player:
            right_len += 1
            r += dr
            c += dc
        right_open = 0 <= r < board_size and 0 <= c < board_size and grid[r, c] == 0

        total_len = 1 + left_len + right_len
        open_ends = left_open + right_open

        if total_len >= 5:
            return True, False, False
        if total_len == 4 and open_ends == 2:
            return False, True, False
        if total_len == 4 and open_ends == 1:
            has_immediate_threat = True
        if total_len == 3 and open_ends >= 1:
            has_immediate_threat = True
        if total_len >= 2:
            promising_directions += 1
        if total_len >= 3 and open_ends >= 1:
            has_potential = True

    return (
        False,
        False,
        has_immediate_threat or has_potential or promising_directions >= 2,
    )


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef tuple analyze_move(
    cnp.ndarray[grid_t, ndim=2] grid,
    int row,
    int col,
    int player,
):
    cdef int board_size = grid.shape[0]
    cdef int center = board_size // 2
    cdef int r
    cdef int c
    cdef int dr
    cdef int dc
    cdef int left_len
    cdef int right_len
    cdef bint left_open
    cdef bint right_open
    cdef int total_len
    cdef int open_ends
    cdef int i
    cdef int dr_values[4]
    cdef int dc_values[4]
    cdef int directional_scores[4]
    cdef int tmp
    cdef int total_score
    cdef int center_bias

    dr_values[0], dc_values[0] = 1, 0
    dr_values[1], dc_values[1] = 0, 1
    dr_values[2], dc_values[2] = 1, 1
    dr_values[3], dc_values[3] = 1, -1

    if grid[row, col] != 0:
        return False, -1

    for i in range(4):
        dr = dr_values[i]
        dc = dc_values[i]

        left_len = 0
        r = row - dr
        c = col - dc
        while 0 <= r < board_size and 0 <= c < board_size and grid[r, c] == player:
            left_len += 1
            r -= dr
            c -= dc
        left_open = 0 <= r < board_size and 0 <= c < board_size and grid[r, c] == 0

        right_len = 0
        r = row + dr
        c = col + dc
        while 0 <= r < board_size and 0 <= c < board_size and grid[r, c] == player:
            right_len += 1
            r += dr
            c += dc
        right_open = 0 <= r < board_size and 0 <= c < board_size and grid[r, c] == 0

        total_len = 1 + left_len + right_len
        if total_len >= 5:
            return True, 200000

        open_ends = left_open + right_open
        directional_scores[i] = _line_tactical_score(total_len, open_ends)

    if directional_scores[1] > directional_scores[0]:
        tmp = directional_scores[0]
        directional_scores[0] = directional_scores[1]
        directional_scores[1] = tmp
    if directional_scores[2] > directional_scores[1]:
        tmp = directional_scores[1]
        directional_scores[1] = directional_scores[2]
        directional_scores[2] = tmp
        if directional_scores[1] > directional_scores[0]:
            tmp = directional_scores[0]
            directional_scores[0] = directional_scores[1]
            directional_scores[1] = tmp
    if directional_scores[3] > directional_scores[2]:
        tmp = directional_scores[2]
        directional_scores[2] = directional_scores[3]
        directional_scores[3] = tmp
        if directional_scores[2] > directional_scores[1]:
            tmp = directional_scores[1]
            directional_scores[1] = directional_scores[2]
            directional_scores[2] = tmp
            if directional_scores[1] > directional_scores[0]:
                tmp = directional_scores[0]
                directional_scores[0] = directional_scores[1]
                directional_scores[1] = tmp

    total_score = (
        directional_scores[0]
        + directional_scores[1]
        + directional_scores[2]
        + directional_scores[3]
    )
    total_score += directional_scores[0] * directional_scores[1] // 20000

    center_bias = 2 * board_size - abs(row - center) - abs(col - center)
    return False, total_score + center_bias
