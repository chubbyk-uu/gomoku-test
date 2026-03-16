"""Cython kernels for Gomoku threat analysis hotspots."""

import cython
cimport numpy as cnp

ctypedef cnp.int8_t grid_t

cdef int _X = 1
cdef int _O = 2
cdef int _E = 0


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


cdef inline int _match_shape_code(int line[9]):
    cdef int start
    cdef int end

    if line[4] != _X:
        return 0

    for start in range(0, 5):
        end = start + 5
        if (
            line[start] == _X
            and line[start + 1] == _X
            and line[start + 2] == _X
            and line[start + 3] == _X
            and line[start + 4] == _X
        ):
            return 7

    for start in range(0, 4):
        end = start + 6
        if (
            line[start] == _E
            and line[start + 5] == _E
            and line[start + 1] == _X
            and line[start + 2] == _X
            and line[start + 3] == _X
            and line[start + 4] == _X
        ):
            return 6

    for start in range(0, 4):
        if (
            line[start + 1] == _X
            and line[start + 2] == _X
            and line[start + 3] == _X
            and line[start + 4] == _X
        ):
            if line[start] == _O and line[start + 5] == _E:
                return 5
            if line[start] == _E and line[start + 5] == _O:
                return 5

    for start in range(0, 5):
        if (
            line[start] == _X
            and line[start + 1] == _E
            and line[start + 2] == _X
            and line[start + 3] == _X
            and line[start + 4] == _X
        ):
            return 5
        if (
            line[start] == _X
            and line[start + 1] == _X
            and line[start + 2] == _X
            and line[start + 3] == _E
            and line[start + 4] == _X
        ):
            return 5
        if (
            line[start] == _X
            and line[start + 1] == _X
            and line[start + 2] == _E
            and line[start + 3] == _X
            and line[start + 4] == _X
        ):
            return 5

    for start in range(0, 4):
        if (
            line[start] == _E
            and line[start + 1] == _E
            and line[start + 2] == _X
            and line[start + 3] == _X
            and line[start + 4] == _X
            and line[start + 5] == _E
        ):
            return 4
        if (
            line[start] == _E
            and line[start + 1] == _X
            and line[start + 2] == _X
            and line[start + 3] == _X
            and line[start + 4] == _E
            and line[start + 5] == _E
        ):
            return 4

    for start in range(0, 4):
        if (
            line[start] == _E
            and line[start + 1] == _X
            and line[start + 2] == _E
            and line[start + 3] == _X
            and line[start + 4] == _X
            and line[start + 5] == _E
        ):
            return 4
        if (
            line[start] == _E
            and line[start + 1] == _X
            and line[start + 2] == _X
            and line[start + 3] == _E
            and line[start + 4] == _X
            and line[start + 5] == _E
        ):
            return 4

    for start in range(0, 3):
        if (
            line[start] == _O
            and line[start + 1] == _E
            and line[start + 2] == _X
            and line[start + 3] == _X
            and line[start + 4] == _X
            and line[start + 5] == _E
            and line[start + 6] == _E
        ):
            return 3
        if (
            line[start] == _E
            and line[start + 1] == _E
            and line[start + 2] == _X
            and line[start + 3] == _X
            and line[start + 4] == _X
            and line[start + 5] == _E
            and line[start + 6] == _O
        ):
            return 3

    for start in range(0, 4):
        if (
            line[start] == _O
            and line[start + 1] == _X
            and line[start + 2] == _X
            and line[start + 3] == _X
            and line[start + 4] == _E
            and line[start + 5] == _E
        ):
            return 3
        if (
            line[start] == _E
            and line[start + 1] == _E
            and line[start + 2] == _X
            and line[start + 3] == _X
            and line[start + 4] == _X
            and line[start + 5] == _O
        ):
            return 3

    for start in range(0, 4):
        if (
            line[start] == _O
            and line[start + 1] == _X
            and line[start + 2] == _E
            and line[start + 3] == _X
            and line[start + 4] == _X
            and line[start + 5] == _E
        ):
            return 3
        if (
            line[start] == _E
            and line[start + 1] == _X
            and line[start + 2] == _X
            and line[start + 3] == _E
            and line[start + 4] == _X
            and line[start + 5] == _O
        ):
            return 3
        if (
            line[start] == _O
            and line[start + 1] == _X
            and line[start + 2] == _X
            and line[start + 3] == _E
            and line[start + 4] == _X
            and line[start + 5] == _E
        ):
            return 3
        if (
            line[start] == _E
            and line[start + 1] == _X
            and line[start + 2] == _E
            and line[start + 3] == _X
            and line[start + 4] == _X
            and line[start + 5] == _O
        ):
            return 3

    for start in range(0, 4):
        if (
            line[start] == _E
            and line[start + 1] == _E
            and line[start + 2] == _X
            and line[start + 3] == _X
            and line[start + 4] == _E
            and line[start + 5] == _E
        ):
            return 2

    for start in range(0, 5):
        if (
            line[start] == _E
            and line[start + 1] == _X
            and line[start + 2] == _E
            and line[start + 3] == _X
            and line[start + 4] == _E
        ):
            return 2

    for start in range(0, 4):
        if (
            line[start] == _O
            and line[start + 1] == _X
            and line[start + 2] == _X
            and line[start + 3] == _E
            and line[start + 4] == _E
            and line[start + 5] == _E
        ):
            return 1
        if (
            line[start] == _E
            and line[start + 1] == _E
            and line[start + 2] == _E
            and line[start + 3] == _X
            and line[start + 4] == _X
            and line[start + 5] == _O
        ):
            return 1

    for start in range(0, 4):
        if (
            line[start] == _O
            and line[start + 1] == _X
            and line[start + 2] == _E
            and line[start + 3] == _X
            and line[start + 4] == _E
            and line[start + 5] == _E
        ):
            return 1
        if (
            line[start] == _E
            and line[start + 1] == _E
            and line[start + 2] == _X
            and line[start + 3] == _E
            and line[start + 4] == _X
            and line[start + 5] == _O
        ):
            return 1

    return 0


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef tuple count_shapes_on_line(
    cnp.ndarray[grid_t, ndim=2] grid,
    int player_val,
    int direction_index,
    int line_id,
    int board_size,
):
    cdef int rows[15]
    cdef int cols[15]
    cdef int seen_indices[15]
    cdef int counts[7]
    cdef int line[9]
    cdef int line_len = 0
    cdef int row
    cdef int col
    cdef int idx
    cdef int dr
    cdef int dc
    cdef int shape_code
    cdef int start
    cdef int end
    cdef int seen_idx
    cdef int r
    cdef int c
    cdef int i
    cdef int k
    cdef int total

    for i in range(15):
        seen_indices[i] = 0
    for i in range(7):
        counts[i] = 0

    if direction_index == 0:
        dr, dc = 1, 0
        for row in range(board_size):
            rows[line_len] = row
            cols[line_len] = line_id
            line_len += 1
    elif direction_index == 1:
        dr, dc = 0, 1
        for col in range(board_size):
            rows[line_len] = line_id
            cols[line_len] = col
            line_len += 1
    elif direction_index == 2:
        dr, dc = 1, 1
        total = line_id - (board_size - 1)
        row = 0 if total >= 0 else -total
        col = row + total
        while row < board_size and col < board_size:
            rows[line_len] = row
            cols[line_len] = col
            line_len += 1
            row += 1
            col += 1
    else:
        dr, dc = 1, -1
        total = line_id
        row = 0 if total < board_size else total - (board_size - 1)
        while row < board_size:
            col = total - row
            if 0 <= col < board_size:
                rows[line_len] = row
                cols[line_len] = col
                line_len += 1
            row += 1

    for idx in range(line_len):
        if seen_indices[idx]:
            continue
        row = rows[idx]
        col = cols[idx]
        if grid[row, col] != player_val:
            continue

        for i in range(9):
            k = i - 4
            r = row + dr * k
            c = col + dc * k
            if 0 <= r < board_size and 0 <= c < board_size:
                if grid[r, c] == player_val:
                    line[i] = _X
                elif grid[r, c] == 0:
                    line[i] = _E
                else:
                    line[i] = _O
            else:
                line[i] = _O

        shape_code = _match_shape_code(line)
        if shape_code:
            counts[shape_code - 1] += 1
            start = idx - 4
            if start < 0:
                start = 0
            end = idx + 5
            if end > line_len:
                end = line_len
            for seen_idx in range(start, end):
                if grid[rows[seen_idx], cols[seen_idx]] == player_val:
                    seen_indices[seen_idx] = 1

    return (
        counts[0],
        counts[1],
        counts[2],
        counts[3],
        counts[4],
        counts[5],
        counts[6],
    )
