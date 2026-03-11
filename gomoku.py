import pygame
import sys
import math
import time

# ============ Parameters ============
BOARD_SIZE = 15               # Number of rows and columns
GRID_SIZE = 40                # Grid spacing (pixels)
MARGIN = 20                   # Margin (pixels)
WINDOW_SIZE = MARGIN * 2 + GRID_SIZE * (BOARD_SIZE - 1)
FPS = 30                      # Frames per second

# ============ Color Definitions ============
BLACK_COLOR = (0, 0, 0)
WHITE_COLOR = (255, 255, 255)
BG_COLOR = (222, 184, 135)     # Background color (wood-like)
LINE_COLOR = (0, 0, 0)
RED = (255, 0, 0)

# ============ Board Initialization ============
def init_board():
    return [[0 for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]

# ============ Draw Board and Pieces ============
def draw_board(screen, board):
    screen.fill(BG_COLOR)
    # Draw horizontal lines
    for i in range(BOARD_SIZE):
        start_pos = (MARGIN, MARGIN + i * GRID_SIZE)
        end_pos = (MARGIN + (BOARD_SIZE - 1) * GRID_SIZE, MARGIN + i * GRID_SIZE)
        pygame.draw.line(screen, LINE_COLOR, start_pos, end_pos, 1)
    # Draw vertical lines
    for j in range(BOARD_SIZE):
        start_pos = (MARGIN + j * GRID_SIZE, MARGIN)
        end_pos = (MARGIN + j * GRID_SIZE, MARGIN + (BOARD_SIZE - 1) * GRID_SIZE)
        pygame.draw.line(screen, LINE_COLOR, start_pos, end_pos, 1)
    # Draw pieces
    for i in range(BOARD_SIZE):
        for j in range(BOARD_SIZE):
            if board[i][j] != 0:
                center = (MARGIN + j * GRID_SIZE, MARGIN + i * GRID_SIZE)
                if board[i][j] == 1:
                    pygame.draw.circle(screen, BLACK_COLOR, center, GRID_SIZE // 2 - 2)
                elif board[i][j] == 2:
                    pygame.draw.circle(screen, WHITE_COLOR, center, GRID_SIZE // 2 - 2)
                    pygame.draw.circle(screen, BLACK_COLOR, center, GRID_SIZE // 2 - 2, 1)
    pygame.display.flip()

# ============ Check for Win ============
def check_win(board, row, col):
    """
    Check if the piece at (row, col) forms a five-in-a-row.
    """
    if board[row][col] == 0:
        return False
    player = board[row][col]
    directions = [(1, 0), (0, 1), (1, 1), (1, -1)]
    for d in directions:
        count = 1
        # Extend in the positive direction
        i = 1
        while True:
            r = row + d[0] * i
            c = col + d[1] * i
            if 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE and board[r][c] == player:
                count += 1
                i += 1
            else:
                break
        # Extend in the negative direction
        i = 1
        while True:
            r = row - d[0] * i
            c = col - d[1] * i
            if 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE and board[r][c] == player:
                count += 1
                i += 1
            else:
                break
        if count >= 5:
            return True
    return False

# ============ Scoring Function ============
def get_score(count, blocks):
    """
    Score based on the number of continuous pieces (count) and the number of blocked ends (blocks)
    """
    if count >= 5:
        return 100000
    if count == 4:
        if blocks == 0:
            return 10000
        elif blocks == 1:
            return 1000
    if count == 3:
        if blocks == 0:
            return 1000
        elif blocks == 1:
            return 100
    if count == 2:
        if blocks == 0:
            return 100
        elif blocks == 1:
            return 10
    if count == 1:
        return 10
    return 0

def evaluate_board(board, ai_player):
    """
    Evaluate the board:
      - Calculate scores for AI and the opponent separately,
      - Return: score = (AI score) - (opponent score)
    """
    opponent = 1 if ai_player == 2 else 2

    def evaluate_for_player(player):
        score = 0
        # Only count starting points in four directions to avoid duplicate counting
        directions = [(1, 0), (0, 1), (1, 1), (1, -1)]
        for i in range(BOARD_SIZE):
            for j in range(BOARD_SIZE):
                if board[i][j] == player:
                    for d in directions:
                        # Skip if the previous position in this direction has the same piece (not a starting point)
                        prev_i = i - d[0]
                        prev_j = j - d[1]
                        if 0 <= prev_i < BOARD_SIZE and 0 <= prev_j < BOARD_SIZE and board[prev_i][prev_j] == player:
                            continue
                        count = 0
                        blocks = 0
                        x, y = i, j
                        # Count continuous pieces in this direction
                        while 0 <= x < BOARD_SIZE and 0 <= y < BOARD_SIZE and board[x][y] == player:
                            count += 1
                            x += d[0]
                            y += d[1]
                        # Check if the endpoint is blocked
                        if x < 0 or x >= BOARD_SIZE or y < 0 or y >= BOARD_SIZE or board[x][y] != 0:
                            blocks += 1
                        # Check the opposite direction from the starting point
                        x = i - d[0]
                        y = j - d[1]
                        if x < 0 or x >= BOARD_SIZE or y < 0 or y >= BOARD_SIZE or board[x][y] != 0:
                            blocks += 1
                        score += get_score(count, blocks)
        return score

    ai_score = evaluate_for_player(ai_player)
    opp_score = evaluate_for_player(opponent)
    return ai_score - opp_score

# ============ Generate Candidate Moves ============
def get_possible_moves(board):
    """
    Return all empty positions that are near existing pieces.
    """
    moves = []
    # If the board is empty, return the center position
    empty = True
    for i in range(BOARD_SIZE):
        for j in range(BOARD_SIZE):
            if board[i][j] != 0:
                empty = False
                break
        if not empty:
            break
    if empty:
        return [(BOARD_SIZE // 2, BOARD_SIZE // 2)]
    
    for i in range(BOARD_SIZE):
        for j in range(BOARD_SIZE):
            if board[i][j] == 0:
                found_neighbor = False
                # Check the surrounding 8 directions
                for di in range(-1, 2):
                    for dj in range(-1, 2):
                        ni = i + di
                        nj = j + dj
                        if 0 <= ni < BOARD_SIZE and 0 <= nj < BOARD_SIZE and board[ni][nj] != 0:
                            found_neighbor = True
                            break
                    if found_neighbor:
                        break
                if found_neighbor:
                    moves.append((i, j))
    return moves

# ============ Minimax Search (Alpha-Beta Pruning) ============
def minimax(board, depth, alpha, beta, maximizingPlayer, ai_player):
    """
    Parameters:
      board: current board (2D list)
      depth: search depth
      alpha, beta: pruning parameters
      maximizingPlayer: True if it's AI's turn (maximizing), False otherwise
      ai_player: AI's piece value (1 or 2)
    Returns: (score, move)
    """
    opponent = 1 if ai_player == 2 else 2

    # At leaf node, return board evaluation
    if depth == 0:
        return evaluate_board(board, ai_player), None

    moves = get_possible_moves(board)
    best_move = None

    if maximizingPlayer:
        max_eval = -math.inf
        for move in moves:
            i, j = move
            board[i][j] = ai_player
            # If this move directly wins, return immediately with a high score
            if check_win(board, i, j):
                board[i][j] = 0
                return 100000, move
            eval_score, _ = minimax(board, depth - 1, alpha, beta, False, ai_player)
            board[i][j] = 0
            if eval_score > max_eval:
                max_eval = eval_score
                best_move = move
            alpha = max(alpha, eval_score)
            if beta <= alpha:
                break
        return max_eval, best_move
    else:
        min_eval = math.inf
        for move in moves:
            i, j = move
            board[i][j] = opponent
            if check_win(board, i, j):
                board[i][j] = 0
                return -100000, move
            eval_score, _ = minimax(board, depth - 1, alpha, beta, True, ai_player)
            board[i][j] = 0
            if eval_score < min_eval:
                min_eval = eval_score
                best_move = move
            beta = min(beta, eval_score)
            if beta <= alpha:
                break
        return min_eval, best_move

# ============ Main Function ============
def main():
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
    pygame.display.set_caption("Gomoku")
    clock = pygame.time.Clock()

    # Outer loop：每次循环开始一局新对局
    while True:
        board = init_board()
        move_history = []  # 保存所有落子记录：格式 (row, col, player)
        human_color = None
        ai_color = None
        game_over = False
        winner = None

        # ---- Color selection ----
        font = pygame.font.SysFont(None, 36)
        text = font.render("Select your color: Press B for Black (first move), Press W for White", True, RED)
        screen.fill(BG_COLOR)
        screen.blit(text, (20, WINDOW_SIZE // 2 - 20))
        pygame.display.flip()

        waiting_color = True
        while waiting_color:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_b:
                        human_color = 1
                        ai_color = 2
                        waiting_color = False
                    elif event.key == pygame.K_w:
                        human_color = 2
                        ai_color = 1
                        waiting_color = False

        # In Gomoku, Black normally moves first
        turn = 1  # 1 for Black, 2 for White
        draw_board(screen, board)

        # ---- Main game loop ----
        while True:
            undo_triggered = False
            # 处理所有事件（包括悔棋、鼠标落子、重启/退出等）
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                # 游戏结束时处理重启/退出
                if game_over:
                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_r:
                            # 退出当前对局，重新开始新对局
                            game_over = False
                            break
                        elif event.key == pygame.K_q:
                            pygame.quit()
                            sys.exit()
                else:
                    # 游戏进行中，按 U 键悔棋（仅允许在游戏未结束时使用）
                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_u:
                            if len(move_history) > 0:
                                # 尽量撤销一轮（两步），如果不足两步则撤销所有
                                if len(move_history) >= 2:
                                    for _ in range(2):
                                        i, j, _ = move_history.pop()
                                        board[i][j] = 0
                                else:
                                    i, j, _ = move_history.pop()
                                    board[i][j] = 0
                                draw_board(screen, board)
                                turn = human_color  # 悔棋后回到玩家回合
                                undo_triggered = True
                    # 玩家落子事件（仅在玩家回合有效）
                    if not game_over and turn == human_color:
                        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                            pos = event.pos
                            # 计算棋盘坐标（四舍五入）
                            j = round((pos[0] - MARGIN) / GRID_SIZE)
                            i = round((pos[1] - MARGIN) / GRID_SIZE)
                            if 0 <= i < BOARD_SIZE and 0 <= j < BOARD_SIZE:
                                if board[i][j] == 0:
                                    board[i][j] = human_color
                                    move_history.append((i, j, human_color))
                                    draw_board(screen, board)
                                    if check_win(board, i, j):
                                        game_over = True
                                        winner = human_color
                                    else:
                                        turn = ai_color
            # 如果有悔棋操作，则跳过本帧后续处理
            if undo_triggered:
                clock.tick(FPS)
                continue

            # 如果游戏未结束且轮到 AI 落子，则进行 AI 走子
            if not game_over and turn == ai_color:
                pygame.time.wait(500)  # 小延时，便于观察
                depth = 2  # 搜索深度，可根据性能调整
                score, move = minimax(board, depth, -math.inf, math.inf, True, ai_color)
                if move is not None:
                    i, j = move
                    board[i][j] = ai_color
                    move_history.append((i, j, ai_color))
                    draw_board(screen, board)
                    if check_win(board, i, j):
                        game_over = True
                        winner = ai_color
                    else:
                        turn = human_color
                else:
                    # 无可用走法，判为平局
                    game_over = True
                    winner = 0

            # 游戏结束后显示结果及重启提示
            if game_over:
                font_large = pygame.font.SysFont(None, 48)
                if winner == human_color:
                    msg = "You win!"
                elif winner == ai_color:
                    msg = "Computer wins!"
                else:
                    msg = "Draw!"
                result_text = font_large.render(msg, True, RED)
                screen.blit(result_text, (WINDOW_SIZE // 2 - result_text.get_width() // 2,
                                           WINDOW_SIZE // 2 - result_text.get_height() // 2))
                font_small = pygame.font.SysFont(None, 36)
                prompt = font_small.render("Press R to restart, Q to quit", True, RED)
                screen.blit(prompt, (WINDOW_SIZE // 2 - prompt.get_width() // 2,
                                     WINDOW_SIZE // 2 + result_text.get_height() // 2 + 10))
                pygame.display.flip()
            clock.tick(FPS)

            # 如果游戏结束且玩家按下重启键，则退出当前内层循环，开始新一局
            if game_over:
                # 在 game_over 状态下，等待玩家按 R 或 Q（上面事件处理已处理此情况）
                # 此处循环等待直到 game_over 状态被重置
                waiting_restart = True
                while waiting_restart:
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            pygame.quit()
                            sys.exit()
                        if event.type == pygame.KEYDOWN:
                            if event.key == pygame.K_r:
                                waiting_restart = False
                                # 退出当前游戏循环，重新开始新对局
                                game_over = False
                                break
                            elif event.key == pygame.K_q:
                                pygame.quit()
                                sys.exit()
                    clock.tick(FPS)
                # 退出内层 while 循环，重新开始新对局
                break

if __name__ == "__main__":
    main()
