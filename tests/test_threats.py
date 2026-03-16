"""Tests for threat classification."""

from gomoku.ai.threats import ThreatType, classify_move, classify_moves
from gomoku.board import Board
from gomoku.config import Player


def test_classify_move_win():
    board = Board()
    for col in range(4):
        board.place(7, col, Player.BLACK)

    info = classify_move(board, 7, 4, Player.BLACK)

    assert info.threat_type == ThreatType.WIN


def test_classify_move_block_win():
    board = Board()
    for col in range(4):
        board.place(0, col, Player.BLACK)

    info = classify_move(board, 0, 4, Player.WHITE)

    assert info.threat_type == ThreatType.BLOCK_WIN


def test_classify_move_open_four():
    board = Board()
    board.place(7, 4, Player.BLACK)
    board.place(7, 5, Player.BLACK)
    board.place(7, 6, Player.BLACK)

    info = classify_move(board, 7, 7, Player.BLACK)

    assert info.threat_type == ThreatType.OPEN_FOUR


def test_classify_move_double_open_three():
    board = Board()
    board.place(7, 6, Player.BLACK)
    board.place(7, 8, Player.BLACK)
    board.place(6, 7, Player.BLACK)
    board.place(8, 7, Player.BLACK)

    info = classify_move(board, 7, 7, Player.BLACK)

    assert info.threat_type == ThreatType.DOUBLE_OPEN_THREE


def test_classify_move_half_four_for_jump_four():
    board = Board()
    board.place(7, 3, Player.BLACK)
    board.place(7, 4, Player.BLACK)
    board.place(7, 6, Player.BLACK)
    board.place(7, 7, Player.BLACK)

    info = classify_move(board, 7, 5, Player.BLACK)

    assert info.attack_type == ThreatType.WIN


def test_classify_move_defense_detects_double_open_three():
    board = Board()
    board.place(7, 6, Player.BLACK)
    board.place(7, 8, Player.BLACK)
    board.place(6, 7, Player.BLACK)
    board.place(8, 7, Player.BLACK)

    info = classify_move(board, 7, 7, Player.WHITE)

    assert info.defense_type in {ThreatType.DOUBLE_OPEN_THREE, ThreatType.WIN}


def test_classify_move_other_when_no_strong_threat():
    board = Board()
    board.place(7, 7, Player.BLACK)
    board.place(8, 8, Player.WHITE)

    info = classify_move(board, 6, 6, Player.BLACK)

    assert info.threat_type == ThreatType.OTHER


def test_classify_moves_returns_matching_infos():
    board = Board()
    for col in range(4):
        board.place(7, col, Player.BLACK)

    infos = classify_moves(board, [(7, 4), (6, 6)], Player.BLACK)

    assert infos[0].threat_type == ThreatType.WIN
    assert infos[1].threat_type in {ThreatType.HALF_FOUR, ThreatType.OPEN_THREE, ThreatType.OTHER}
