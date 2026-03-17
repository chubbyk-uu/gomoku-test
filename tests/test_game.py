"""Tests for GameController state transitions and helpers."""

from __future__ import annotations

from dataclasses import dataclass

import pygame

import gomoku.game as game_module
from gomoku.config import GameState, Player


@dataclass
class _DummyClock:
    ticks: list[int]

    def tick(self, fps: int) -> None:
        self.ticks.append(fps)


class _DummyRenderer:
    def __init__(self, screen: object) -> None:
        self.screen = screen
        self.pixel_result: tuple[int, int] | None = None
        self.menu_draws = 0
        self.board_draws = 0
        self.game_over_texts: list[str] = []

    def draw_menu(self) -> None:
        self.menu_draws += 1

    def draw_board(self, board: object) -> None:
        self.board_draws += 1

    def draw_game_over(self, winner_text: str) -> None:
        self.game_over_texts.append(winner_text)

    def pixel_to_board(self, pos: tuple[int, int]) -> tuple[int, int] | None:
        return self.pixel_result


def _make_controller(monkeypatch) -> game_module.GameController:
    clock = _DummyClock([])
    screen = object()

    monkeypatch.setattr(game_module.pygame, "init", lambda: None)
    monkeypatch.setattr(game_module.pygame.display, "set_mode", lambda size: screen)
    monkeypatch.setattr(game_module.pygame.display, "set_caption", lambda title: None)
    monkeypatch.setattr(game_module.pygame.display, "flip", lambda: None)
    monkeypatch.setattr(game_module.pygame.time, "Clock", lambda: clock)
    monkeypatch.setattr(game_module.pygame.time, "wait", lambda ms: None)
    monkeypatch.setattr(game_module, "Renderer", _DummyRenderer)

    controller = game_module.GameController()
    controller._start_new_game()
    return controller


def test_menu_black_key_enters_playing(monkeypatch):
    controller = _make_controller(monkeypatch)
    event = pygame.event.Event(pygame.KEYDOWN, key=pygame.K_b)
    monkeypatch.setattr(game_module.pygame.event, "get", lambda: [event])

    controller._handle_menu_events()

    assert controller._state == GameState.PLAYING
    assert controller._human_player == Player.BLACK
    assert controller._ai_player == Player.WHITE
    assert controller._turn == Player.BLACK


def test_game_over_restart_returns_to_menu(monkeypatch):
    controller = _make_controller(monkeypatch)
    controller._state = GameState.GAME_OVER
    event = pygame.event.Event(pygame.KEYDOWN, key=pygame.K_r)
    monkeypatch.setattr(game_module.pygame.event, "get", lambda: [event])

    controller._handle_game_over_events()

    assert controller._state == GameState.MENU


def test_ai_turn_without_move_declares_draw(monkeypatch):
    controller = _make_controller(monkeypatch)
    controller._enter_playing()
    controller._turn = controller._ai_player
    monkeypatch.setattr(controller._ai, "find_best_move", lambda board: None)

    controller._ai_turn()

    assert controller._state == GameState.GAME_OVER
    assert controller._winner_text == "Draw!"
    assert controller.renderer.game_over_texts[-1] == "Draw!"


def test_undo_handles_zero_one_and_multiple_moves(monkeypatch):
    controller = _make_controller(monkeypatch)
    controller._enter_playing()

    controller._undo()
    assert controller._board.move_history == []

    controller._board.place(7, 7, Player.BLACK)
    controller._turn = controller._ai_player
    controller._undo()
    assert controller._board.move_history == []
    assert controller._turn == controller._human_player

    controller._board.place(7, 7, Player.BLACK)
    controller._board.place(7, 8, Player.WHITE)
    controller._board.place(8, 8, Player.BLACK)
    controller._turn = controller._ai_player
    controller._undo()

    assert controller._board.move_history == [(7, 7, Player.BLACK)]
    assert controller._turn == controller._human_player
