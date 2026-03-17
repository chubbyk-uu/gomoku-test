"""Tests for Renderer helpers."""

from gomoku.config import GRID_SIZE, MARGIN, WINDOW_SIZE
from gomoku.ui.renderer import Renderer


def test_pixel_to_board_maps_exact_intersections():
    renderer = Renderer(object())

    assert renderer.pixel_to_board((MARGIN, MARGIN)) == (0, 0)
    assert renderer.pixel_to_board((WINDOW_SIZE - MARGIN, WINDOW_SIZE - MARGIN)) == (14, 14)


def test_pixel_to_board_rejects_negative_and_far_outside_positions():
    renderer = Renderer(object())

    assert renderer.pixel_to_board((-GRID_SIZE, -GRID_SIZE)) is None
    assert renderer.pixel_to_board((WINDOW_SIZE + GRID_SIZE, WINDOW_SIZE + GRID_SIZE)) is None
