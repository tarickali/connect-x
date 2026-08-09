"""Contract every ``GameEngine`` must satisfy.

Add a new game by adding it to ``ENGINES`` below. If it passes this file, the
arena, adapters, recorders, and dataset tooling will work with it; if it does
not, they will fail in ways that are much harder to diagnose than a red test.

These assertions are deliberately dynamics-agnostic. Nothing here assumes
gravity, a column-shaped action space, or that tokens stay where they land — so
a free-placement or piece-removing game can be checked against exactly the same
contract.
"""

from __future__ import annotations

import numpy as np
import pytest

from connectx import Game, GameEngine, make_config, preset
from connectx.types import Config

#: (name, factory, config). Extend this when adding a game.
ENGINES = [
    ("Game/connect4", Game, preset("connect4")),
    ("Game/tiny", Game, preset("tiny")),
    ("Game/three-player", Game, make_config((5, 6), 3, [1, 2, 3])),
]

IDS = [name for name, _, _ in ENGINES]
CASES = [(factory, config) for _, factory, config in ENGINES]


@pytest.fixture(params=CASES, ids=IDS)
def engine_case(request):
    factory, config = request.param
    return factory, config


def legal_actions(engine) -> list[int]:
    return [i for i, ok in enumerate(np.asarray(engine.actions)) if ok]


def play_random(engine, rng, limit: int = 10_000) -> int:
    moves = 0
    while not engine.terminal() and moves < limit:
        options = legal_actions(engine)
        assert options, "terminal() must be True when no action is legal"
        engine.transition(int(rng.choice(options)))
        moves += 1
    return moves


class TestProtocol:
    def test_satisfies_the_runtime_protocol(self, engine_case) -> None:
        factory, config = engine_case
        assert isinstance(factory(config), GameEngine)

    def test_accepts_the_factory_signature(self, engine_case) -> None:
        # The harness builds engines as engine(config) and engine(config, record=True).
        factory, config = engine_case
        factory(config)
        factory(config, record=True)

    def test_exposes_its_config(self, engine_case) -> None:
        factory, config = engine_case
        assert factory(config).config == config

    def test_action_space_size_is_positive(self, engine_case) -> None:
        factory, config = engine_case
        assert factory(config).action_space_size >= 1


class TestLifecycle:
    def test_start_returns_state_and_actions(self, engine_case) -> None:
        factory, config = engine_case
        engine = factory(config)
        state, actions = engine.start()
        assert "grid" in state and "info" in state
        assert state["info"]["time"] == 0
        assert 0 <= state["info"]["active"] < len(config["players"])
        assert len(np.asarray(actions)) == engine.action_space_size

    def test_reset_matches_start(self, engine_case) -> None:
        factory, config = engine_case
        engine = factory(config)
        engine.start()
        engine.transition(legal_actions(engine)[0])
        state, _ = engine.reset()
        assert state["info"]["time"] == 0

    def test_fresh_game_is_not_terminal(self, engine_case) -> None:
        factory, config = engine_case
        engine = factory(config)
        engine.start()
        assert not engine.terminal()

    def test_legal_actions_agrees_with_actions(self, engine_case) -> None:
        factory, config = engine_case
        engine = factory(config)
        engine.start()
        np.testing.assert_array_equal(
            np.asarray(engine.legal_actions()), np.asarray(engine.actions)
        )


class TestStateIsolation:
    def test_grid_is_not_writeable(self, engine_case) -> None:
        factory, config = engine_case
        engine = factory(config)
        state, _ = engine.start()
        with pytest.raises(ValueError):
            state["grid"][0, 0] = 7

    def test_info_is_a_copy(self, engine_case) -> None:
        factory, config = engine_case
        engine = factory(config)
        state, _ = engine.start()
        state["info"]["time"] = 999
        assert engine.state["info"]["time"] == 0


class TestTransitions:
    def test_time_advances_and_turn_rotates(self, engine_case) -> None:
        factory, config = engine_case
        engine = factory(config)
        engine.start()
        n_players = len(config["players"])
        for move in range(1, min(6, n_players * 2 + 1)):
            options = legal_actions(engine)
            if not options:
                break
            state, _ = engine.transition(options[0])
            assert state["info"]["time"] == move
            assert 0 <= state["info"]["active"] < n_players

    def test_illegal_action_raises(self, engine_case) -> None:
        factory, config = engine_case
        engine = factory(config)
        engine.start()
        illegal = engine.action_space_size + 5
        with pytest.raises((ValueError, IndexError)):
            engine.transition(illegal)

    def test_games_terminate(self, engine_case) -> None:
        factory, config = engine_case
        rng = np.random.default_rng(0)
        for _ in range(10):
            engine = factory(config)
            engine.start()
            play_random(engine, rng)
            assert engine.terminal()

    def test_moving_after_the_end_raises(self, engine_case) -> None:
        factory, config = engine_case
        rng = np.random.default_rng(1)
        engine = factory(config)
        engine.start()
        play_random(engine, rng)
        with pytest.raises((RuntimeError, ValueError)):
            engine.transition(0)


class TestOutcome:
    def test_report_is_empty_before_the_end(self, engine_case) -> None:
        factory, config = engine_case
        engine = factory(config)
        engine.start()
        report = engine.report()
        assert report["winner"] is None
        assert report["tie"] is False

    def test_rewards_are_zero_before_the_end(self, engine_case) -> None:
        factory, config = engine_case
        engine = factory(config)
        engine.start()
        rewards = engine.rewards()
        assert len(rewards) == len(config["players"])
        assert not np.asarray(rewards).any()

    def test_terminal_outcome_is_a_win_or_a_tie_but_never_both(self, engine_case) -> None:
        factory, config = engine_case
        rng = np.random.default_rng(2)
        for _ in range(20):
            engine = factory(config)
            engine.start()
            play_random(engine, rng)
            report = engine.report()
            # At the end, exactly one of "somebody won" and "it was a tie".
            won = report["winner"] is not None
            assert won != bool(report["tie"])
            if won:
                seat = report["winner"]["id"]
                assert config["players"][seat] == report["winner"]["token"]

    def test_rewards_match_the_report(self, engine_case) -> None:
        factory, config = engine_case
        rng = np.random.default_rng(3)
        for _ in range(20):
            engine = factory(config)
            engine.start()
            play_random(engine, rng)
            report = engine.report()
            rewards = np.asarray(engine.rewards())
            if report["winner"] is None:
                assert (rewards == rewards[0]).all()
            else:
                winner = report["winner"]["id"]
                assert rewards[winner] == rewards.max()
                assert rewards[winner] > rewards.min()


class TestStepApi:
    def test_step_agrees_with_transition(self, engine_case) -> None:
        factory, config = engine_case
        engine = factory(config)
        engine.start()
        result = engine.step(legal_actions(engine)[0])
        assert result.terminated == engine.terminal()
        assert result.report == engine.report()
        np.testing.assert_allclose(result.rewards, engine.rewards())
        assert len(result.rewards) == len(config["players"])


class TestRecording:
    def test_trajectory_replays_to_the_same_positions(self, engine_case) -> None:
        factory, config = engine_case
        rng = np.random.default_rng(4)
        engine = factory(config, record=True)
        engine.start()
        play_random(engine, rng)
        trajectory = engine.trajectory()

        # replay() must reconstruct positions through the same engine, not by
        # assuming a placement rule. This is what breaks first for a new game.
        replayed = list(trajectory.replay(factory))
        assert len(replayed) == len(trajectory.steps)

        check = factory(config)
        state, _ = check.start()
        for (seen_state, action), step in zip(replayed, trajectory.steps, strict=True):
            np.testing.assert_array_equal(seen_state["grid"], state["grid"])
            assert action == step.action
            state, _ = check.transition(action)

    def test_trajectory_records_every_move(self, engine_case) -> None:
        factory, config = engine_case
        rng = np.random.default_rng(5)
        engine = factory(config, record=True)
        engine.start()
        moves = play_random(engine, rng)
        assert len(engine.trajectory().steps) == moves

    def test_recording_off_by_default(self, engine_case) -> None:
        factory, config = engine_case
        engine = factory(config)
        engine.start()
        with pytest.raises(RuntimeError):
            engine.trajectory()


class TestHarnessIntegration:
    """The engine must actually drive the measurement and training stacks."""

    def test_play_match_runs(self, engine_case) -> None:
        from connectx.arena import play_match

        factory, config = engine_case
        n_players = len(config["players"])
        specs = ["random"] * n_players
        # Seat rotation requires a whole number of rotations.
        games = 2 * n_players
        match = play_match(config, specs, games=games, seed=0, engine=factory)
        assert match.games == games
        assert sum(match.wins) + match.draws == games

    def test_build_supervised_runs(self, engine_case) -> None:
        from connectx.dataset import build_supervised

        factory, config = engine_case
        rng = np.random.default_rng(6)
        engine = factory(config, record=True)
        engine.start()
        play_random(engine, rng)
        data = build_supervised([engine.trajectory()], engine=factory)
        width = factory(config).action_space_size
        assert data["policies"].shape[1] == width
        np.testing.assert_allclose(data["policies"].sum(axis=1), 1.0)


class TestConfigIsExtensible:
    def test_unknown_keys_survive_validation(self) -> None:
        # A new game will need extra config fields; validate_config must not
        # reject what it does not recognize.
        from connectx.config import validate_config

        config: Config = {"shape": (6, 7), "k": 4, "players": [1, 2]}  # type: ignore[assignment]
        config["gravity"] = False  # type: ignore[typeddict-unknown-key]
        assert validate_config(config) is config
        assert config["gravity"] is False  # type: ignore[typeddict-item]
