import numpy as np
import pytest

import connectx.functional as cxf
from agents import (
    GreedyAgent,
    MCTSAgent,
    MinimaxAgent,
    RandomAgent,
    available_agents,
    make_agent,
)
from connectx import Game, make_config, preset
from connectx.types import Actions, Config, State
from connectx.utils import make_state


@pytest.fixture
def open_board() -> tuple[State, Actions]:
    grid = np.zeros((6, 7), dtype=np.uint8)
    return make_state(grid, time=0, active=0), cxf.generate_actions(grid)


def win_in_one(token: int) -> tuple[State, Actions]:
    """Three of ``token`` on the bottom row; column 3 completes the line."""
    grid = np.zeros((6, 7), dtype=np.uint8)
    grid[5, 0:3] = token
    return make_state(grid, time=3, active=0), np.ones(7, dtype=np.uint8)


class TestRandomAgent:
    def test_returns_a_legal_column(self) -> None:
        grid = np.zeros((6, 7), dtype=np.uint8)
        grid[:, 2:] = 1
        actions = np.array([1, 1, 0, 0, 0, 0, 0], dtype=np.uint8)
        agent = RandomAgent(seed=0)
        for _ in range(50):
            action = agent.select(make_state(grid, 0, 0), actions)
            assert actions[action] == 1

    def test_same_seed_gives_same_moves(self, open_board) -> None:
        state, actions = open_board
        first = [RandomAgent(seed=7).select(state, actions) for _ in range(5)]
        second = [RandomAgent(seed=7).select(state, actions) for _ in range(5)]
        assert first == second

    def test_different_seeds_diverge(self, open_board) -> None:
        state, actions = open_board
        a = [int(RandomAgent(seed=1).select(state, actions)) for _ in range(30)]
        b = [int(RandomAgent(seed=2).select(state, actions)) for _ in range(30)]
        assert a != b

    def test_does_not_touch_the_global_rng(self, open_board) -> None:
        state, actions = open_board
        np.random.seed(0)
        expected = np.random.random()
        np.random.seed(0)
        RandomAgent(seed=3).select(state, actions)
        assert np.random.random() == expected

    def test_returns_a_python_int(self, open_board) -> None:
        state, actions = open_board
        assert type(RandomAgent(seed=0).select(state, actions)) is int

    def test_raises_without_legal_moves(self) -> None:
        grid = np.ones((6, 7), dtype=np.uint8)
        with pytest.raises(ValueError):
            RandomAgent(seed=0).select(make_state(grid, 0, 0), np.zeros(7, np.uint8))


class TestGreedyAgent:
    def test_takes_the_win(self) -> None:
        config = preset("connect4")
        state, actions = win_in_one(token=1)
        assert GreedyAgent(config, seed=0).select(state, actions) == 3

    def test_blocks_the_loss(self) -> None:
        config = preset("connect4")
        state, actions = win_in_one(token=2)
        assert GreedyAgent(config, seed=0).select(state, actions) == 3

    def test_prefers_the_win_over_the_block(self) -> None:
        config = preset("connect4")
        grid = np.zeros((6, 7), dtype=np.uint8)
        grid[5, 0:3] = 1  # we can win at column 3
        grid[4, 0:3] = 2  # opponent could win at column 3 next turn too
        state = make_state(grid, time=6, active=0)
        assert GreedyAgent(config, seed=0).select(state, np.ones(7, np.uint8)) == 3

    def test_works_with_three_players(self) -> None:
        config = make_config((5, 6), 3, [1, 2, 3])
        grid = np.zeros((5, 6), dtype=np.uint8)
        grid[4, 0:2] = 3  # player 3 threatens; we are player 1
        state = make_state(grid, time=2, active=0)
        assert GreedyAgent(config, seed=0).select(state, cxf.generate_actions(grid)) == 2

    def test_needs_a_config(self) -> None:
        state, actions = win_in_one(1)
        with pytest.raises(RuntimeError, match="config"):
            GreedyAgent(seed=0).select(state, actions)


class TestMinimaxAgent:
    @pytest.mark.parametrize("depth", [1, 2, 3, 4, 5])
    def test_takes_immediate_win_at_every_depth(self, depth: int) -> None:
        # Regression: evaluate() used to receive the wrong opponent token at odd
        # depths, so the default depth of 3 scored the opponent's pieces as its
        # own. Correct play must not depend on the parity of the depth.
        config = preset("connect4")
        state, actions = win_in_one(token=1)
        assert MinimaxAgent(config, depth=depth, seed=0).select(state, actions) == 3

    @pytest.mark.parametrize("depth", [2, 3, 4, 5])
    def test_blocks_opponent_win_at_every_depth(self, depth: int) -> None:
        config = preset("connect4")
        state, actions = win_in_one(token=2)
        assert MinimaxAgent(config, depth=depth, seed=0).select(state, actions) == 3

    def test_prefers_the_faster_win(self) -> None:
        config = preset("connect4")
        state, actions = win_in_one(token=1)
        agent = MinimaxAgent(config, depth=6, seed=0)
        assert agent.select(state, actions) == 3

    def test_rejects_more_than_two_players(self) -> None:
        with pytest.raises(ValueError, match="two players"):
            MinimaxAgent(make_config((6, 7), 4, [1, 2, 3]))

    def test_alpha_beta_prunes(self) -> None:
        config = preset("connect4")
        state, actions = (
            make_state(cxf.create_grid((6, 7)), 0, 0),
            cxf.generate_actions(cxf.create_grid((6, 7))),
        )
        agent = MinimaxAgent(config, depth=5, seed=0)
        agent.select(state, actions)
        # A full depth-5 tree on seven columns is 7**5 = 16807 nodes.
        assert agent.nodes < 7**5

    def test_transposition_table_can_be_disabled(self) -> None:
        config = preset("connect4")
        state, actions = win_in_one(token=1)
        agent = MinimaxAgent(config, depth=4, use_transpositions=False, seed=0)
        assert agent.select(state, actions) == 3

    def test_time_limit_bounds_the_search(self) -> None:
        import time

        config = preset("connect4")
        grid = cxf.create_grid((6, 7))
        agent = MinimaxAgent(config, depth=20, time_limit=0.2, seed=0)
        started = time.perf_counter()
        action = agent.select(make_state(grid, 0, 0), cxf.generate_actions(grid))
        assert time.perf_counter() - started < 3.0
        assert 0 <= action < 7

    def test_can_switch_variants_via_reset(self) -> None:
        agent = MinimaxAgent(preset("connect4"), depth=3, seed=0)
        other = make_config((5, 6), 3, [1, 2])
        agent.reset(other)
        grid = cxf.create_grid(other["shape"])
        action = agent.select(make_state(grid, 0, 0), cxf.generate_actions(grid))
        assert 0 <= action < other["shape"][1]


class TestMCTSAgent:
    def test_takes_the_win(self) -> None:
        config = preset("connect4")
        state, actions = win_in_one(token=1)
        assert MCTSAgent(config, simulations=300, seed=0).select(state, actions) == 3

    def test_reproducible_with_a_seed(self) -> None:
        config = preset("small")
        grid = cxf.create_grid(config["shape"])
        state, actions = make_state(grid, 0, 0), cxf.generate_actions(grid)
        first = MCTSAgent(config, simulations=120, seed=11).select(state, actions)
        second = MCTSAgent(config, simulations=120, seed=11).select(state, actions)
        assert first == second

    def test_supports_more_than_two_players(self) -> None:
        config = make_config((5, 6), 3, [1, 2, 3])
        grid = cxf.create_grid(config["shape"])
        agent = MCTSAgent(config, simulations=80, seed=0)
        action = agent.select(make_state(grid, 0, 0), cxf.generate_actions(grid))
        assert 0 <= action < config["shape"][1]

    def test_payoff_table_is_zero_sum(self) -> None:
        # Backups come from a table built once in reset(); a simulation must
        # never allocate, and the table has to stay zero-sum for any seat count.
        for players in ([1, 2], [1, 2, 3], [1, 2, 3, 4]):
            config = make_config((6, 7), 4, players)
            agent = MCTSAgent(config, simulations=1, seed=0)
            # Keyed by winning seat; -1 is a draw.
            assert agent._payoffs[-1] == [0.0] * len(players)
            for seat in range(len(players)):
                payoff = agent._payoffs[seat]
                assert payoff[seat] == 1.0
                assert sum(payoff) == pytest.approx(0.0)

    def test_payoff_table_follows_reset(self) -> None:
        agent = MCTSAgent(preset("connect4"), simulations=1, seed=0)
        agent.reset(make_config((5, 6), 3, [1, 2, 3]))
        assert len(agent._payoffs[1]) == 3

    def test_backup_keeps_the_root_consistent(self) -> None:
        config = preset("small")
        grid = cxf.create_grid(config["shape"])
        agent = MCTSAgent(config, simulations=200, seed=0)
        agent.select(make_state(grid, 0, 0), cxf.generate_actions(grid))
        # Every simulation increments exactly one child of the root.
        assert sum(agent.last_visits.values()) == 200

    def test_visit_counts_exposed(self) -> None:
        config = preset("small")
        grid = cxf.create_grid(config["shape"])
        agent = MCTSAgent(config, simulations=100, seed=0)
        agent.select(make_state(grid, 0, 0), cxf.generate_actions(grid))
        assert sum(agent.last_visits.values()) > 0


class TestRegistry:
    def test_lists_agents(self) -> None:
        assert {"random", "greedy", "minimax", "mcts"} <= set(available_agents())

    def test_make_agent_parses_options(self) -> None:
        agent = make_agent("minimax:depth=6", preset("connect4"), seed=0)
        assert isinstance(agent, MinimaxAgent)
        assert agent.depth == 6

    def test_make_agent_parses_bools_and_floats(self) -> None:
        agent = make_agent(
            "mcts:simulations=50,exploration=0.9,greedy_rollouts=false",
            preset("connect4"),
        )
        assert agent.simulations == 50
        assert agent.exploration == pytest.approx(0.9)
        assert agent.greedy_rollouts is False

    def test_name_keeps_the_full_spec(self) -> None:
        assert make_agent("minimax:depth=2", preset("connect4")).name == "minimax:depth=2"

    def test_unknown_agent(self) -> None:
        with pytest.raises(KeyError):
            make_agent("nonexistent")

    def test_bad_option_syntax(self) -> None:
        with pytest.raises(ValueError, match="key=value"):
            make_agent("minimax:depth", preset("connect4"))


class TestLadder:
    """The ordering of the baselines is the thing experiments are read against."""

    @staticmethod
    def _duel(config: Config, first: str, second: str, games: int = 12) -> int:
        wins = 0
        for index in range(games):
            specs = [first, second] if index % 2 == 0 else [second, first]
            agents = [
                make_agent(s, config, seed=index * 10 + i) for i, s in enumerate(specs)
            ]
            game = Game(config)
            state, actions = game.start()
            while not game.terminal():
                action = agents[state["info"]["active"]].select(state, actions)
                state, actions = game.transition(action)
            report = game.report()
            if report["winner"] is not None and specs[report["winner"]["id"]] == first:
                wins += 1
        return wins

    def test_greedy_beats_random(self) -> None:
        assert self._duel(preset("small"), "greedy", "random") >= 10

    def test_minimax_beats_greedy(self) -> None:
        assert self._duel(preset("small"), "minimax:depth=4", "greedy") >= 8
