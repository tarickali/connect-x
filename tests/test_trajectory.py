from connectx import ReplayMemory, Trajectory, TrajectoryStep
from connectx.utils import make_config


class TestReplayMemory:
    def test_push_and_cap(self) -> None:
        cfg = make_config((6, 7), 4, [1, 2])
        mem = ReplayMemory(max_episodes=2)
        mem.push(Trajectory(config=cfg, steps=[]))
        mem.push(Trajectory(config=cfg, steps=[]))
        mem.push(Trajectory(config=cfg, steps=[]))
        assert len(mem) == 2

    def test_trajectory_numpy(self) -> None:
        cfg = make_config((6, 7), 4, [1, 2])
        tr = Trajectory(
            config=cfg,
            steps=[
                TrajectoryStep(
                    action=0,
                    player_index=0,
                    player_token=1,
                    time_after=1,
                    reward=1.0,
                )
            ],
        )
        d = tr.as_numpy()
        assert d["actions"].shape == (1,)
        assert d["rewards"][0] == 1.0
