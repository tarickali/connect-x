import pickle

import connectx.types as cxt

__all__ = ["save", "load", "make_state", "make_config"]


def make_config(
    shape: cxt.Shape,
    k: int,
    players: list[int],
) -> cxt.Config:
    return {"shape": shape, "k": k, "players": list(players)}


def make_state(grid: cxt.Grid, time: int, active: int) -> cxt.State:
    return {"grid": grid, "info": {"active": active, "time": time}}


def save(instance: cxt.Instance, filepath: str) -> None:
    with open(filepath, "wb+") as f:
        pickle.dump(instance, f)


def load(filepath: str) -> cxt.Instance:
    with open(filepath, "rb") as f:
        instance = pickle.load(f)
    return instance
