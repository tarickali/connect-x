import pickle

import connectx.types as cxt

__all__ = ["save", "load", "make_state"]


def save(instance: cxt.Instance, filepath: str) -> None:
    with open(filepath, "wb+") as f:
        pickle.dump(instance, f)


def load(filepath: str) -> cxt.Instance:
    with open(filepath, "rb") as f:
        instance = pickle.load(f)
    return instance


def make_state(grid: cxt.Grid, time: int, active: int) -> cxt.State:
    return {"grid": grid, "time": time, "active": active}
