"""Environment split helpers."""

from __future__ import annotations


def build_env_splits(
    n_train_envs: int,
    n_val_envs: int,
    n_test_envs: int,
) -> dict[str, list[str]]:
    """Build disjoint environment splits with deterministic names."""
    total_envs = n_train_envs + n_val_envs + n_test_envs
    env_ids = [f"env_{idx}" for idx in range(total_envs)]
    train_end = n_train_envs
    val_end = train_end + n_val_envs
    splits = {
        "train": env_ids[:train_end],
        "val": env_ids[train_end:val_end],
        "test": env_ids[val_end:],
    }
    split_union = set().union(*map(set, splits.values()))
    assert len(split_union) == total_envs, "Environment splits must be disjoint."
    return splits
