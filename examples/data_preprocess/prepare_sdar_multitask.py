import argparse
import logging
import os
from copy import deepcopy

import pandas as pd

from verl.utils.hdfs_io import copy, makedirs


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

TASKS = ("alfworld", "search", "webshop")
EMPTY_PROMPT = [{"role": "user", "content": ""}]


def _sample_dataframe(df: pd.DataFrame, size: int, seed: int) -> pd.DataFrame:
    replace = len(df) < size
    if replace:
        logger.warning("Sampling %s rows with replacement from %s available rows.", size, len(df))
    return df.sample(n=size, replace=replace, random_state=seed).reset_index(drop=True)


def _dummy_task_dataframe(task_name: str, split: str, size: int) -> pd.DataFrame:
    rows = []
    for idx in range(size):
        rows.append(
            {
                "data_source": task_name,
                "prompt": deepcopy(EMPTY_PROMPT),
                "ability": "agent",
                "reward_model": None,
                "extra_info": {
                    "split": split,
                    "index": idx,
                    "task_name": task_name,
                },
                "metadata": None,
                "env_kwargs": {"task_name": task_name},
                "task_name": task_name,
            }
        )
    return pd.DataFrame(rows)


def _as_dict(value):
    return deepcopy(value) if isinstance(value, dict) else {}


def _process_search_row(row: pd.Series, split: str, idx: int) -> dict:
    item = row.to_dict()
    env_kwargs = _as_dict(item.get("env_kwargs"))
    extra_info = _as_dict(item.get("extra_info"))

    question = env_kwargs.get("question", item.get("question", ""))
    data_source = env_kwargs.get("data_source", item.get("data_source", "unknown"))
    ground_truth = env_kwargs.get("ground_truth")
    if ground_truth is None:
        reward_model = item.get("reward_model")
        if isinstance(reward_model, dict):
            ground_truth = reward_model.get("ground_truth")
    if ground_truth is None:
        ground_truth = item.get("golden_answers", [])

    env_kwargs.update(
        {
            "ground_truth": ground_truth,
            "question": question,
            "data_source": data_source,
            "task_name": "search",
        }
    )
    extra_info.update({"split": split, "index": idx, "task_name": "search"})

    item["data_source"] = data_source
    item["extra_info"] = extra_info
    item["env_kwargs"] = env_kwargs
    item["task_name"] = "search"
    return item


def _search_dataframe(search_dir: str, split: str, size: int, seed: int) -> pd.DataFrame:
    path = os.path.join(os.path.expanduser(search_dir), f"{split}.parquet")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Search parquet not found: {path}. Run examples.data_preprocess.preprocess_search_r1_dataset first."
        )

    sampled = _sample_dataframe(pd.read_parquet(path), size=size, seed=seed)
    rows = [_process_search_row(row, split=split, idx=idx) for idx, (_, row) in enumerate(sampled.iterrows())]
    return pd.DataFrame(rows)


def _build_split(search_dir: str, split: str, per_task_size: int, seed: int) -> pd.DataFrame:
    frames = [
        _dummy_task_dataframe("alfworld", split, per_task_size),
        _search_dataframe(search_dir, split, per_task_size, seed),
        _dummy_task_dataframe("webshop", split, per_task_size),
    ]
    df = pd.concat(frames, ignore_index=True)
    counts = df["task_name"].value_counts().to_dict()
    expected = {task: per_task_size for task in TASKS}
    if counts != expected:
        raise RuntimeError(f"Unexpected multitask counts for {split}: got {counts}, expected {expected}")
    return df


def main():
    local_dir = os.path.expanduser(args.local_dir)
    os.makedirs(local_dir, exist_ok=True)

    train_per_task_size = args.total_training_steps * args.per_task_batch_size
    split_sizes = {"train": train_per_task_size, "test": args.val_per_task_size}

    for split, per_task_size in split_sizes.items():
        df = _build_split(args.search_dir, split, per_task_size, seed=args.seed)
        output_path = os.path.join(local_dir, f"{split}.parquet")
        df.to_parquet(output_path, index=False)
        logger.info("Saved %s rows to %s with task counts %s", len(df), output_path, df["task_name"].value_counts().to_dict())

        search_env_kwargs = df[df["task_name"] == "search"]["env_kwargs"].iloc[0]
        missing = {"ground_truth", "question", "data_source"} - set(search_env_kwargs)
        if missing:
            raise RuntimeError(f"Search env_kwargs is missing keys: {sorted(missing)}")

    if args.hdfs_dir:
        makedirs(args.hdfs_dir)
        copy(src=local_dir, dst=args.hdfs_dir)
        logger.info("Copied multitask data to HDFS: %s", args.hdfs_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build balanced SDAR multitask parquet files.")
    parser.add_argument("--search_dir", default="~/data/searchR1_processed_direct")
    parser.add_argument("--local_dir", default="~/data/verl-agent/sdar_multitask")
    parser.add_argument("--hdfs_dir", default=None)
    parser.add_argument("--total_training_steps", default=150, type=int)
    parser.add_argument("--per_task_batch_size", default=16, type=int)
    parser.add_argument("--val_per_task_size", default=128, type=int)
    parser.add_argument("--seed", default=1, type=int)
    args = parser.parse_args()

    main()
