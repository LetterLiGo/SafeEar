# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
import sys

import numpy as np
from sklearn.cluster import MiniBatchKMeans

import joblib

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("learn_kmeans")


def load_feature(feat_path):
    feat = np.load(feat_path, mmap_mode="r")
    return feat


def load_feature(feat_dir, split, nshard, seed, percent):
    assert percent <= 1.0
    feat = np.concatenate(
        [
            load_feature_shard(feat_dir, split, nshard, r, percent)
            for r in range(nshard)
        ],
        axis=0,
    )
    logging.info(f"loaded feature with dimension {feat.shape}")
    return feat


def load_data(
    feat_dir,
    split,
    nshard,
    seed,
    percent,
):
    np.random.seed(seed)
    feat = load_feature(feat_dir, split, nshard, seed, percent)
    logger.info("finished successfully")


if __name__ == "__main__":
    # load_feature_shard(
    #     feat_dir="../hubert_base_ls960_avg_feature/",
    #     split='valid',
    #     nshard=1,
    #     rank=0
    # )
    load_feature(feat_dir="../hubert_base_ls960_avg_feature/",
                 split='valid',
                 nshard=1,
                 seed=1,
                 percent=1)
    # import argparse

    # parser = argparse.ArgumentParser()
    # parser.add_argument("feat_dir", type=str)
    # parser.add_argument("split", type=str)
    # parser.add_argument("nshard", type=int)
    # parser.add_argument("km_path", type=str)
    # parser.add_argument("n_clusters", type=int)
    # parser.add_argument("--seed", default=0, type=int)
    # parser.add_argument(
    #     "--percent", default=-1, type=float, help="sample a subset; -1 for all"
    # )
    # parser.add_argument("--init", default="k-means++")
    # parser.add_argument("--max_iter", default=100, type=int)
    # parser.add_argument("--batch_size", default=10000, type=int)
    # parser.add_argument("--tol", default=0.0, type=float)
    # parser.add_argument("--max_no_improvement", default=100, type=int)
    # parser.add_argument("--n_init", default=20, type=int)
    # parser.add_argument("--reassignment_ratio", default=0.0, type=float)
    # args = parser.parse_args()
    # logging.info(str(args))

    # load_data(**vars(args))
