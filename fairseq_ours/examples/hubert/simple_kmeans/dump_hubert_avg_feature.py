# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
import sys

import tqdm
import fairseq
import soundfile as sf
import torch
import torch.nn.functional as F
from npy_append_array import NpyAppendArray
from pathlib import Path
from feature_utils import get_path_iterator, dump_feature


logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("dump_hubert_feature")


class HubertFeatureReader(object):
    def __init__(self, ckpt_path, layer, max_chunk=1600000):
        (
            model,
            cfg,
            task,
        ) = fairseq.checkpoint_utils.load_model_ensemble_and_task([ckpt_path])
        self.model = model[0].eval().cuda()
        self.task = task
        self.layer = layer
        self.max_chunk = max_chunk
        logger.info(f"TASK CONFIG:\n{self.task.cfg}")
        logger.info(f" max_chunk = {self.max_chunk}")

    def read_audio(self, path, ref_len=None):
        wav, sr = sf.read(path)
        assert sr == self.task.cfg.sample_rate, sr
        if wav.ndim == 2:
            wav = wav.mean(-1)
        assert wav.ndim == 1, wav.ndim
        if ref_len is not None and abs(ref_len - len(wav)) > 160:
            logging.warning(f"ref {ref_len} != read {len(wav)} ({path})")
        return wav

    def get_feats(self, path, ref_len=None):
        x = self.read_audio(path, ref_len)
        with torch.no_grad():
            x = torch.from_numpy(x).float().cuda()
            if self.task.cfg.normalize:
                x = F.layer_norm(x, x.shape)
            x = x.view(1, -1)

            avg_feat = []
            for start in range(0, x.size(1), self.max_chunk):
                x_chunk = x[:, start: start + self.max_chunk]
                feat_chunk, _, avg_feat_chunk = self.model.extract_features(
                    source=x_chunk,
                    padding_mask=None,
                    mask=False,
                    output_layer=self.layer,
                )
                avg_feat.append(avg_feat_chunk)
        return torch.cat(avg_feat, 1).squeeze(0)

def dump_feature(reader,audio_dir,save_dir):
    save_dir = Path(save_dir)
    audio_dir = Path(audio_dir)
    
    audio_files = list(audio_dir.glob("**/*.flac"))
    for audio_file in tqdm.tqdm(audio_files):
        releative_path = audio_file.relative_to(audio_dir).with_suffix(".npy")
        save_path = save_dir / releative_path
        import pdb; pdb.set_trace()
        if not save_path.parent.exists():
            save_path.parent.mkdir(parents=True)
        
        feat_f = NpyAppendArray(save_path)
        feat = reader.get_feats(audio_file)
        feat_f.append(feat.cpu().numpy())
    logger.info("finished successfully")

def main(audio_dir, save_dir, ckpt_path, layer, max_chunk):
    reader = HubertFeatureReader(ckpt_path, layer, max_chunk)
    dump_feature(reader, audio_dir, save_dir)
# def dump_feature(reader, generator, num, split, nshard, rank, feat_dir):
#     iterator = generator()

#     feat_path = f"{feat_dir}/{split}_{rank}_{nshard}.npy"
#     leng_path = f"{feat_dir}/{split}_{rank}_{nshard}.len"

#     os.makedirs(feat_dir, exist_ok=True)
#     if os.path.exists(feat_path):
#         os.remove(feat_path)

#     feat_f = NpyAppendArray(feat_path)
#     with open(leng_path, "w") as leng_f:
#         for path, nsample in tqdm.tqdm(iterator, total=num):
#             feat = reader.get_feats(path, nsample)
#             feat_f.append(feat.cpu().numpy())
#             leng_f.write(f"{path}\t{len(feat)}\n")
#     logger.info("finished successfully")

# def main(tsv_dir, split, ckpt_path, layer, nshard, rank, feat_dir, max_chunk):
#     reader = HubertFeatureReader(ckpt_path, layer, max_chunk)
#     generator, num = get_path_iterator(f"{tsv_dir}/{split}.tsv", nshard, rank)
#     dump_feature(reader, generator, num, split, nshard, rank, feat_dir)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    # parser.add_argument("tsv_dir")
    parser.add_argument("audio_dir", default="/home/lxf/Documents/xinfeng/datasets/ASVSpoof2019")
    parser.add_argument("save_dir", default="/home/lxf/Documents/xinfeng/datasets/ASVSpoof2019_Hubert_L9")
    # parser.add_argument("split")
    parser.add_argument("ckpt_path", default="/home/lxf/Documents/xinfeng/model_zoo/hubert")
    parser.add_argument("layer", type=int, default=9)
    # parser.add_argument("nshard", type=int)
    # parser.add_argument("rank", type=int)
    # parser.add_argument("feat_dir")
    parser.add_argument("--max_chunk", type=int, default=1600000)
    args = parser.parse_args()
    logger.info(args)

    main(**vars(args))
