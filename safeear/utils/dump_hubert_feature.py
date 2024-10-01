# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
import sys
import tqdm
sys.path.append('../../fairseq_ours/') # we recommend an abosulte path here.
import fairseq
import soundfile as sf
import torch
import torch.nn.functional as F
from npy_append_array import NpyAppendArray
from pathlib import Path
# from feature_utils import get_path_iterator, dump_feature


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

            feat = []
            for start in range(0, x.size(1), self.max_chunk):
                x_chunk = x[:, start: start + self.max_chunk]
                feat_chunk, _, _ = self.model.extract_features(
                    source=x_chunk,
                    padding_mask=None,
                    mask=False,
                    output_layer=self.layer,
                )
                feat.append(feat_chunk)
        return torch.cat(feat, 1).squeeze(0)

def dump_feature(reader,audio_dir,save_dir):
    save_dir = Path(save_dir)
    audio_dir = Path(audio_dir)
    
    audio_files = list(audio_dir.glob("**/*.flac"))
    
    for audio_file in tqdm.tqdm(audio_files):
        releative_path = audio_file.relative_to(audio_dir).with_suffix(".npy")
        save_path = save_dir / releative_path
        # import pdb; pdb.set_trace()
        if not save_path.parent.exists():
            save_path.parent.mkdir(parents=True)
        
        feat_f = NpyAppendArray(save_path)
        feat = reader.get_feats(audio_file)
        feat_f.append(feat.cpu().numpy())
    logger.info("finished successfully")

def main(audio_dir, save_dir, ckpt_path, layer, max_chunk):
    reader = HubertFeatureReader(ckpt_path, layer, max_chunk)
    dump_feature(reader, audio_dir, save_dir)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--audio_dir", default="./datasets/ASVSpoof2021/ASVspoof2021_LA_eval/flac")
    parser.add_argument("--save_dir", default="./datasets/ASVSpoof2021/ASVspoof2021_LA_eval/Hubert_L9")
   
    parser.add_argument("--ckpt_path", default="./model_zoo/hubert/hubert_base_ls960.pt")
    parser.add_argument("--layer", type=int, default=9)
    parser.add_argument("--max_chunk", type=int, default=1600000)
    args = parser.parse_args()
    logger.info(args)

    main(**vars(args))
