import glob
import random
import os
import torch
import torchaudio
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
import numpy as np
import torchaudio.functional

def get_path_iterator(tsv):
    """
    Get the root path and list of file lines from the TSV file.

    Args:
        tsv (str): Path to the TSV file.

    Returns:
        tuple: Root path and list of file lines.
    """
    with open(tsv, "r") as f:
        root = f.readline().rstrip()
        lines = [line.rstrip() for line in f]
    return root, lines

def load_feature(feat_path):
    """
    Load feature from the specified path.

    Args:
        feat_path (str): Path to the feature file.

    Returns:
        np.ndarray: Loaded feature.
    """
    feat = np.load(feat_path, mmap_mode="r")
    return feat


class ASVSppof2021(Dataset):
    def __init__(self, tsv_path, protocol_path, feat_dir, max_len=64600, is_train=True, codec=True):
        """
        Initialize the dataset with paths and parameters.

        Args:
            tsv_path (str): Path to the TSV file containing data paths.
            protocol_path (str): Path to the protocol file containing metadata.
            feat_dir (str): Directory where features are stored.
            max_len (int): Maximum length of audio data in samples.
            is_train (bool): Flag indicating if the dataset is for training.
            codec (bool): Flag indicating if audio codec transformations should be applied.
        """
        super().__init__()
        root, self.lines = get_path_iterator(tsv_path)
        self.feat_dir = Path(feat_dir)
        _, self.sr = torchaudio.load(root + "/" + self.lines[0].split('\t')[0])
        self.max_len = max_len
        self.is_train = is_train
        self.codec = codec
        self.root = Path(root)

        with open(protocol_path) as file:
            meta_infos = file.readlines()
        self.meta_infos = meta_infos
        self.mapping = {
            meta_info.replace('\n', '').split(' ')[1]: meta_info.replace('\n', '').split(' ')[-1]
            for meta_info in meta_infos
        }

        self.formated_lines = ["wav", "mp3", "ogg", "vorbis", "amr-nb", "amb", "flac", "sph", "gsm", "htk"]

    def __len__(self):
        """
        Return the total number of samples in the dataset.
        """
        return len(self.lines)

    def __getitem__(self, index):
        """
        Retrieve an item from the dataset at the specified index.

        Args:
            index (int): Index of the item to retrieve.

        Returns:
            tuple: A tuple containing audio data, Hubert features, and the target label.
        """
        feat_duration = self.max_len // 320
        relative_path = Path(self.lines[index].split('\t')[0])
        feat_path = self.feat_dir / relative_path
        audio_path = self.root / relative_path
        audio, sr = torchaudio.load(str(audio_path))

        if random.random() < 0.5 and self.codec:
            # Apply codec transformation
            format_select = random.choice(self.formated_lines)
            if format_select == "gsm":
                audio = torchaudio.functional.resample(audio, sr, 8000)
                audio = torchaudio.functional.apply_codec(audio, 8000, format_select)
                audio = torchaudio.functional.resample(audio, 8000, sr)
            else:
                audio = torchaudio.functional.apply_codec(audio, sr, format_select)

        avg_hubert_feat = torch.tensor(load_feature(feat_path.with_suffix(".npy")))
        waveform_info = self.mapping[self.lines[index].split('.')[0]]
        target = 1 if waveform_info == 'spoof' else 0

        if avg_hubert_feat.ndim == 3:
            avg_hubert_feat = avg_hubert_feat.permute(2, 1, 0).squeeze(1)  # [T,1,768] -> [1,T,768]
        else:
            avg_hubert_feat = avg_hubert_feat.permute(1, 0)  # [T,768] -> [768, T]

        if self.is_train and audio.shape[1] > self.max_len:
            st = random.randint(0, audio.shape[1] - self.max_len - 1)
            feat_st = st // 320
            ed = st + self.max_len
            if avg_hubert_feat[:, feat_st:feat_st + feat_duration].shape[1] < feat_duration:
                avg_hubert_feat = avg_hubert_feat[:, feat_st:feat_st + feat_duration]
                avg_hubert_feat = torch.nn.functional.pad(avg_hubert_feat, (0, feat_duration - avg_hubert_feat.shape[1]), "constant", 0)
                return audio[:, st:ed], avg_hubert_feat, target
            else:
                return audio[:, st:ed], avg_hubert_feat[:, feat_st:feat_st + feat_duration], target
        
        if self.is_train == False and audio.shape[1] > self.max_len:
            st = 0
            feat_st = 0
            ed = st + self.max_len
            if avg_hubert_feat[:, feat_st:feat_st + feat_duration].shape[1] < feat_duration:
                avg_hubert_feat = avg_hubert_feat[:, feat_st:feat_st + feat_duration]
                avg_hubert_feat = torch.nn.functional.pad(avg_hubert_feat, (0, feat_duration - avg_hubert_feat.shape[1]), "constant", 0)
                return audio[:, st:ed], avg_hubert_feat, target, str(audio_path)
            else:
                return audio[:, st:ed], avg_hubert_feat[:, feat_st:feat_st + feat_duration], target, str(audio_path)

        if self.is_train and audio.shape[1] < self.max_len:
            audio_pad_length = self.max_len - audio.shape[1]
            audio = torch.nn.functional.pad(audio, (0, audio_pad_length), "constant", 0)

        if self.is_train and avg_hubert_feat.shape[1] < feat_duration:
            avg_hubert_feat = torch.nn.functional.pad(avg_hubert_feat, (0, feat_duration - avg_hubert_feat.shape[1]), "constant", 0)

        if self.is_train == False:
            return audio, avg_hubert_feat, target, str(audio_path)

        return audio, avg_hubert_feat, target
    
def pad_sequence(batch):
    """
    Pad a sequence of tensors to have the same length.

    Args:
        batch (list of Tensors): List of tensors to pad.

    Returns:
        Tensor: Padded tensor with shape (batch_size, max_length, feature_dim).
    """
    batch = [item.permute(1, 0) for item in batch]  # Change shape from (feature_dim, length) to (length, feature_dim)
    batch = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=0.0)  # Pad sequences
    return batch.permute(0, 2, 1)  # Change shape back to (batch_size, feature_dim, max_length)

def collate_fn(batch):
    """
    Collate function to combine multiple samples into a batch.

    Args:
        batch (list of tuples): Each tuple contains (wav, feat, target).

    Returns:
        tuple: Batch of wavs, feats, and targets.
    """
    wavs = []
    feats = []
    targets = []
    for wav, feat, target in batch:
        wavs.append(wav)
        feats.append(feat)
        targets.append(target)

    wavs = pad_sequence(wavs)  # Pad wavs to the same length
    feats = pad_sequence(feats).permute(0, 2, 1)  # Pad feats and permute to (batch_size, max_length, feature_dim)
    return wavs, feats, torch.tensor(targets).long()  # Convert targets to tensor

class DataClass:
    def __init__(
        self,
        train_path, 
        val_path, 
        test_path, 
        max_len=64600,
    ) -> None:

        super().__init__()

        self.train_path = train_path
        self.val_path = val_path
        self.test_path = test_path
        self.max_len = max_len

        # Get different datasets
        self.train = ASVSppof2021(
            self.train_path[0], 
            self.train_path[1], 
            self.train_path[2], 
            self.max_len, 
            is_train=True
        )
        self.val = ASVSppof2021(
            self.val_path[0], 
            self.val_path[1], 
            self.val_path[2], 
            self.max_len, 
            is_train=True
        )
        self.test = ASVSppof2021(
            self.test_path[0], 
            self.test_path[1], 
            self.test_path[2],
            is_train=False,
            codec=False
        )
    def __call__(self, mode: str) -> ASVSppof2021:
        """Get dataset for a given mode.

        Args:
        ----
            mode (str): Mode of the dataset.

        Returns:
        -------
            ASVSppof2019: Dataset for the given mode.

        """
        if mode == "train":
            return self.train
        elif mode == "val":
            return self.val
        elif mode == "test":
            return self.test
        else:
            raise ValueError(f"Unknown mode: {mode}.")

class DataModule(LightningDataModule):
    def __init__(self, DataClass_dict, batch_size, num_workers, pin_memory):
        super().__init__()
        self.save_hyperparameters(logger=False)
        DataClass_dict.pop("_target_")
        self.dataset_select = DataClass(**DataClass_dict)

        self.data_train: Dataset = None
        self.data_val: Dataset = None
        self.data_test: Dataset = None

    def setup(self, stage = None) -> None:
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by Lightning before `trainer.fit()`, `trainer.validate()`, `trainer.test()`, and
        `trainer.predict()`, so be careful not to execute things like random split twice! Also, it is called after
        `self.prepare_data()` and there is a barrier in between which ensures that all the processes proceed to
        `self.setup()` once the data is prepared and available for use.

        :param stage: The stage to setup. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`. Defaults to ``None``.
        """
        # load and split datasets only if not loaded already
        if not self.data_train and not self.data_val and not self.data_test:
            self.data_train = self.dataset_select("train")
            self.data_val = self.dataset_select("val")
            self.data_test = self.dataset_select("test")

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
            collate_fn=collate_fn
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.data_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            collate_fn=collate_fn
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.data_test,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )
