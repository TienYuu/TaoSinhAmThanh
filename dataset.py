

import os
from typing import Dict, Optional

import librosa
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from utils import (
    PAD_TOKEN_ID,
    SAMPLE_RATE,
    text_to_tensor,
)

# THÊM VÀO — sau các import ở đầu dataset.py
import logging
logger = logging.getLogger(__name__)

CONFUSION_PAIRS = [
    ("s", "x"),  # s→x: lỗi phổ biến nhất
    ("r", "d"),  # r→d
    ("n", "l"),  # n→l
    ("l", "n"),  # l→n (chiều ngược)
]
AUGMENT_PROB = 0.3   # 30% chance augment một utterance (chỉ áp dụng cho sample không có lỗi thật)


class MDDDataset(Dataset):
    def __init__(
        self,
        data: pd.DataFrame,
        wav_dir: str,
        vocab: Dict[str, int],
        text_df: pd.DataFrame = None,
    ):
        self.data      = data
        self.len_data  = len(data)
        self.path      = self._get_column(data, ['Path', 'path'])
        self.audio_path= self._get_column(data, ['AudioPath', 'audio_path'])
        self.canonical = self._get_column(data, ['Canonical', 'canonical'])
        self.transcript= self._get_column(data, ['Transcript', 'transcript'])
        self.wav_dir   = wav_dir
        self.vocab     = vocab
    
        # ✅ Khởi tạo rng và confusion_map sớm, trước khi dùng
        self._rng = np.random.default_rng(seed=42)
        self._confusion_map = {
            self.vocab[src]: self.vocab[dst]
            for src, dst in CONFUSION_PAIRS
            if src in self.vocab and dst in self.vocab
        }
    
        self.error_ids: set = set()
        if text_df is not None:
            self.error_ids = self._build_error_ids(data, text_df)
    
        logger.info(
            f"MDDDataset: {len(data):,} samples | "
            f"real errors: {len(self.error_ids):,} "
            f"({len(self.error_ids)/max(len(data),1):.1%})"
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _build_error_ids(phones_df: pd.DataFrame, text_df: pd.DataFrame) -> set:
        """Return a set of row-indices (in phones_df) that have real errors."""
        has_error = (
            text_df['canonical'].astype(str).str.strip()
            != text_df['transcript'].astype(str).str.strip()
        )
        error_ids: set = set()

        if 'id' in phones_df.columns and 'id' in text_df.columns:
            # Build a lookup: id → has_error
            error_lookup = dict(zip(text_df['id'].astype(str), has_error))
            for row_idx, sample_id in enumerate(phones_df['id'].astype(str)):
                if error_lookup.get(sample_id, False):
                    error_ids.add(row_idx)
        else:
            # Align by position (works when CSVs share the same row order)
            n = min(len(phones_df), len(text_df))
            for row_idx in range(n):
                if has_error.iloc[row_idx]:
                    error_ids.add(row_idx)

        return error_ids

    # SAU
    def _augment_transcript(self, transcript: list) -> list:
        if self._rng.random() > AUGMENT_PROB:
            return transcript
    
        result     = transcript.copy()
        candidates = [i for i, tid in enumerate(result) if tid in self._confusion_map]
        if candidates:
            pos         = self._rng.choice(candidates)
            result[pos] = self._confusion_map[result[pos]]
        return result

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------
    def __len__(self):
        return self.len_data

    def __getitem__(self, index):
        wav_path  = self._resolve_wav_path(index)
        waveform, _ = librosa.load(wav_path, sr=SAMPLE_RATE)

        linguistic = text_to_tensor(self.canonical[index], self.vocab)
        transcript = text_to_tensor(self.transcript[index], self.vocab)

        # Augment phoneme confusion ONLY on samples without a real error.
        # Samples that already contain a genuine mispronunciation are kept
        # untouched so the model learns from real error patterns as-is.
        if index not in self.error_ids:
            transcript = self._augment_transcript(transcript)

        return waveform, linguistic, transcript

    # ------------------------------------------------------------------
    # Static helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _get_column(data: pd.DataFrame, names, default=None):
        for name in names:
            if name in data.columns:
                return list(data[name])
        return [default] * len(data)

    def _resolve_wav_path(self, index: int) -> str:
        if self.audio_path[index]:
            return str(self.audio_path[index])
        raw_path = str(self.path[index])
        if raw_path.lower().endswith('.wav'):
            return os.path.join(self.wav_dir, raw_path)
        return os.path.join(self.wav_dir, f"{raw_path}.wav")


# ----------------------------------------------------------------------
# Collate function — unchanged
# ----------------------------------------------------------------------
def make_collate_fn(feature_extractor, device: torch.device):
    def collate_fn(batch):
        with torch.no_grad():
            max_col = [-1] * 3
            for row in batch:
                max_col[0] = max(max_col[0], row[0].shape[0])
                max_col[1] = max(max_col[1], len(row[1]))
                max_col[2] = max(max_col[2], len(row[2]))

            cols = {
                'waveform':      [],
                'linguistic':    [],
                'transcript':    [],
                'outputlengths': [],
                'input_lengths': [],
            }
            # SAU
            for row in batch:
                cols['input_lengths'].append(row[0].shape[0])
                cols['waveform'].append(row[0])
            
                ling = list(row[1])  # copy trước khi pad
                ling.extend([PAD_TOKEN_ID] * (max_col[1] - len(ling)))
                cols['linguistic'].append(ling)
            
                trans = list(row[2])  # copy trước khi pad
                cols['outputlengths'].append(len(trans))
                trans.extend([PAD_TOKEN_ID] * (max_col[2] - len(trans)))
                cols['transcript'].append(trans)

            inputs = feature_extractor(
                cols['waveform'],
                sampling_rate=SAMPLE_RATE,
                padding=True,           # pad tất cả waveform về cùng độ dài
                return_tensors='pt',    # trả về tensor ngay, tránh list-of-array
            )
            input_values = inputs.input_values.to(device)

            cols['linguistic']    = torch.tensor(cols['linguistic'],    dtype=torch.long, device=device)
            cols['transcript']    = torch.tensor(cols['transcript'],    dtype=torch.long, device=device)
            cols['outputlengths'] = torch.tensor(cols['outputlengths'], dtype=torch.long, device=device)
            cols['input_lengths'] = torch.tensor(cols['input_lengths'], dtype=torch.long, device=device)

            return (
                input_values,
                cols['linguistic'],
                cols['transcript'],
                cols['outputlengths'],
                cols['input_lengths'],
            )
    return collate_fn