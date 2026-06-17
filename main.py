import argparse

import pandas as pd
from sklearn.model_selection import train_test_split

from trainer import MDDTrainer


def build_args():
    parser = argparse.ArgumentParser(
        description="Structured XLSR-based MDD training pipeline"
    )

    # =========================
    # Data
    # =========================
    parser.add_argument(
        '--confusion_csv',
        type=str,
        default='/kaggle/working/phone_confusion_matrix.csv'
    )
    
    parser.add_argument(
        '--train_phones_csv',
        type=str,
        default='/kaggle/input/datasets/dungdao111/mdd-hust-st/MDD-Challenge-2025-training-set/metadata/train_phones.csv'
    )

    parser.add_argument(
        '--train_text_csv',
        type=str,
        default='/kaggle/input/datasets/dungdao111/mdd-hust-st/MDD-Challenge-2025-training-set/metadata/train.csv'
    )

    parser.add_argument(
        '--train_wav_dir',
        type=str,
        default='/kaggle/input/datasets/dungdao111/mdd-hust-st/MDD-Challenge-2025-training-set'
    )

    parser.add_argument(
        '--dev_split',
        type=float,
        default=0.1
    )

    parser.add_argument(
        '--random_seed',
        type=int,
        default=42
    )

    # =========================
    # Model
    # =========================
    # Thêm vào build_args(), cùng nhóm Model:
    parser.add_argument(
        '--patience',
        type=int,
        default=8
    )

    parser.add_argument(
        '--num_freeze_layers',
        type=int,
        default=4
    )

    parser.add_argument(
        '--focal_weight',
        type=float,
        default=0.9
    )

    parser.add_argument(
        '--unfreeze_epoch',
        type=int,
        default=2
    )
    
    parser.add_argument(
        '--vocab_path',
        type=str,
        default='vocab.json'
    )

    parser.add_argument(
        '--checkpoint_dir',
        type=str,
        default='./checkpoint'
    )

    parser.add_argument(
        '--pretrained_model',
        type=str,
        default='nguyenvulebinh/wav2vec2-base-vietnamese-250h'
    )

    parser.add_argument(
        '--num_epoch',
        type=int,
        default=20
    )

    parser.add_argument(
        '--eval_start_epoch',
        type=int,
        default=0
    )

    parser.add_argument(
        '--batch_size',
        type=int,
        default=8
    )

    parser.add_argument(
        '--learning_rate',
        type=float,
        default=1e-5
    )

    # =========================
    # Mode
    # =========================
    parser.add_argument(
        '--infer',
        action='store_true',
        help='Run inference instead of training'
    )

    # =========================
    # Inference settings
    # =========================
    parser.add_argument(
        '--infer_csv',
        type=str,
        default='/kaggle/input/datasets/dungdao111/mdd-private/MDD-Challenge-2025-private-test/metadata/private_test_submission.csv'
    )

    parser.add_argument(
        '--infer_wav_dir',
        type=str,
        default='/kaggle/input/datasets/dungdao111/mdd-private/MDD-Challenge-2025-private-test'
    )

    parser.add_argument(
        '--infer_checkpoint',
        type=str,
        default='/kaggle/working/MDD-Framework-Public/checkpoint/checkpoint_wl.pth'
    )

    parser.add_argument(
        '--output_csv',
        type=str,
        default='/kaggle/working/results.csv'
    )

    return parser.parse_args()


# SAU — merge trước, split một lần, tách ra sau — đảm bảo alignment tuyệt đối
import logging
logger = logging.getLogger(__name__)

def split_data(phones_csv, text_csv, dev_split=0.1, random_seed=42):
    phones_df = pd.read_csv(phones_csv)
    text_df   = pd.read_csv(text_csv)

    text_df = text_df.copy()
    text_df['has_error'] = (
        text_df['canonical'].astype(str).str.strip()
        != text_df['transcript'].astype(str).str.strip()
    ).astype(int)

    # Merge phones + text vào một DataFrame duy nhất trước khi split
    # → đảm bảo sau khi split, từng dòng phones luôn khớp đúng dòng text
    if 'id' in phones_df.columns and 'id' in text_df.columns:
        merged = phones_df.merge(
            text_df.add_suffix('_text'),         # tránh trùng tên cột
            left_on='id',
            right_on='id_text',
            how='left'
        )
    else:
        # Fallback: align theo vị trí — reset index để chắc chắn
        phones_df = phones_df.reset_index(drop=True)
        text_df   = text_df.reset_index(drop=True)
        merged    = pd.concat([phones_df, text_df.add_suffix('_text')], axis=1)

    merged['has_error'] = merged['has_error_text'].fillna(0).astype(int)

    # Split một lần duy nhất trên merged
    train_merged, dev_merged = train_test_split(
        merged,
        test_size=dev_split,
        random_state=random_seed,
        stratify=merged['has_error']
    )

    # Tách lại phones và text từ merged sau khi split — alignment được đảm bảo
    phone_cols = phones_df.columns.tolist()
    text_cols  = [c for c in merged.columns if c.endswith('_text')]

    train_phones = train_merged[phone_cols].reset_index(drop=True)
    dev_phones   = dev_merged[phone_cols].reset_index(drop=True)

    # Đổi tên cột _text về tên gốc
    train_text = (train_merged[text_cols]
                  .rename(columns=lambda c: c.removesuffix('_text'))
                  .reset_index(drop=True))
    dev_text   = (dev_merged[text_cols]
                  .rename(columns=lambda c: c.removesuffix('_text'))
                  .reset_index(drop=True))

    error_rate = merged['has_error'].mean()
    logger.info(
        f"split_data: total={len(merged):,} "
        f"train={len(train_phones):,} "
        f"dev={len(dev_phones):,} "
        f"error rate={error_rate:.2%}"
    )

    return train_phones, dev_phones, train_text, dev_text


def build_trainer(args):
    train_phones, dev_phones, train_text, dev_text = split_data(
        phones_csv=args.train_phones_csv,
        text_csv=args.train_text_csv,
        dev_split=args.dev_split,
        random_seed=args.random_seed,
    )

    trainer = MDDTrainer(
        args,
        train_phones,
        dev_phones,
        train_text,
        dev_text
    )

    return trainer


def train(args):
    trainer = build_trainer(args)
    trainer.train()


def inference(args):
    trainer = build_trainer(args)

    output_path = trainer.inference(
        csv_path=args.infer_csv,
        wav_dir=args.infer_wav_dir,
        batch_size=args.batch_size,
        checkpoint=args.infer_checkpoint,
        out_path=args.output_csv,
    )

    print(f"\nInference completed.")
    print(f"Results saved to: {output_path}")


if __name__ == '__main__':
    args = build_args()

    if args.infer:
        inference(args)
    else:
        train(args)