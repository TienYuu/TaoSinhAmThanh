import os
import sys  # Thêm thư viện sys để cấu hình đường dẫn tuyệt đối
import librosa
import pandas as pd
import torch
from transformers import AutoConfig
import torch.nn as nn
import torch.nn.functional as F
from jiwer import wer
from transformers import Wav2Vec2Model
from torch.utils.data import (
    DataLoader,
    WeightedRandomSampler
)
from tqdm import tqdm
# Thêm vào đầu trainer.py nếu chưa có
from evaluate import compute_score

# THÊM VÀO — sau các import ở đầu trainer.py
import logging
logger = logging.getLogger(__name__)

# Ép buộc Python tìm kiếm module trong thư mục Framework trước khi import
EVAL_DIR = '/kaggle/working/MDD-Framework-Public'
if EVAL_DIR not in sys.path:
    sys.path.insert(0, EVAL_DIR)


from dataset import MDDDataset, make_collate_fn
from model import PL
from utils import (
    BLANK_TOKEN_ID,
    SAMPLE_RATE,
    PAD_TOKEN_ID,
    build_feature_extractor,
    get_device,
    load_vocab,
    text_to_tensor,
)

def build_phone_weights(vocab: dict, confusion_csv: str, device) -> torch.Tensor:
    df = pd.read_csv(confusion_csv)
    weights = torch.ones(len(vocab))  # ✅ 125, không +1

    total_errors = df['Số lần'].sum()
    # SAU
    max_count = df['Số lần'].max()  # tính một lần ngoài vòng lặp
    
    for _, row in df.iterrows():
        canonical_ph = row['Gốc (Canonical)']
        wrong_ph     = row['Sai (Transcript)']
        count        = row['Số lần']
    
        if canonical_ph in vocab:
            w = 1.0 + 1.5 * (count / max_count)
            weights[vocab[canonical_ph]] = max(weights[vocab[canonical_ph]], w)
    
        if wrong_ph in vocab:
            w = 1.0 + 1.2 * (count / max_count)
            weights[vocab[wrong_ph]] = max(weights[vocab[wrong_ph]], w)

    # n=49, l=46 — upweight mạnh vì chiếm 31% lỗi
    for ph in ['n', 'l']:
        if ph in vocab:
            weights[vocab[ph]] = max(weights[vocab[ph]], 2.0)

    return weights.to(device)
    
class MDDTrainer:
    def __init__(
        self,
        args,
        train_phones: pd.DataFrame,
        dev_phones: pd.DataFrame,
        train_text: pd.DataFrame,
        dev_text: pd.DataFrame,
    ):
        self.args = args
        self.device = get_device()
        logger.info(f"Training device: {self.device}")

        self.feature_extractor = build_feature_extractor()
        self.vocab = load_vocab(args.vocab_path)

        self.df_train = train_phones
        self.df_dev   = dev_phones

        self.train_dataset = MDDDataset(
            self.df_train,
            wav_dir=args.train_wav_dir,
            vocab=self.vocab,
            text_df=train_text,
        )
        
        self.train_loader = DataLoader(
            dataset=self.train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=make_collate_fn(
                self.feature_extractor,
                self.device
            ),
        )
        

        self.dev_dataset = MDDDataset(
            self.df_dev,
            wav_dir=args.train_wav_dir,
            vocab=self.vocab,
            text_df=dev_text,
        )
        self.dev_loader = DataLoader(
            dataset=self.dev_dataset,
            batch_size=1,
            shuffle=False,
            collate_fn=make_collate_fn(self.feature_extractor, self.device),
        )

        vocab_size = len(self.vocab)
        config = AutoConfig.from_pretrained(args.pretrained_model)
        
        self.model = PL(config=config, vocab_size=vocab_size)
        
        wav2vec_state = Wav2Vec2Model.from_pretrained(args.pretrained_model).state_dict()
        self.model.wav2vec2.load_state_dict(wav2vec_state, strict=False)
        # SAU — truyền từ args, mặc định 4 nếu không có
        self.model.freeze_feature_extractor(
            num_freeze_layers=getattr(self.args, 'num_freeze_layers', 4)
        )
        for p in self.model.wav2vec2.parameters():
            p.requires_grad = False
        self.model = self.model.to(self.device) 
        
        self.id2token = {idx: tok for tok, idx in self.vocab.items()}
        self._build_optimizer()
        self.scaler = torch.amp.GradScaler(device='cuda')
        
        self.best_score = 0.0

        self.patience        = getattr(args, 'patience', 5)
        self.no_improve_cnt  = 0
    
        self.phone_weights = build_phone_weights(
            self.vocab,
            args.confusion_csv,   # thêm field này vào args
            self.device
        )
        os.makedirs(args.checkpoint_dir, exist_ok=True)

    def _build_optimizer(self):

        wav2vec_params = []
        other_params = []
    
        for name, p in self.model.named_parameters():
    
            if not p.requires_grad:
                continue
    
            if name.startswith("wav2vec2"):
                wav2vec_params.append(p)
            else:
                other_params.append(p)
    
        self.optimizer = torch.optim.AdamW(
            [
                {
                    "params": wav2vec_params,
                    "lr": 1e-5,
                },
                {
                    "params": other_params,
                    "lr": 1e-4,
                }
            ],
            weight_decay=0.01
        )
    

    def _focal_ctc_loss(self, logits_T_B_C, labels, input_lengths, target_lengths, gamma=1.0):
        per_sample = F.ctc_loss(
            logits_T_B_C, labels, input_lengths, target_lengths,
            blank=BLANK_TOKEN_ID, zero_infinity=True, reduction='none'
        )
        pt = torch.exp(-per_sample.clamp(max=20))
        focal_w = (1 - pt) ** gamma
        return (focal_w * per_sample).mean()

    def _weighted_ctc_loss(
        self,
        logits_T_B_C,
        labels,
        input_lengths,
        target_lengths
    ):  
        base_loss = F.ctc_loss(
            logits_T_B_C,
            labels,
            input_lengths,
            target_lengths,
            blank=BLANK_TOKEN_ID,
            zero_infinity=True,
            reduction='none'
        )
    
        sample_weights = []
        start = 0
    
        for tlen in target_lengths:
            tlen = int(tlen.item())
            if tlen == 0:
                # Tránh lỗi chuỗi rỗng: Gán trọng số mặc định là 1.0
                sample_weights.append(torch.tensor(1.0, device=self.device))
                continue

            target = labels[start : start + tlen]
            curr = self.phone_weights[target]

            # ✅ Khắc phục RuntimeError bằn cách kiểm tra numel() trước khi log debug
            if curr.numel() > 0:
                w = curr.mean()
            else:
                w = torch.tensor(1.0, device=self.device)
    
            sample_weights.append(w)
            start += tlen
    
        sample_weights = torch.stack(sample_weights)
        return (base_loss * sample_weights).mean()

    # SAU
    def _compute_loss(self, logits_T_B_C_log, labels, input_lengths, target_lengths):
        ctc = self._focal_ctc_loss(
            logits_T_B_C_log, labels, input_lengths, target_lengths
        )
        w_ctc = self._weighted_ctc_loss(
            logits_T_B_C_log, labels, input_lengths, target_lengths
        )
        focal_w = getattr(self.args, 'focal_weight', 0.9)  # mặc định 0.9 nếu không có
        return focal_w * ctc + (1.0 - focal_w) * w_ctc
        
    def train(self):
        logger.info(f"Starting Training Pipeline for {self.args.num_epoch} epochs...")
        for epoch in range(self.args.num_epoch):
            self._train_one_epoch(epoch)
            if self.no_improve_cnt >= self.patience:
                logger.warning(f"Early Stopping tại epoch {epoch} — không cải thiện sau {self.patience} epochs.") 
                break

    def _train_one_epoch(self, epoch):
        self.model.train().to(self.device)
        # SAU
        if epoch == getattr(self.args, 'unfreeze_epoch', 2):
            logger.info(f"Starting Training Pipeline for {self.args.num_epoch} epochs...")
            for p in self.model.wav2vec2.parameters():
                p.requires_grad = True
            self._build_optimizer()
        
        running_loss = []
        
        batch_pbar = tqdm(
            enumerate(self.train_loader), 
            total=len(self.train_loader),
            desc=f"Epoch {epoch}/{self.args.num_epoch - 1}", 
            leave=True, 
            bar_format="{l_bar}{bar:8}{r_bar}"
        )
        
        for batch_idx, data in batch_pbar:
            input_values, linguistic, labels, target_lengths, wav_lengths = data
            
            with torch.amp.autocast(device_type='cuda'):
                logits = self.model(input_values, linguistic)
                input_lengths = self.model._get_feat_extract_output_lengths(wav_lengths)
                
                if (input_lengths < target_lengths).any():
                    batch_pbar.write(f"[WARN] Invalid CTC lengths at batch {batch_idx}")
                    continue
                
                logits_T = logits.transpose(0, 1)       # [T, B, C]
                logits_T = F.log_softmax(logits_T, dim=2)
                loss = self._compute_loss(logits_T, labels, input_lengths, target_lengths)

            if batch_idx == 0:
                for i, group in enumerate(
                    self.optimizer.param_groups
                ):
                    batch_pbar.write(
                        f"LR group {i}: {group['lr']}"
                    )
                batch_pbar.write(f"--- Debug Epoch {epoch} ---")
                batch_pbar.write(f"input_values nan: {torch.isnan(input_values).any()} | min/max: {input_values.min():.3f} / {input_values.max():.3f}")
                batch_pbar.write(f"logits nan: {torch.isnan(logits).any()}")
               
            if not torch.isfinite(loss):
                batch_pbar.write(f"[WARN] Invalid loss at batch {batch_idx}")
                self.optimizer.zero_grad(set_to_none=True)
                continue
            
            self.optimizer.zero_grad(set_to_none=True)
            self.scaler.scale(loss).backward()
            
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.scaler.step(self.optimizer)
            self.scaler.update()
            running_loss.append(loss.item())
            batch_pbar.set_postfix({'loss': f"{loss.item():.4f}"}, refresh=False)
            
        batch_pbar.write(f"--- Đã chạy hết các batch của Epoch {epoch} -> Đang tiến hành cập nhật chỉ số... ---")
        
        avg_train_loss = sum(running_loss) / len(running_loss) if running_loss else 0.0
        
        status_dict = {
            'Loss': f"{avg_train_loss:.4f}", 
            'V_L': 'N/A', 
            'F1': 'N/A',
            'PER': 'N/A',
            'DER': 'N/A'
        }
        
        if epoch >= self.args.eval_start_epoch:
            logger.info(f"[Epoch {epoch}] Đang chạy evaluate...")
            
            try:
                epoch_wer, avg_val_loss, epoch_score, epoch_f1, epoch_per, epoch_der = self._evaluate()
                
                status_dict['V_L'] = f"{avg_val_loss:.4f}"
                status_dict['F1'] = f"{epoch_f1:.4f}"
                status_dict['PER'] = f"{epoch_per:.4f}"
                status_dict['DER'] = f"{epoch_der:.4f}"
                
                if epoch_score > self.best_score:
                    self.best_score  = epoch_score
                    self.no_improve_cnt = 0          # ✅ reset đếm
                    ckpt_path = os.path.join(self.args.checkpoint_dir, "checkpoint_wl.pth")
                    torch.save(self.model.state_dict(), ckpt_path)
                    logger.info(f"Checkpoint mới: Score={epoch_score:.4f} (F1={epoch_f1:.4f}, DER={epoch_der:.4f})")
                else:
                    self.no_improve_cnt += 1         # ✅ tăng đếm
                    logger.warning(f"Không cải thiện ({self.no_improve_cnt}/{self.patience})")
                    
            except Exception as e:
                import traceback
                print(f"\n[NGUY HIỂM - LỖI EVALUATE TẠI EPOCH {epoch}]:")
                print(traceback.format_exc())
        
        batch_pbar.set_postfix(status_dict, refresh=True)
        batch_pbar.close()
        
        logger.info(f"KẾT THÚC EPOCH {epoch} -> Loss: {status_dict['Loss']} | V_Loss: {status_dict['V_L']} | F1: {status_dict['F1']} | PER: {status_dict['PER']} | DER: {status_dict['DER']}")
        
        return running_loss
    
    def _evaluate(self):
        self.model.eval().to(self.device)
        worderrorrate = []
        val_losses    = []
        gt_rows       = []
        predict_rows  = []
        sample_idx    = 0  # ✅ biến đếm riêng, tránh tính sai global_idx
    
        with torch.no_grad():
            for batch_idx, data in enumerate(self.dev_loader):
                input_values, linguistic, labels, target_lengths, wav_lengths = data
                input_lengths = self.model._get_feat_extract_output_lengths(wav_lengths)
    
                with torch.amp.autocast(device_type='cuda'):
                    logits = self.model(input_values, linguistic)
                    logits_loss = logits.transpose(0, 1)
                    logits_loss = F.log_softmax(logits_loss, dim=2)
                    v_loss = self._compute_loss(
                        logits_loss, labels, input_lengths, target_lengths
                    )
    
                val_losses.append(v_loss.item())
                logits = F.log_softmax(logits, dim=2)
    
                for b in range(logits.shape[0]):
                    if sample_idx >= len(self.df_dev):
                        break
    
                    valid_len     = input_lengths[b].item()
                    sample_logits = logits[b, :valid_len, :]
                    hypothesis    = self._greedy_decode_tokens(sample_logits)
    
                    ref_str        = self._get_column_value(self.df_dev, sample_idx, ['Canonical', 'canonical'])
                    transcript_raw = self._get_column_value(self.df_dev, sample_idx, ['Transcript', 'transcript'])
                    transcript     = ' '.join(t for t in transcript_raw.split() if t and t != '$')
    
                    worderrorrate.append(wer(transcript, hypothesis))
                    gt_rows.append({'canonical': ref_str, 'transcript': transcript})
                    predict_rows.append({'predict': hypothesis})
    
                    sample_idx += 1
    
        # ✅ Ghi file tạm → gọi compute_score → xóa file, đảm bảo xóa kể cả khi lỗi
        gt_temp_path   = "val_gt_temp.csv"
        pred_temp_path = "val_predict_temp.csv"
    
        pd.DataFrame(gt_rows).to_csv(gt_temp_path, index=False)
        pd.DataFrame(predict_rows).to_csv(pred_temp_path, index=False)
    
        try:
            # ✅ Lấy thẳng epoch_score từ compute_score, không tính lại
            epoch_score, epoch_f1, epoch_der, epoch_per = compute_score(
                gt_temp_path, pred_temp_path
            )
        finally:
            if os.path.exists(gt_temp_path):   os.remove(gt_temp_path)
            if os.path.exists(pred_temp_path): os.remove(pred_temp_path)
    
        epoch_wer    = sum(worderrorrate) / len(worderrorrate) if worderrorrate else 0.0
        avg_val_loss = sum(val_losses)    / len(val_losses)    if val_losses    else 0.0
    
        return epoch_wer, avg_val_loss, epoch_score, epoch_f1, epoch_per, epoch_der

    def _greedy_decode_tokens(self, frame_logits: torch.Tensor) -> str:
        pred_ids = torch.argmax(frame_logits, dim=-1).tolist()
        collapsed_ids = []
        prev = None
        for token_id in pred_ids:
            if token_id != prev:
                collapsed_ids.append(token_id)
            prev = token_id
    
        SKIP_IDS = {PAD_TOKEN_ID, BLANK_TOKEN_ID}
    
        tokens = []
        for token_id in collapsed_ids:
            if token_id in SKIP_IDS:
                continue
            token = self.id2token.get(token_id, '')
            if token and token != '$' and token != '<unk>':
                tokens.append(token)
        return ' '.join(tokens)

    @staticmethod
    def _get_column_value(dataframe, index: int, names, default=''):
        for name in names:
            if name in dataframe.columns:
                return dataframe[name][index]
        return default

    def inference(
        self,
        csv_path: str,
        wav_dir: str = None,
        out_path: str = 'results.csv',
        batch_size: int = None,
        checkpoint: str = "checkpoint/checkpoint_wl.pth",
        include_path: bool = True,
    ):
        state_dict = torch.load(checkpoint, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.eval().to(self.device)
    
        df = pd.read_csv(csv_path)
    
        paths = df['path'].tolist() if 'path' in df.columns else df['Path'].tolist()
        canonicals_all = (
            df['canonical'].tolist()
            if 'canonical' in df.columns
            else df['Canonical'].tolist()
        )
    
        wav_dir = wav_dir or self.args.train_wav_dir
        batch_size = batch_size or max(
            1, getattr(self.args, 'batch_size', 1) * 2
        )
    
        results = []
    
        inf_pbar = tqdm(
            range(0, len(paths), batch_size),
            desc="Inference",
            bar_format="{l_bar}{bar:20}{r_bar}"
        )
    
        with torch.no_grad():
            for i in inf_pbar:
                batch_paths = paths[i:i + batch_size]
    
                waveforms = []
                wav_lengths = []
    
                for p in batch_paths:
                    raw = str(p)
    
                    if raw.lower().endswith('.wav'):
                        wav_path = (
                            raw if os.path.isabs(raw)
                            else os.path.join(wav_dir, raw)
                        )
                    else:
                        wav_path = os.path.join(wav_dir, f"{raw}.wav")
    
                    audio, _ = librosa.load(wav_path, sr=SAMPLE_RATE)
    
                    waveforms.append(audio)
                    wav_lengths.append(len(audio))
    
                inputs = self.feature_extractor(
                    waveforms,
                    sampling_rate=SAMPLE_RATE,
                    padding=True,
                    return_tensors='pt'
                )
    
                input_values = inputs.input_values.to(self.device)
    
                wav_lengths = torch.tensor(
                    wav_lengths,
                    dtype=torch.long,
                    device=self.device
                )
    
                batch_can = canonicals_all[i:i + batch_size]
    
                token_lists = [
                    text_to_tensor(t, self.vocab)
                    for t in batch_can
                ]
    
                max_can_len = max(
                    (len(t) for t in token_lists),
                    default=1
                )
    
                canonical = torch.full(
                    (input_values.shape[0], max_can_len),
                    fill_value=PAD_TOKEN_ID,
                    dtype=torch.long,
                    device=self.device
                )
    
                for j, toks in enumerate(token_lists):
                    if toks:
                        canonical[j, :len(toks)] = torch.tensor(
                            toks,
                            dtype=torch.long,
                            device=self.device
                        )
    
                logits = self.model(input_values, canonical)
                logits = F.log_softmax(logits, dim=2)
    
                input_lengths = self.model._get_feat_extract_output_lengths(
                    wav_lengths
                )
    
                for b in range(logits.shape[0]):
                    valid_len = int(input_lengths[b].item())
    
                    sample_logits = logits[b, :valid_len, :]
    
                    prediction = self._greedy_decode_tokens(sample_logits)
    
                    results.append(prediction)
    
        # Trích xuất id từ đường dẫn path (ví dụ: path/to/abc.wav -> abc)
        ids = [os.path.splitext(os.path.basename(str(p)))[0] for p in paths]
    
        # Tạo dataframe chứa đầy đủ các cột theo format yêu cầu
        output_df = pd.DataFrame({
            'id': ids,
            'path': paths,
            'predict': results
        })
    
        # Nếu không yêu cầu include_path, loại bỏ cột path
        if not include_path:
            output_df = output_df.drop(columns=['path'])
    
        output_df.to_csv(out_path, index=False)
        print(f"Saved {len(output_df)} predictions -> {out_path}")
    
        return out_path