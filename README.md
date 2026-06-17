# MDD-Framework-Public

Pipeline Phát hiện và Chẩn đoán Lỗi Phát Âm Tiếng Việt (**M**ispronunciation **D**etection and **D**iagnosis) cho **MDD Challenge 2025** — môn học *Tạo sinh âm thanh*.

Mô hình kết hợp một backbone âm học tự giám sát (`wav2vec2-base-vietnamese-250h`) với một luồng ngôn ngữ mã hoá chuỗi âm vị chuẩn (canonical), hợp nhất qua cơ chế chú ý chéo (cross-attention) và cổng điều khiển (gated fusion), huấn luyện bằng CTC có trọng số dựa trên phân tích lỗi phát âm thực tế trong dữ liệu.

> Chi tiết phương pháp, phân tích dữ liệu và kết quả thực nghiệm đầy đủ được trình bày trong Technical Report của nhóm (file `.docx`/`.pdf` nộp kèm).

---

## Mục lục

- [Cấu trúc repo](#cấu-trúc-repo)
- [Cài đặt](#cài-đặt)
- [Chuẩn bị dữ liệu](#chuẩn-bị-dữ-liệu)
- [Cách sử dụng](#cách-sử-dụng)
  - [Huấn luyện](#huấn-luyện)
  - [Suy luận / sinh dự đoán](#suy-luận--sinh-dự-đoán)
  - [Danh sách tham số dòng lệnh](#danh-sách-tham-số-dòng-lệnh)
- [Kiến trúc mô hình](#kiến-trúc-mô-hình)
- [Hàm mất mát](#hàm-mất-mát)
- [Chỉ số đánh giá](#chỉ-số-đánh-giá)
- [Kết quả tốt nhất](#kết-quả-tốt-nhất)
- [Mô tả các file mã nguồn](#mô-tả-các-file-mã-nguồn)
- [Hạn chế & hướng phát triển](#hạn-chế--hướng-phát-triển)
- [Thành viên nhóm](#thành-viên-nhóm)

---

## Cấu trúc repo

```
MDD-Framework-Public/
├── main.py              # Entry point: điều phối train / infer qua CLI
├── trainer.py            # Vòng lặp huấn luyện, loss, optimizer, evaluate, checkpoint
├── model.py               # Kiến trúc mô hình hai luồng (PL = Phonetic-Linguistic)
├── dataset.py             # MDDDataset, augmentation, collate_fn
├── evaluate.py            # compute_score(): F1 / PER / DER / Score tổng hợp
├── utils.py                # Hằng số, load_vocab, text_to_tensor, feature extractor, device
├── vocab.json              # Bộ từ vựng âm vị CTC (125 token)
├── requirements.txt        # Danh sách thư viện cần cài
└── README.md
```

## Cài đặt

Yêu cầu Python ≥ 3.10 và GPU có CUDA (khuyến nghị; có thể chạy CPU nhưng rất chậm).

```bash
git clone https://github.com/trungntdsai/MDD-Framework-Public.git
cd MDD-Framework-Public
pip install -r requirements.txt
```

Các thư viện lõi mà mã nguồn sử dụng trực tiếp (đã có trong `requirements.txt`):

| Thư viện | Vai trò |
|---|---|
| `torch`, `transformers` | Mô hình nền `wav2vec2-base-vietnamese-250h`, autocast/AMP |
| `librosa` | Đọc và resample waveform |
| `pandas`, `numpy`, `scikit-learn` | Xử lý CSV, chia train/dev (stratified split) |
| `jiwer` | Tính WER hỗ trợ trong quá trình theo dõi huấn luyện |
| `python-Levenshtein` | Căn chỉnh canonical/transcript khi phân tích lỗi |
| `tqdm` | Progress bar |
| `kenlm`, `pyctcdecode` *(tuỳ chọn)* | Hỗ trợ giải mã CTC beam-search + mô hình ngôn ngữ (chưa được dùng trong bản nộp hiện tại, decode đang ở dạng greedy) |

## Chuẩn bị dữ liệu

Mặc định, các đường dẫn trong `main.py` được cấu hình sẵn cho môi trường Kaggle (`/kaggle/input/...`). Nếu chạy ở môi trường khác (local, Colab,...), hãy ghi đè các tham số đường dẫn khi gọi `main.py` (xem [bảng tham số](#danh-sách-tham-số-dòng-lệnh)).

Dữ liệu cần có cấu trúc tối thiểu sau:

```
<train_wav_dir>/
└── (các file .wav theo cột path/audio_path trong CSV)

metadata/
├── train.csv          # cột: id, path, canonical, transcript   (dạng văn bản đọc được)
└── train_phones.csv   # cột: id, path, canonical, transcript   (dạng chuỗi âm vị IPA, từ phân tách bởi "$")
```

Ngoài ra, `trainer.py` cần một file thống kê lỗi phát âm để tính trọng số loss:

```
phone_confusion_matrix.csv   # cột: "Gốc (Canonical)", "Sai (Transcript)", "Số lần"
```

File này được sinh ra từ bước phân tích lỗi (căn chỉnh Levenshtein giữa canonical/transcript trên `train_phones.csv` — xem Mục 2.2 trong Technical Report). Nếu chưa có sẵn trong repo, cần chạy script/notebook phân tích dữ liệu một lần để tạo file này trước khi train (đường dẫn mặc định: `--confusion_csv`).

Với tập private test (giai đoạn suy luận), cấu trúc tương tự nhưng chỉ cần cột `id`, `path` và (tuỳ chọn) `canonical` — không cần `transcript` vì đây là nhãn ẩn.

## Cách sử dụng

### Huấn luyện

```bash
python main.py
```

Mặc định sẽ: đọc `train.csv` + `train_phones.csv` → chia train/dev theo tỷ lệ 90/10 (stratify theo cờ có lỗi, `random_seed=42`) → huấn luyện tối đa 20 epoch với early-stopping (`patience=8`) → lưu checkpoint tốt nhất vào `./checkpoint/checkpoint_wl.pth`.

Ví dụ tuỳ biến một số tham số:

```bash
python main.py \
  --train_text_csv   /path/to/train.csv \
  --train_phones_csv /path/to/train_phones.csv \
  --train_wav_dir    /path/to/audio_root \
  --confusion_csv    /path/to/phone_confusion_matrix.csv \
  --batch_size 8 \
  --num_epoch 20 \
  --patience 8
```

### Suy luận / sinh dự đoán

```bash
python main.py --infer
```

Mặc định sẽ load checkpoint tại `./checkpoint/checkpoint_wl.pth`, chạy giải mã CTC greedy trên tập được chỉ định bởi `--infer_csv`/`--infer_wav_dir`, và lưu kết quả (cột `id`, `path`, `predict`) vào `--output_csv` (mặc định `results.csv`).

```bash
python main.py --infer \
  --infer_csv        /path/to/private_test_submission.csv \
  --infer_wav_dir    /path/to/private_test_audio \
  --infer_checkpoint ./checkpoint/checkpoint_wl.pth \
  --output_csv       ./results.csv
```

> Lưu ý: lệnh `--infer` hiện vẫn cần `--train_text_csv`/`--train_phones_csv` hợp lệ vì `main.py` build lại `MDDTrainer` (bao gồm vocab và train/dev loader) trước khi gọi `inference()`. Hãy giữ nguyên các tham số dữ liệu train khi chạy infer, chỉ thay đổi các tham số bắt đầu bằng `--infer_*`.

### Đánh giá độc lập (không qua main.py)

Nếu chỉ muốn tính điểm cho một file `results.csv` đã có sẵn so với ground truth:

```bash
python evaluate.py /path/to/ground_truth.csv /path/to/results.csv
```

trong đó `ground_truth.csv` cần có cột `canonical`, `transcript`; `results.csv` cần có cột `predict`.

### Danh sách tham số dòng lệnh

| Tham số | Mặc định | Ý nghĩa |
|---|---|---|
| `--train_text_csv` | `.../metadata/train.csv` | File CSV canonical/transcript dạng văn bản |
| `--train_phones_csv` | `.../metadata/train_phones.csv` | File CSV canonical/transcript dạng âm vị |
| `--train_wav_dir` | `.../MDD-Challenge-2025-training-set` | Thư mục gốc chứa audio train |
| `--confusion_csv` | `/kaggle/working/phone_confusion_matrix.csv` | File thống kê cặp âm vị nhầm lẫn, dùng để tính trọng số loss |
| `--dev_split` | `0.1` | Tỉ lệ tách tập dev từ train |
| `--random_seed` | `42` | Seed cho việc chia train/dev và augmentation |
| `--vocab_path` | `vocab.json` | Đường dẫn bộ từ vựng âm vị (CTC vocab) |
| `--checkpoint_dir` | `./checkpoint` | Thư mục lưu checkpoint khi train |
| `--pretrained_model` | `nguyenvulebinh/wav2vec2-base-vietnamese-250h` | Mô hình nền HuggingFace |
| `--num_freeze_layers` | `4` | Số lớp transformer đầu của wav2vec2 bị đóng băng lúc khởi đầu |
| `--unfreeze_epoch` | `2` | Epoch bắt đầu mở đóng băng toàn bộ backbone |
| `--focal_weight` | `0.9` | Trọng số của Focal CTC trong tổng loss (phần còn lại là Weighted CTC) |
| `--num_epoch` | `20` | Số epoch huấn luyện tối đa |
| `--eval_start_epoch` | `0` | Epoch bắt đầu đánh giá trên dev |
| `--patience` | `8` | Số epoch không cải thiện trước khi early-stop |
| `--batch_size` | `8` | Kích thước batch |
| `--learning_rate` | `1e-5` | Learning rate cơ sở (backbone) |
| `--infer` | `False` | Cờ chuyển sang chế độ suy luận thay vì huấn luyện |
| `--infer_csv` | `.../private_test_submission.csv` | File CSV cần suy luận |
| `--infer_wav_dir` | `.../MDD-Challenge-2025-private-test` | Thư mục audio cần suy luận |
| `--infer_checkpoint` | `.../checkpoint/checkpoint_wl.pth` | Checkpoint dùng để suy luận |
| `--output_csv` | `/kaggle/working/results.csv` | Đường dẫn lưu kết quả dự đoán |

## Kiến trúc mô hình

Mô hình (class `PL` trong `model.py`) gồm hai luồng:

- **Luồng âm học**: `wav2vec2-base-vietnamese-250h` (4 lớp transformer đầu đóng băng ban đầu, mở từ epoch `unfreeze_epoch`) → `PhoneticEncoder` (Conv1D → LayerNorm → ReLU → BiLSTM, residual add).
- **Luồng ngôn ngữ**: chuỗi âm vị canonical → `Embedding` + `Positional Embedding` → `BiLSTM` → chiếu thành Key/Value 768 chiều.
- **Hợp nhất**: Multi-Head Cross-Attention (16 đầu, Query = đặc trưng âm học, Key/Value = đặc trưng ngôn ngữ) → Gated Fusion (`gate = σ(Linear([acoustic, attn_out]))`) → `Linear` Classifier → 125 lớp âm vị → CTC.

Việc đưa canonical vào như một luồng Key/Value riêng (thay vì chỉ dùng làm nhãn) là điểm khác biệt cốt lõi giữa mô hình MDD này với một mô hình ASR thông thường.

## Hàm mất mát

```
L = focal_weight × FocalCTCLoss(γ=1) + (1 − focal_weight) × WeightedCTCLoss
```

`WeightedCTCLoss` dùng trọng số theo từng âm vị, được tính trong `build_phone_weights()` (`trainer.py`) từ `phone_confusion_matrix.csv`; hai âm vị `n` và `l` được ép trọng số tối thiểu 2.0 vì gộp lại chiếm khoảng 31% tổng lỗi thay thế ở cấp âm vị trong dữ liệu huấn luyện.

## Chỉ số đánh giá

Cài đặt chính xác trong `evaluate.py` (`compute_score`):

- **PER** (Phoneme Error Rate): tỉ lệ (chèn + xoá + thay thế) khi căn chỉnh `transcript` (nhãn thật) với `predict` (dự đoán mô hình), chia cho tổng số âm vị tham chiếu.
- **F1**: F1 của bài toán phát hiện lỗi (detection), tính từ True Rejection / False Rejection / False Acceptance theo cấu trúc đánh giá phân cấp ba-chiều-căn-chỉnh (canonical↔transcript, transcript↔predict, canonical↔predict).
- **DER** (**Diagnosis** Error Rate — không phải Diarization Error Rate): trong số các trường hợp đã phát hiện đúng là có lỗi, tỉ lệ mà mô hình chẩn đoán sai âm vị thay thế cụ thể, `DER = Error_Diag / (Correct_Diag + Error_Diag)`.
- **Score** (điểm tổng hợp dùng để chọn checkpoint):

  ```
  Score = 0.5 × F1 + 0.4 × (1 − DER) + 0.1 × (1 − PER)
  ```

## Kết quả tốt nhất

Trên tập dev (318 câu), checkpoint tốt nhất được chọn tại **epoch 17**:

| F1 | PER | DER | Score |
|---|---|---|---|
| 0,1184 | 0,0958 | 0,1333 | 0,4963 |

Checkpoint này được dùng để sinh dự đoán cho toàn bộ 856 câu của tập private test (`results.csv`). Chi tiết đầy đủ (đường cong huấn luyện, phân tích lỗi) xem trong Technical Report.

## Mô tả các file mã nguồn

| File | Nội dung chính |
|---|---|
| `main.py` | Định nghĩa CLI args, `split_data()` (chia train/dev có stratify), điều phối `train()`/`inference()` |
| `trainer.py` | Class `MDDTrainer`: khởi tạo dataset/loader, xây dựng mô hình, vòng lặp huấn luyện (mixed-precision, gradient clipping, freeze/unfreeze theo epoch), `_evaluate()`, lưu checkpoint, `inference()` (greedy decode) |
| `model.py` | Class `PL` (Phonetic–Linguistic model): `PhoneticEncoder`, `LinguisticEncoder`, Cross-Attention, Gated Fusion, Classifier |
| `dataset.py` | Class `MDDDataset` (đọc audio, áp dụng augmentation cặp âm nhầm lẫn chỉ trên câu không lỗi thật), `make_collate_fn()` (padding + feature extraction theo batch) |
| `evaluate.py` | `compute_score()`: căn chỉnh Needleman–Wunsch ba chiều, tính PER/F1/DER/Score |
| `utils.py` | Hằng số (`SAMPLE_RATE`, `PAD_TOKEN_ID`, `BLANK_TOKEN_ID`), `load_vocab()`, `text_to_tensor()`, `build_feature_extractor()`, `get_device()` |

## Hạn chế & hướng phát triển

- Giải mã hiện tại là greedy CTC; chưa tích hợp beam-search + KenLM dù môi trường đã cài `kenlm`/`pyctcdecode`.
- `WeightedRandomSampler` được import trong `trainer.py` nhưng chưa được dùng triệt để để oversample câu lỗi.
- Dữ liệu lỗi phát âm thật còn ít (~22% số câu); phần lớn tín hiệu lỗi mô hình học được đến từ augmentation tổng hợp dựa trên các cặp âm phổ biến nhất (s/x, r/d, n/l).

Xem Mục 5 của Technical Report để biết phân tích lỗi và đề xuất cải thiện chi tiết hơn.

## Thành viên nhóm

| Họ và tên | MSSV |
|---|---|
| Đào Tiến Dũng | 20241038E| 
| Lê Anh Duy | 20240944E | 
| Nguyễn Công Tú| 20241628E | 

Môn học: *Speech Technology* — MDD Challenge 2025.
