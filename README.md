# Hướng dẫn cài đặt & chạy notebook (`mobilev3_code/`)

Tài liệu này hướng dẫn cài đặt môi trường để chạy các notebook trong thư mục `mobilev3_code/`.
Tất cả notebook được thiết kế để chạy trên **Google Colab** (có GPU): chúng tự clone repo,
mount Google Drive và cài dependency. Phần cuối có hướng dẫn chạy local nếu cần.

---

## 1. Danh sách notebook

| Notebook | Mục đích | Cần gì thêm |
| -------- | -------- | ----------- |
| `selected_Train_MobileNetV3_Standalone_Colab.ipynb` | Train MobileNetV3 trực tiếp với MagFace loss (không KD) | Dataset |
| `KD_ConvNextV2_to_MobileNetV3_Colab.ipynb` | KD cơ bản: teacher ConvNeXtV2 → student MobileNetV3 | Dataset + **teacher checkpoint** |
| `KD_RKD_ConvNextV2_to_MobileNetV3_Colab.ipynb` | KD + Relational KD | Dataset + teacher checkpoint |
| `KD_RKD_v2_ConvNextV2_to_MobileNetV3_Colab.ipynb` | RKD biến thể v2 | Dataset + teacher checkpoint |
| `KD_RKD_noKDcosine_ConvNextV2_to_MobileNetV3_Colab.ipynb` | RKD bỏ thành phần KD cosine | Dataset + teacher checkpoint |
| `KD_RKD_Projector_ConvNextV2_to_MobileNetV3_Colab.ipynb` | RKD + ProjectionHead (512→1024) | Dataset + teacher checkpoint |
| `KD_CrossModal_ConcatTeacher_to_MobileNetV3_Colab.ipynb` | KD cross-modal từ teacher ghép 2 modal | Dataset + **2 teacher checkpoint** |
| `selected_(1_15_100_200)KD_CrossModal_ConcatTeacher_to_MobileNetV3_Colab.ipynb` | Bản tinh chỉnh trọng số loss của CrossModal | Dataset + 2 teacher checkpoint |
| `EmbeddingSpaceAnalysis_Colab.ipynb` | Phân tích/visualize không gian embedding (UMAP) | Checkpoint đã train |
| `selected_W8A8_Quantization_MobileNetV3_QAIHub_FromColab.ipynb` | Lượng tử hóa W8A8 qua Qualcomm AI Hub | Checkpoint/ONNX + **QAI Hub token** |
| `selected_ver04_W8A16_Quantization_MobileNetV3_QAIHub_FromColab.ipynb` | Lượng tử hóa W8A16 qua Qualcomm AI Hub | Checkpoint/ONNX + QAI Hub token |

---

## 2. Yêu cầu chung (mọi notebook)

1. **Tài khoản Google + Google Colab** (khuyến nghị bật GPU runtime: `Runtime → Change runtime type → GPU`).
2. **Google Drive** chứa dataset, đã được mount vào `/content/drive`.
3. **Dataset Photometric** đặt tại Drive, mặc định:
   ```
   /content/drive/MyDrive/Photometric_DB_Full/
   ```
   Có thể đổi qua biến `DRIVE_DATASET_DIR` trong cell **CONFIGURATION** của từng notebook.

### Dependency được cài tự động trong notebook

Các notebook tự chạy lệnh cài, **không cần cài tay** khi dùng Colab:

| Notebook | Lệnh cài trong notebook |
| -------- | ----------------------- |
| Train / KD / RKD / CrossModal | `pip install -q albumentations==1.3.1 timm tabulate termcolor` |
| Embedding analysis | `pip install -q albumentations==1.3.1 timm tabulate umap-learn` |
| Quantization (W8A8 / W8A16) | `pip install -q 'qai-hub[torch]'` + `pip install -q albumentations==1.3.1 timm tabulate onnxruntime` |

`torch`, `torchvision`, `pandas`, `numpy`, `scikit-learn`, `tensorboard` đã có sẵn trên Colab.

---

## 3. Cấu trúc dataset mong đợi

Notebook đọc các file CSV split trong thư mục dataset:

```
Photometric_DB_Full/
├── train_split.csv      (hoặc train_set.csv / dataset/train_split.csv)
├── gallery_split.csv
├── probe_split.csv
└── data/{id}/{session}/  ... ảnh .npy
```

CSV tối thiểu phải có cột `id`; các notebook MTL/KD dùng thêm các cột task
(`Gender`, `Spectacles`, `Facial_Hair`, `Pose`, `Emotion`). Xem thêm `README.md` ở thư mục gốc.

---

## 4. Quy trình chạy

### 4.1 Notebook training (Standalone / KD / RKD / CrossModal)
Các notebook được lưu trong mục knowledge distillation, với các mục nhỏ thực nghiệm (được mô tả rõ trong chương Thực nghiệm của quyển đồ án kết hợp với các comment trong notebook)
Link tập dữ liệu Photomestereo vào google drive và chỉnh sửa dataset drive và link github lưu trữ source_code nếu muốn

### 4.2 Notebook Quantization (W8A8 / W8A16 — Qualcomm AI Hub)

Hai notebook này submit job lượng tử hóa lên **Qualcomm AI Hub**, nên cần **API token**.

1. Lấy token tại <https://aihub.qualcomm.com> (đăng nhập → Account/Settings → API token).
2. Lưu token vào **Colab Secrets**: panel bên trái `🔑 Secrets` → thêm key tên `QAI_HUB_TOKEN`, dán giá trị token, bật *Notebook access*.
   Notebook đọc bằng:
   ```python
   from google.colab import userdata
   QAI_HUB_TOKEN = userdata.get('QAI_HUB_TOKEN').strip()
   ```
   và tự cấu hình:
   ```python
   !qai-hub configure --api_token {QAI_HUB_TOKEN}
   ```
3. Chỉnh trong cell config:
   - `DRIVE_DATASET_DIR` — dataset (dùng để hiệu chuẩn/calibration + đo accuracy).
   - `OUTPUT_DIR` — nơi lưu model lượng tử hóa, ví dụ
     `/content/drive/MyDrive/experiments/quantized_mobilenetv3/w8a16/ver04`.
   - Đường dẫn model nguồn: ONNX đã export, hoặc `STUDENT_CKPT` (`.pth`) để trace lại.
4. Chạy tuần tự: setup → compile/quantize job → inference job → so sánh accuracy
   Float32 vs lượng tử hóa (đo AUC & Rank-1 local bằng `onnxruntime`).
5. Kết quả: file ONNX/`.bin` đã lượng tử hóa + `qai_hub_model_ids.json` (lưu id các job) trong `OUTPUT_DIR`.

---

## 5. Tóm tắt dependency

| Thư viện | Dùng cho |
| -------- | -------- |
| `torch`, `torchvision` | Model, training, ONNX export |
| `timm` | Backbone `mobilenetv3_large_100`, `convnextv2_tiny` |
| `albumentations==1.3.1` | Augmentation ảnh |
| `pandas`, `numpy`, `scikit-learn` | Đọc CSV, xử lý dữ liệu, metric |
| `tabulate` | Bảng kết quả |
| `tensorboard` | Log training |
| `termcolor` | Log màu |
| `umap-learn` | Visualize embedding (chỉ notebook analysis) |
| `qai-hub[torch]`, `onnxruntime`, `onnxscript` | Lượng tử hóa qua Qualcomm AI Hub |
