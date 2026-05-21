# Kiến trúc MTLFaceRecognition

## 1. Tổng quan

`MTLFaceRecognition` là mạng **Multi-Task Learning (MTL)** nhận diện khuôn mặt từ ảnh Photometric Stereo (Albedo / Normal Map / Depth Map). Mạng đồng thời giải quyết 6 task: nhận diện danh tính và 5 thuộc tính khuôn mặt, sử dụng cơ chế **Feature Disentanglement** và **Domain Adversarial Training** để tách biệt thông tin giữa các task.

---

## 2. Input / Output

### Input

| Thuộc tính | Giá trị |
|------------|---------|
| Shape | `[B, 3, 112, 112]` (float32) |
| Loại ảnh | Albedo map / Normal map / Depth map |
| Định dạng lưu trữ | `.npy` (chuyển từ `.exr` HDR) |
| Chuẩn hóa | Không dùng ImageNet mean/std — giữ nguyên giá trị HDR |

Label đi kèm mỗi sample `y ∈ ℤ^6`:

```
y = [id, gender, spectacles, facial_hair, pose, emotion]
     [0]   [1]      [2]          [3]       [4]     [5]
```

| Index | Task | Nhãn |
|-------|------|------|
| 0 | Identity | 0 → N-1 (N người) |
| 1 | Gender | 0=Nữ, 1=Nam |
| 2 | Spectacles | 0=Không, 1=Có kính |
| 3 | Facial Hair | 0=Không, 1=Có râu |
| 4 | Pose | 0=Thẳng, 1=Lệch |
| 5 | Emotion | 0=Trung tính, 1=Biểu cảm |

### Output (training — `forward`)

```python
logits = (
    (x_spectacles    [B,2],  x_da_spectacles  [B,2]),
    (x_facial_hair   [B,2],  x_da_facial_hair [B,2]),
    (x_pose          [B,2],  x_da_pose        [B,2]),
    (x_emotion       [B,2],  x_da_emotion     [B,2]),
    (x_gender        [B,2],  x_da_gender      [B,2]),
    x_id_logits = ([cos_theta [B,N], cos_theta_m [B,N]]),
    x_id_norm   [B,1]
)
```

| Output | Shape | Ý nghĩa |
|--------|-------|---------|
| `x_<attr>` | `[B,2]` | Logit phân loại thuộc tính (main branch) |
| `x_da_<attr>` | `[B,2]` | Logit adversarial (qua GRL) |
| `cos_theta` | `[B,N]` | Cosine similarity với N class centers |
| `cos_theta_m` | `[B,N]` | Cosine sau adaptive angular margin |
| `x_id_norm` | `[B,1]` | L2 norm của embedding (dùng cho MagFace) |

### Output (inference — `get_embedding`)

```python
x_spectacles [B,512], x_facial_hair [B,512], x_pose [B,512],
x_emotion [B,512], x_gender [B,512], x_id [B,512]
```

---

## 3. Kiến trúc chi tiết

```
┌─────────────────────────────────────────────────────────────────────────┐
│  INPUT  [B, 3, 112, 112]                                                │
└──────────────────────────────┬──────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  BACKBONE: MIConvNeXtV2                                                 │
│                                                                         │
│  timm.ConvNeXtV2 (features_only=True)                                  │
│  → last feature map  [B, C_out, H, W]   (C_out phụ thuộc variant)      │
│                                                                         │
│  Adapter:                                                               │
│    Conv2d(C_out → 512, kernel=1) → BN → PReLU                          │
│  → [B, 512, H, W]                                                       │
│                                                                         │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │  FSM × 5  (Feature Separation Module = AttentionModule)           │  │
│  │                                                                   │  │
│  │  Input x [B,512,H,W]                                              │  │
│  │    │                                                              │  │
│  │    ├─ Channel Attention:                                          │  │
│  │    │    SPP_avg(1,2,3) + SPP_max(1,2,3) → [B, 512×14, 1, 1]     │  │
│  │    │    → Conv1×1 → ReLU → Conv1×1 → BN → Sigmoid               │  │
│  │    │    → channel_scale [B, 512, 1, 1]                           │  │
│  │    │                                                              │  │
│  │    ├─ Spatial Attention:                                          │  │
│  │    │    [max_C(x); mean_C(x)] → [B, 2, H, W]                    │  │
│  │    │    → Conv7×7 → BN → Sigmoid                                 │  │
│  │    │    → spatial_scale [B, 1, H, W]                             │  │
│  │    │                                                              │  │
│  │    x_non_id = 0.5 × (x × channel_scale + x × spatial_scale)     │  │
│  │    x_id     = x − x_non_id                                       │  │
│  │                                                                   │  │
│  └───────────────────────────────────────────────────────────────────┘  │
│                                                                         │
│  Chuỗi disentangle (sequential):                                        │
│                                                                         │
│  x ──[FSM_spec]──────► x_spec    [B,512,H,W]  (spectacles info)        │
│           └──────────► x_non_spec [B,512,H,W]                          │
│                              │                                          │
│  x_non_spec ──[FSM_fh]──────► x_fh     [B,512,H,W]  (facial hair)     │
│                   └──────────► x_non_fh [B,512,H,W]                    │
│                                     │                                   │
│  x_non_fh ──[FSM_emot]─────────────► x_emot     [B,512,H,W]           │
│                   └─────────────────► x_non_emot [B,512,H,W]           │
│                                            │                            │
│  x_non_emot ──[FSM_pose]─────────────────► x_pose     [B,512,H,W]     │
│                    └─────────────────────► x_non_pose [B,512,H,W]      │
│                                                  │                      │
│  x_non_pose ──[FSM_gender]──────────────────────► x_gender [B,512,H,W] │
│                    └────────────────────────────► x_id     [B,512,H,W] │
└─────────────────────────────────────────────────────────────────────────┘
                               │
              ┌────────────────┼────────────────┐
              │                │                │
              ▼                ▼                ▼
     Task Branches        GRL Branches      ID Branch
     (main heads)    (adversarial heads)   (id head)
              │                │                │
              ▼                ▼                ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  HEADS                                                                  │
│                                                                         │
│  x_spec   ──────────────────────────► spectacles_head   → [B,2]        │
│  x_non_spec ──[GRL]── da_spec_head  → [B,2]                            │
│                                                                         │
│  x_fh     ──────────────────────────► facial_hair_head  → [B,2]        │
│  x_non_fh   ──[GRL]── da_fh_head   → [B,2]                            │
│                                                                         │
│  x_emot   ──────────────────────────► emotion_head      → [B,2]        │
│  x_non_emot ──[GRL]── da_emot_head → [B,2]                            │
│                                                                         │
│  x_pose   ──────────────────────────► pose_head         → [B,2]        │
│  x_non_pose ──[GRL]── da_pose_head → [B,2]                            │
│                                                                         │
│  x_gender ──────────────────────────► gender_head       → [B,2]        │
│  x_id       ──[GRL]── da_gender_head→ [B,2]                            │
│                                                                         │
│  x_id ──► BN(512) → AdaptiveAvgPool(1,1) → Flatten → vec [B,512]      │
│           └─► MagLinear(512, N_classes)                                 │
│               → [cos_theta [B,N], cos_theta_m [B,N]], x_norm [B,1]    │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Feature Separation Module (FSM / AttentionModule)

Mỗi FSM học cách **tách** một loại thông tin thuộc tính ra khỏi feature map, trả về 2 nhánh:

```
           ┌─────────────────────────────────────┐
Input x    │  Channel Attention                  │
[B,512,H,W]│                                     │
     │     │  SPP_avg ─┐                         │
     ├────►│  SPP_max ─┴─► Sum ─► Conv1×1(512×14│
     │     │              → ReLU ─► Conv1×1(512) │
     │     │              → BN ─► Sigmoid        │
     │     │              → channel_scale [B,512,1,1]│
     │     └─────────────────────────────────────┘
     │
     │     ┌─────────────────────────────────────┐
     │     │  Spatial Attention                  │
     ├────►│  max_C(x)  ─┐                       │
     │     │  mean_C(x) ─┴─► Cat [B,2,H,W]       │
     │     │              → Conv7×7 → BN → Sig.  │
     │     │              → spatial_scale [B,1,H,W]│
     │     └─────────────────────────────────────┘
     │
     │  x_non_id = 0.5 × (x × ch_scale + x × sp_scale)
     │  x_id     = x − x_non_id
     │
     ├──────────────────► x_id     (phần chứa identity / attribute)
     └──────────────────► x_non_id (phần không chứa attribute đó)
```

**SPP (Spatial Pyramid Pooling):** dùng sizes `(1, 2, 3)` → tổng `1+4+9 = 14` vị trí mỗi channel → concat thành `[B, 512×14, 1, 1]`.

---

## 5. Gradient Reversal Layer (GRL)

```
Forward:   output = input   (pass-through)
Backward:  grad   = −λ × grad_input   (đảo dấu gradient)
```

**Mục đích:** Khi train, GRL buộc `x_non_attr` **không chứa** thông tin thuộc tính `attr`. Nếu head DA vẫn predict được, gradient ngược chiều sẽ "xóa" thông tin đó khỏi `x_non_attr`.

---

## 6. MagLinear — ID Head

```
Input: x [B, 512]  (đã flatten)

x_norm = ||x||₂  (clamp vào [l_a=10, u_a=110])
margin  = (u_m − l_m) / (u_a − l_a) × (x_norm − l_a) + l_m   ← adaptive

W_norm = normalize(W)   W ∈ ℝ^{512×N}
cos_θ  = normalize(x) · W_norm              ∈ ℝ^{B×N}
cos_θ_m = cos_θ·cos_m − sin_θ·sin_m        (margin penalty cho GT class)

output = one_hot × cos_θ_m + (1−one_hot) × cos_θ
```

**Ý nghĩa:** Ảnh chất lượng cao (norm lớn) → margin lớn → bị phạt nặng hơn → embedding phân biệt rõ hơn.

---

## 7. Loss Function

### 7.1 Các thành phần loss (11 task)

| # | Task | Hàm loss | Class weight |
|---|------|----------|-------------|
| 0 | Identity (Main) | WeightClassMagFace | Theo tần suất ngược |
| 1 | Gender | CrossEntropy | `[2.0, 1.0]` |
| 2 | Emotion | CrossEntropy | `[1.0, 2.0]` |
| 3 | Pose | CrossEntropy | `[1.0, 3.0]` |
| 4 | Facial Hair | CrossEntropy | `[1.0, 1.5]` |
| 5 | Spectacles | CrossEntropy | `[1.0, 2.0]` |
| 6–10 | DA × 5 attrs | CrossEntropy | (same as main) |

### 7.2 WeightClassMagFace

```
loss_g = 1/u_a² × ||x|| + 1/||x||          (regularize norm)
output = one_hot × cos_θ_m + (1−one_hot) × cos_θ
loss_id = CrossEntropy(scale × output, target)
alpha_i = class weight theo tần suất ngược (min=0.3, max=1.0)
loss = mean((loss_id + λ_g × loss_g) × alpha)
```

### 7.3 Auto-weighting (Kendall 2018)

```
L_total = Σᵢ [ 0.5 × exp(−sᵢ) × Lᵢ  +  0.5 × sᵢ ]
            └────────────────────────┘    └─────────┘
               precision × task loss       regularize
```

11 tham số `sᵢ = log(σᵢ²)` được học như parameter. Task khó (loss cao, nhiễu) → `σ` tăng → trọng số giảm tự động.

---

## 8. Metric đánh giá

| Metric | Mô tả | Cách tính |
|--------|-------|-----------|
| **AUC ID (Cosine)** | Face verification | Pairwise cosine similarity trên toàn tập, ROC-AUC |
| **AUC ID (Euclidean)** | Face verification | Pairwise Euclidean distance (negated), ROC-AUC |
| **AUC Gender** | Binary classification | ROC-AUC với prob[:, 1] là positive |
| **AUC Spectacles** | Binary classification | ROC-AUC |
| **AUC Facial Hair** | Binary classification | ROC-AUC |
| **AUC Pose** | Binary classification | ROC-AUC |
| **AUC Emotion** | Binary classification | ROC-AUC |

**Cách tính AUC ID:**
```
1. Chạy inference toàn dataset → ma trận embedding [N, 512]
2. Normalize L2: emb_norm = emb / ||emb||
3. cos_sim = emb_norm @ emb_norm.T          [N, N]
4. euc_dist = cdist(emb_norm, emb_norm)     [N, N]
5. labels = (id_i == id_j).int()           [N, N]  (1=cùng người)
6. Lấy upper triangle (tránh duplicate)
7. AUC = roc_auc_score(labels, cos_sim)
         roc_auc_score(labels, -euc_dist)
```

**Monitoring theo epoch:**

```
Metric            Train    Test
────────────────────────────────
loss              X        -
loss_id           X        -
loss_gender       X        X
loss_da_gender    X        X
loss_emotion      X        X
loss_da_emotion   X        X
loss_pose         X        X
loss_da_pose      X        X
loss_facial_hair  X        X
loss_da_fh        X        X
loss_spectacles   X        X
loss_da_spec      X        X
auc_gender        X        X
auc_spectacles    X        X
auc_facial_hair   X        X
auc_pose          X        X
auc_emotion       X        X
auc_id_cosine     X        X    ← metric chính
auc_id_euclidean  X        X
```

---

## 9. Data Pipeline

```
Photometric Stereo data
  ├── {id}/{session}/albedo_map_new_crop.exr.npy     [H,W,3] float32
  ├── {id}/{session}/normal_map_new_crop.exr.npy
  └── {id}/{session}/depth_map_new_crop.exr.npy

train_set.csv / test_set.csv:
  columns: id, session, Gender, Spectacles, Facial_Hair, Pose, Emotion

PhotometricDataset.__getitem__:
  → load .npy → augment (Albumentations) → Tensor [3, 112, 112]
  → label Tensor [6]  (id, gender, spec, fh, pose, emotion)

Sampler (tuỳ chọn):
  UniqueIdBatchSampler: mỗi batch có batch_size identity khác nhau (PK sampling)
  → tốt hơn cho metric learning
```

**Augmentations:**
- `RandomResizedCropRect(112, scale=(0.7,1.0))` — crop ngẫu nhiên rồi resize
- `GaussianNoise(std=0–0.1)` — nhiễu vùng ngẫu nhiên (10–25% diện tích ảnh)

---

## 10. Backbone variants

| Variant | Nguồn | Kích thước feature cuối | Ghi chú |
|---------|-------|------------------------|---------|
| `convnextv2_tiny` | timm | 768 | Mặc định |
| `convnextv2_base` | timm | 1024 | Nặng hơn |
| `miresnet18/34` | custom iResNet | 512 | Nhẹ, baseline |
| `miresnet50/101/152` | custom iResNet | 2048→512 | Nặng |

Tất cả variant đều qua **Adapter Conv1×1 → 512ch** trước khi vào FSM.

---

## 11. Luồng gradient

```
Loss_total
   │
   ├──► grad ID head     ──► FSM_gender (x_id branch)    ──┐
   │                                                        │
   ├──► grad gender head ──► FSM_gender (x_gender branch) ──┤
   │                                                        │
   ├──► grad da_gender   ──[GRL: flip]──► FSM_gender (x_id) ← buộc x_id không chứa gender info
   │                                                        │
   ├──► grad pose head   ──► FSM_pose                      ──┤
   ├──► grad da_pose     ──[GRL: flip]──► FSM_pose          ──┤  ... tương tự
   │                                                        │
   └──► ... (emotion, facial_hair, spectacles)              │
                                                            ▼
                                                      Backbone weights
```

GRL tạo ra áp lực ngược chiều: feature `x_non_attr` bị "đẩy" ra khỏi không gian phân biệt được thuộc tính `attr` → disentanglement thực sự được enforce bởi gradient.

---

## 12. Tham khảo

| Paper | Vai trò trong project |
|-------|-----------------------|
| MagFace (CVPR 2021) | ID loss + adaptive margin + norm regularization |
| ConvNeXt V2 (CVPR 2023) | Backbone chính |
| Multi-Task Learning Using Uncertainty (CVPR 2018) | Auto loss weighting (log_vars) |
| Domain-Adversarial Training (JMLR 2016) | GRL + DA heads |
| CBAM / BAM | Inspiration cho AttentionModule (channel + spatial) |
| SPP-Net | SPP trong AttentionModule |