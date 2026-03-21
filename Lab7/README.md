# Lab 7: CNN với PyTorch - CIFAR-10, Cats vs Dogs & PlantVillage

## Tổng Quan

Repository này chứa bài thực hành mở rộng từ Lab 6 (CNN thuần – MNIST), áp dụng CNN vào **3 bài toán phân loại ảnh thực tế phức tạp hơn** với yêu cầu đạt **>90% accuracy** và **tránh overfitting**:

| File | Bài toán | Số lớp | Kích thước ảnh | Best Acc |
|------|----------|--------|----------------|----------|
| `cifar10-cnn.ipynb` | Phân loại 10 đối tượng | 10 | 32×32 RGB | >90% |
| `catdog-cnn.ipynb` | Phân loại Mèo / Chó | 2 | 160×160 RGB | >90% |
| `plantvillage-cnn.ipynb` | Phân loại bệnh lá cây | 38 | 128×128 RGB | >90% |

> **Ràng buộc**: Không sử dụng pretrained models (ResNet, ConvNeXt, EfficientNet,...). Tất cả CNN được xây dựng và huấn luyện từ đầu (from scratch).

---

## So Sánh với Lab 6

| Tiêu chí | Lab 6 (MNIST) | Lab 7 (CIFAR-10 / CatDog / PlantVillage) |
|----------|---------------|------------------------------------------|
| **Dữ liệu** | Ảnh xám 28×28, 1 kênh | Ảnh màu RGB, kích thước lớn hơn |
| **Số lớp** | 10 | 2 – 38 |
| **Độ phức tạp** | Thấp | Cao (texture, màu sắc, biến thiên ánh sáng) |
| **Kiến trúc** | 2 Conv layer, FC | 4–5 Conv block, BatchNorm, GAP |
| **Augmentation** | Không | RandomCrop, Flip, ColorJitter, Cutout/Erasing |
| **Optimizer** | SGD + Momentum | AdamW + LR Scheduler |
| **Cân bằng dữ liệu** | Không cần | WeightedRandomSampler |
| **Accuracy mục tiêu** | ~98-99% | >90% |

---

## Công Nghệ Sử Dụng

### Thư viện chính:
- **PyTorch** – Framework deep learning
- **torchvision** – Dataset, transforms, ImageFolder
- **NumPy** – Xử lý mảng số
- **Matplotlib / Seaborn** – Trực quan hóa kết quả
- **scikit-learn** – Classification report, confusion matrix

---

## Phân Tích & Cân Bằng Dữ Liệu

Trước khi huấn luyện, mỗi notebook thực hiện **phân tích dữ liệu** đầy đủ:

- Biểu đồ phân phối số lượng mẫu theo lớp
- Thống kê kích thước ảnh (min / max / mean width & height)
- Hiển thị mẫu ảnh từ từng lớp
- Tính **mean và std theo từng kênh màu** để chuẩn hóa chính xác

### Xử lý mất cân bằng dữ liệu

Với CIFAR-10 (cân bằng sẵn 5.000 mẫu/lớp), augmentation là đủ.

Với Cats vs Dogs và PlantVillage (có thể mất cân bằng), dùng:

```python
# Tính trọng số nghịch đảo tần suất cho mỗi mẫu
label_counts  = Counter(train_ds.targets)
class_weights = {k: 1.0 / v for k, v in label_counts.items()}
sample_weights = [class_weights[t] for t in train_ds.targets]

# Sampler đảm bảo mỗi lớp xuất hiện đều nhau trong mỗi batch
sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
train_loader = DataLoader(train_ds, batch_size=64, sampler=sampler, ...)
```

---

## Data Augmentation

Mỗi dataset dùng pipeline augmentation riêng phù hợp đặc thù:

### CIFAR-10
```python
transforms.Compose([
    transforms.RandomCrop(32, padding=4),       # Dịch chuyển ngẫu nhiên
    transforms.RandomHorizontalFlip(),          # Lật ngang
    transforms.ColorJitter(...),                # Biến đổi màu sắc
    transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    Cutout(n_holes=1, length=8),               # Che vùng ngẫu nhiên 8×8
])
```

**Cutout** là kỹ thuật che ngẫu nhiên một vùng ảnh (đặt về 0), buộc model không phụ thuộc vào một vùng đặc trưng duy nhất:

```python
class Cutout:
    def __call__(self, img):   # img: [C, H, W] tensor
        mask = torch.ones(h, w)
        mask[y1:y2, x1:x2] = 0.0
        return img * mask.unsqueeze(0)
```

### Cats vs Dogs
```python
transforms.Compose([
    transforms.Resize((180, 180)),
    transforms.RandomCrop(160),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.1),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
    transforms.RandomErasing(p=0.25, scale=(0.02, 0.2)),
])
```

### PlantVillage
```python
transforms.Compose([
    transforms.Resize((148, 148)),
    transforms.RandomCrop(128),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(30),             # Xoay nhiều hơn vì lá cây mọi hướng
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
    transforms.RandomErasing(p=0.3),
])
```

---

## Kiến Trúc Model

### 1. CIFAR-10 CNN (4 Conv Block)

```
Input (3×32×32)
    ↓ Block1: Conv(3→64)→BN→ReLU→Conv(64→64)→BN→ReLU → MaxPool(2×2) → Dropout(0.10)
    ↓ [32×32 → 16×16]
    ↓ Block2: Conv(64→128)→BN→ReLU→Conv(128→128)→BN→ReLU → MaxPool → Dropout(0.20)
    ↓ [16×16 → 8×8]
    ↓ Block3: Conv(128→256)→BN→ReLU→Conv(256→256)→BN→ReLU → MaxPool → Dropout(0.30)
    ↓ [8×8 → 4×4]
    ↓ Block4: Conv(256→512)→BN→ReLU→Conv(512→512)→BN→ReLU → MaxPool → Dropout(0.35)
    ↓ [4×4 → 2×2]
    ↓ Global Average Pooling → [512]
    ↓ FC(512→256) → ReLU → Dropout(0.4)
    ↓ FC(256→10)
    ↓ Output (10 classes)
```

**Code:**
```python
class CIFAR10_CNN(nn.Module):
    def __init__(self, num_classes=10, dropout=0.35):
        super().__init__()

        def block(in_c, out_c, drop=dropout):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_c), nn.ReLU(inplace=True),
                nn.Conv2d(out_c, out_c, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_c), nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2),
                nn.Dropout2d(drop),
            )

        self.features = nn.Sequential(
            block(3,   64,  drop=0.10),  # 32→16
            block(64,  128, drop=0.20),  # 16→8
            block(128, 256, drop=0.30),  # 8→4
            block(256, 512, drop=0.35),  # 4→2
        )
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256), nn.ReLU(inplace=True), nn.Dropout(0.4),
            nn.Linear(256, num_classes),
        )
```

---

### 2. Cats vs Dogs CNN (5 Conv Block)

```
Input (3×160×160)
    ↓ Block1: Conv(3→32)×2  → MaxPool → Dropout(0.05)   [160→80]
    ↓ Block2: Conv(32→64)×2 → MaxPool → Dropout(0.10)   [80→40]
    ↓ Block3: Conv(64→128)×2→ MaxPool → Dropout(0.20)   [40→20]
    ↓ Block4: Conv(128→256)×2→MaxPool → Dropout(0.30)   [20→10]
    ↓ Block5: Conv(256→512)×2→MaxPool → Dropout(0.40)   [10→5]
    ↓ Global Average Pooling → [512]
    ↓ FC(512→256) → ReLU → Dropout(0.4)
    ↓ FC(256→2)
    ↓ Output (cat / dog)
```

Sử dụng **Global Average Pooling** thay Flatten thẳng để giảm số tham số và tránh overfitting với ảnh lớn 160×160.

---

### 3. PlantVillage CNN (5 Residual Block)

```
Input (3×128×128)
    ↓ ResBlock(3→64,   drop=0.05) → MaxPool  [128→64]
    ↓ ResBlock(64→128, drop=0.10) → MaxPool  [64→32]
    ↓ ResBlock(128→256,drop=0.20) → MaxPool  [32→16]
    ↓ ResBlock(256→256,drop=0.30) → MaxPool  [16→8]
    ↓ ResBlock(256→512,drop=0.35) → MaxPool  [8→4]
    ↓ Global Average Pooling → [512]
    ↓ FC(512→256) → ReLU → Dropout(0.4)
    ↓ FC(256→38)
    ↓ Output (38 classes)
```

**ResBlock** dùng **skip connection** (kết nối tắt) để gradient truyền ngược dễ hơn qua mạng sâu:

```python
class ResBlock(nn.Module):
    def __init__(self, in_c, out_c, dropout=0.2):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c,  out_c, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_c), nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
        )
        # 1×1 conv để khớp số channel khi in_c ≠ out_c
        self.skip = nn.Sequential(
            nn.Conv2d(in_c, out_c, 1, bias=False),
            nn.BatchNorm2d(out_c)
        ) if in_c != out_c else nn.Identity()

    def forward(self, x):
        return relu(self.conv(x) + self.skip(x))  # F(x) + x
```

---

## Các Kỹ Thuật Tránh Overfitting

| Kỹ thuật | CIFAR-10 | CatDog | PlantVillage | Tác dụng |
|----------|----------|--------|--------------|----------|
| **BatchNorm** | ✓ | ✓ | ✓ | Ổn định quá trình huấn luyện, regularization nhẹ |
| **Dropout2d** | ✓ | ✓ | ✓ | Tắt ngẫu nhiên feature map, giảm co-adaptation |
| **Global Average Pooling** | ✓ | ✓ | ✓ | Giảm params FC, tránh overfit ảnh lớn |
| **Label Smoothing** | ✓ (0.1) | ✓ (0.05) | ✓ (0.1) | Tránh model quá tự tin, cải thiện calibration |
| **Weight Decay** | ✓ (5e-4) | ✓ (1e-3) | ✓ (5e-4) | L2 regularization trên trọng số |
| **Cutout** | ✓ | – | – | Che vùng ngẫu nhiên, tránh overfit vùng đặc trưng |
| **RandomErasing** | – | ✓ | ✓ | Tương tự Cutout nhưng linh hoạt hơn |
| **MixUp** | – | ✓ | – | Trộn 2 ảnh, ranh giới quyết định mềm hơn |

### MixUp (Cats vs Dogs)

```python
def mixup_data(x, y, alpha=0.4):
    """Trộn hai ảnh và nhãn theo λ ~ Beta(alpha, alpha)."""
    lam   = np.random.beta(alpha, alpha)
    index = torch.randperm(x.size(0))
    mixed_x = lam * x + (1 - lam) * x[index]
    return mixed_x, y, y[index], lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)
```

---

## Training

### Optimizer & Learning Rate Scheduler

| | CIFAR-10 | Cats vs Dogs | PlantVillage |
|--|----------|--------------|--------------|
| **Optimizer** | AdamW | AdamW | AdamW |
| **Learning Rate** | 1e-3 | 3e-4 | 5e-4 |
| **Weight Decay** | 5e-4 | 1e-3 | 5e-4 |
| **Scheduler** | Warmup(5ep) + Cosine | OneCycleLR | CosineAnnealingWarmRestarts |
| **Epochs** | 60 | 50 | 60 |
| **Batch Size** | 128 | 64 | 64 |

**Warmup + Cosine Annealing (CIFAR-10):**
```python
def warmup_cosine(epoch):
    if epoch < WARMUP_EPOCHS:
        return (epoch + 1) / WARMUP_EPOCHS          # Tăng tuyến tính 0 → 1
    progress = (epoch - WARMUP_EPOCHS) / (NUM_EPOCHS - WARMUP_EPOCHS)
    return 0.5 * (1 + np.cos(np.pi * progress))    # Cosine 1 → 0

scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_cosine)
```

**OneCycleLR (Cats vs Dogs)** – tăng lr đến max rồi giảm mạnh trong 1 chu kỳ:
```python
scheduler = optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr=3e-4,
    steps_per_epoch=len(train_loader),
    epochs=NUM_EPOCHS,
    pct_start=0.1,       # 10% đầu: warmup
    anneal_strategy='cos',
)
```

**CosineAnnealingWarmRestarts (PlantVillage)** – restart định kỳ để thoát local minima:
```python
scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer, T_0=15, T_mult=2, eta_min=1e-5
)
```

### Vòng lặp Huấn luyện

```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

for epoch in range(1, NUM_EPOCHS + 1):
    model.train()
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        out  = model(imgs)
        loss = criterion(out, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Gradient clipping
        optimizer.step()
    scheduler.step()
```

---

## Test-Time Augmentation (TTA) – Cats vs Dogs

Sau khi huấn luyện xong, dự đoán trên **5 phiên bản augment** của mỗi ảnh rồi lấy trung bình xác suất để cải thiện thêm ~1-2%:

```python
def tta_predict(model, imgs, n_aug=5):
    tta_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomCrop(IMG_SIZE, padding=10),
    ])
    probs_sum = torch.softmax(model(imgs), dim=1)
    for _ in range(n_aug - 1):
        aug_imgs = torch.stack([tta_transform(img) for img in imgs])
        probs_sum += torch.softmax(model(aug_imgs.to(device)), dim=1)
    return probs_sum / n_aug
```

---

## Kết Quả Mong Đợi

| Dataset | Train Acc | Val/Test Acc | Generalization Gap |
|---------|-----------|--------------|-------------------|
| **CIFAR-10** | ~90-92% | **>90%** | < 5% |
| **Cats vs Dogs** | ~88-92% | **>90%** | < 5% |
| **PlantVillage** | ~92-95% | **>90%** | < 5% |

- **Training time (GPU T4 Kaggle)**: ~20-40 phút/notebook
- **Training time (CPU)**: ~2-4 giờ/notebook

---

## Trực Quan Hóa

### 1. Biểu Đồ Loss & Accuracy

```python
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].plot(epochs, train_losses, label='Train')
axes[0].plot(epochs, val_losses,   label='Val', linestyle='--')
axes[1].plot(epochs, train_accs,   label='Train')
axes[1].plot(epochs, val_accs,     label='Val', linestyle='--')
axes[1].axhline(90, color='green', linestyle=':', label='Mục tiêu 90%')
```

### 2. Confusion Matrix

```python
from sklearn.metrics import confusion_matrix
import seaborn as sns

cm = confusion_matrix(all_labels, all_preds)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)
```

### 3. Accuracy từng lớp (PlantVillage)

```python
per_class_acc = cm.diagonal() / cm.sum(axis=1) * 100
# Hiển thị theo thứ tự tăng dần để dễ xác định lớp khó
sorted_idx = np.argsort(per_class_acc)
ax.barh(short_labels, per_class_acc[sorted_idx], color=colors)
```

### 4. Dự đoán mẫu

Hiển thị ảnh kèm nhãn dự đoán, nhãn thật và confidence score. Màu **xanh** = đúng, **đỏ** = sai.

---

## Cấu Trúc Thư Mục Dữ Liệu

### CIFAR-10
Tải tự động qua `torchvision.datasets.CIFAR10(root='./data', download=True)`.

### Cats vs Dogs
Tải tự động qua `tensorflow-datasets`:
```bash
pip install tensorflow-datasets tensorflow
```
Hoặc từ Kaggle: `kaggle competitions download -c dogs-vs-cats`

Cấu trúc sau khi tải:
```
data/catdog/
├── train/
│   ├── cat/  (cat_00000.jpg, ...)
│   └── dog/  (dog_00000.jpg, ...)
└── val/
    ├── cat/
    └── dog/
```

### PlantVillage
Từ Kaggle: Thêm dataset **emmarex/plantdisease** vào notebook (`Add Input`).

Cấu trúc:
```
/kaggle/input/plantvillage-dataset/PlantVillage/
├── Apple__Apple_scab/
├── Apple__Black_rot/
├── Apple__Cedar_apple_rust/
├── Apple__healthy/
├── ...  (38 lớp tổng cộng)
```

> Notebook tự động tìm đúng đường dẫn bằng cách quét đệ quy `/kaggle/input/`.

---

## Troubleshooting

### 1. PlantVillage – FileNotFoundError

**Triệu chứng:** `❌ Không tìm thấy thư mục chứa các lớp PlantVillage!`

**Giải pháp:** Vào **Add Input** (góc phải Kaggle notebook) → tìm `plantdisease` (emmarex/plantdisease) → Add. Notebook sẽ tự tìm đường dẫn đúng.

### 2. Accuracy không đạt 90%

**Kiểm tra:**
- GPU có được bật không? (`Settings → Accelerator → GPU T4`)
- Đã chạy đủ epoch chưa? (CIFAR: 60ep, CatDog: 50ep, PlantVillage: 60ep)
- Thử tăng thêm 10-20 epoch nếu accuracy vẫn đang tăng

### 3. Overfitting (Train Acc >> Val Acc)

**Giải pháp:**
- Tăng `weight_decay` (1e-3 → 3e-3)
- Tăng `dropout` (0.4 → 0.5)
- Thêm augmentation mạnh hơn

### 4. Loss Không Giảm

**Nguyên nhân:** Learning rate quá nhỏ hoặc quá lớn.

**Giải pháp:**
- CIFAR-10: thử `lr=5e-4` hoặc `lr=2e-3`
- CatDog / PlantVillage: thử `lr=1e-4` hoặc `lr=1e-3`

### 5. Out of Memory (OOM)

**Giải pháp:**
- Giảm `batch_size`: 128→64 (CIFAR-10), 64→32 (CatDog/PlantVillage)
- Giảm `IMG_SIZE`: 160→128 (CatDog), 128→96 (PlantVillage)

---

## So Sánh Kiến Trúc Giữa 3 Notebook

| | CIFAR-10 CNN | CatDog CNN | PlantVillage CNN |
|--|--------------|------------|-----------------|
| **Input size** | 3×32×32 | 3×160×160 | 3×128×128 |
| **Conv blocks** | 4 | 5 | 5 (Residual) |
| **Max channels** | 512 | 512 | 512 |
| **Skip connection** | Không | Không | Có (ResBlock) |
| **Pooling cuối** | GAP | GAP | GAP |
| **Kỹ thuật đặc biệt** | Cutout | MixUp + TTA | ResBlock |
| **LR Scheduler** | Warmup+Cosine | OneCycleLR | CosineWarmRestart |
| **Số lớp output** | 10 | 2 | 38 |

---

## Hướng Phát Triển

### 1. Cải thiện kiến trúc
- Thêm **Squeeze-and-Excitation (SE) block** để model tự học trọng số kênh quan trọng
- Thử **Depthwise Separable Convolution** để giảm tham số mà giữ hiệu năng

### 2. Cải thiện training
- **Stochastic Weight Averaging (SWA)**: trung bình trọng số cuối để model stable hơn
- **Progressive Resizing**: huấn luyện trên ảnh nhỏ trước, tăng dần

### 3. So sánh với Transfer Learning
```python
# Fine-tune ResNet18 (pretrained) để so sánh
from torchvision.models import resnet18
model_pretrained = resnet18(pretrained=True)
model_pretrained.fc = nn.Linear(512, NUM_CLASSES)
```
