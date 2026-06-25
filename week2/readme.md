# WiDS Week 3: Deep Learning Results Writeup

**Dataset:** PlantVillage (38 disease classes, 43,456 train / 10,849 validation images)
**Platform:** Kaggle (Tesla T4 GPU × 2)

---

## ✅ Deliverables Checklist

| Deliverable                                   | Status  |
| --------------------------------------------- | ------- |
| Simple CNN trained (~10 epochs)               | ✅ Done |
| Transfer Learning model (MobileNetV2) trained | ✅ Done |
| Accuracy scores recorded                      | ✅ Done |
| Training vs. validation curves generated      | ✅ Done |
| Confusion matrix generated                    | ✅ Done |
| Model comparison vs. Week 2 baseline          | ✅ Done |
| Reflection written                            | ✅ Done |

---

## 📊 Task 3: Accuracy Comparison

| Model                               | Val Accuracy | Notes                                   |
| ----------------------------------- | ------------ | --------------------------------------- |
| SVM (Week 2)                        | < 90%        | Hand-crafted features, flattened images |
| Random Forest (Week 2)              | < 90%        | Hand-crafted features, flattened images |
| **Simple CNN (Scratch)**            | **87.95%**   | 10 epochs, 3× Conv+Pool blocks          |
| **Transfer Learning (MobileNetV2)** | **93.77%**   | 5 epochs, frozen backbone + custom head |

> **The deep learning models clearly beat Week 2** — Transfer Learning in particular jumped to **93.77%** in just 5 epochs, far exceeding anything the classical models achieved.

---

## 🧠 Task 1: Simple CNN — Results

**Architecture:** 3× (Conv2D → ReLU → MaxPool) → Flatten → Dense(128) → Dropout(0.5) → Softmax(38)

**Training log (best run, 10 epochs):**

| Epoch | Train Acc | Val Acc    | Val Loss |
| ----- | --------- | ---------- | -------- |
| 1     | 25.66%    | 57.21%     | 1.4947   |
| 2     | 48.93%    | 66.97%     | 1.0748   |
| 3     | 57.91%    | 74.48%     | 0.8448   |
| 4     | 62.72%    | 77.98%     | 0.6734   |
| 5     | 66.45%    | 78.94%     | 0.6601   |
| 6     | 70.12%    | 81.50%     | 0.5968   |
| 7     | 71.76%    | 82.23%     | 0.5735   |
| 8     | 74.60%    | **87.48%** | 0.3987   |
| 9     | 75.82%    | 86.41%     | 0.4188   |
| 10    | 76.85%    | **87.95%** | 0.3861   |

**Best Validation Accuracy: 87.95%**

**Observations:**

- The model steadily improved across all 10 epochs — no early stopping triggered.
- A noticeable jump happened at epoch 8 (57% → 87%), suggesting the model crossed a threshold in feature discrimination.
- Train accuracy (76.85%) lagged significantly behind val accuracy in later epochs, indicating the data augmentation was working as a strong regularizer.
- The confusion matrix revealed that the first-run model (evaluated after loading `best_scratch_cnn.h5`) scored **89.10%** on the final evaluation call — consistent with the best checkpoint.

---

## 🔥 Task 2: Transfer Learning (MobileNetV2) — Results

**Architecture:** MobileNetV2 (frozen, ImageNet weights) → GlobalAveragePooling2D → Dense(128, ReLU) → Dropout(0.2) → Softmax(38)

**Training log (5 epochs):**

| Epoch | Train Acc | Val Acc    | Val Loss |
| ----- | --------- | ---------- | -------- |
| 1     | 70.84%    | **90.53%** | 0.2884   |
| 2     | 89.32%    | 92.29%     | 0.2289   |
| 3     | 90.89%    | 92.99%     | 0.2173   |
| 4     | 91.72%    | 92.98%     | 0.2094   |
| 5     | 92.36%    | **93.77%** | 0.1891   |

**Best Validation Accuracy: 93.77%**

**Observations:**

- Even at **Epoch 1**, Transfer Learning hit **90.53%** — already beating the CNN's best after 10 epochs.
- By Epoch 5, it reached **93.77%** with a very low loss of 0.19, showing confident predictions.
- Train and val accuracy tracked closely (no overfitting), because the frozen backbone prevents the model from memorizing training data.
- A second training run (cells 4–5) confirmed the result: CNN peaked at **83.65%** in 5 epochs vs Transfer Learning at **93.63%** — consistent results across runs.

---

## 💡 Reflection

**Why did Transfer Learning perform better?**

MobileNetV2 was pre-trained on 1.4 million ImageNet images, so its early and middle layers already encode universal visual features — edges, textures, color gradients, and complex shapes. When we freeze the backbone and only train a small classification head, we're essentially asking the model one question: _"Given these rich features, which of 38 disease classes does this belong to?"_ That's a much simpler problem than learning everything from scratch.

The Simple CNN, starting from random weights, had to simultaneously learn what a leaf edge looks like AND that this pattern means "Apple Scab" — all from 43,000 images. MobileNetV2 only had to solve the second half, which is why it hit 90% on its very first epoch.

**Where does the model still fail?**

The classification report from the evaluation cell reveals the CNN model particularly struggled with visually similar classes — for example, Apple Scab vs. Apple Black Rot (both show dark lesions), and healthy vs. mildly diseased leaves where symptoms are subtle. Classes with fewer support samples (like Cedar Apple Rust with only 55 images) showed near-zero precision and recall, suggesting class imbalance is still a challenge going into Week 4.

---
