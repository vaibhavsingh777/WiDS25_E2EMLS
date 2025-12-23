# 🌿 Plant Disease Classification: My Week 1 Data Audit

## 1. The Imbalance Trap

My first task was quantifying the distribution. I discovered a massive **36:1 imbalance ratio**.

* **The Majority**: `Orange___Haunglongbing (Citrus greening)` is a giant with over **5,500 images**.
* **The Minority**: `Potato___healthy` is a tiny fraction with only **152 images**.

**The Learning:** If I just look at raw accuracy, my model could hit 90% by simply ignoring potatoes and raspberries entirely. For next steps, I’ve decided to prioritize **F1-Score** and **Balanced Accuracy** to ensure every plant gets a fair diagnosis.

---

## 🔍 2. Scaling the Audit: 11,000+ Image Check

I didn't just look at 10 images; I scaled my sanity check to **300 samples per class**.

* **Hard Issues**: I found **2 critical errors** (corrupt or extreme brightness) that would have crashed a training loop or introduced pure noise.
* **The "Blurry" Truth**: I flagged **285 images** with a Laplacian variance score below 100.

**The Insight:** I noticed a pattern where **Healthy Corn** images were consistently out of focus (some scores as low as **4.66**). Neptune.ai warns that blurriness kills small-feature detection. Since disease spots are often tiny, these 285 images are "noisy data" that might cause False Negatives later on.

---

## 🧬 3. Decoding Visual Similarity with t-SNE

I used **t-SNE (t-Distributed Stochastic Neighbor Embedding)** to see how a computer actually perceives these leaves.

* **What I did**: I extracted 2,048 features using a pre-trained ResNet50 and squashed them into a 2D map.
* **What I found**: The map showed that "Healthy" leaves across different species cluster closer together than diseased leaves of the same species.
* **The Overlap**: I saw significant overlap between `Tomato Late Blight` and `Septoria Leaf Spot`. This "Visual Similarity Trap" confirms that the model will need high-resolution feature maps to tell these apart.

---

## 🖼️ 4. The Gallery: Side-by-Side Consistency

Finally, I created a **10x10 Gallery** to test my own eyes.

* **Consistency**: `Corn Rust` is incredibly uniform; every image looks the same.
* **Chaos**: `Peach Bacterial Spot` is highly variable—some leaves have tiny specks while others are totally yellowed.

**The Verdict:** My manual gallery check confirmed that my model can't just look for "color"—it needs to understand **texture** and **edge patterns** to be reliable in a real farm environment.

---

## ✅ Week 1 Deliverables

* [x] **Linked Kaggle & GitHub** for version control.
* [x] **Confirmed 256x256 resolution** (No destructive resizing needed!).
* [x] **Audited 11,000+ images** for integrity.
* [x] **Mapped Similarity** via unsupervised t-SNE.

