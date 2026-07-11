# 🌱 Plant Disease Detection | End-to-End ML Systems | WiDS 5.0

**Dataset:** [PlantVillage (Kaggle)](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset/data)

---

## 📌 Project Overview
This project explores the progression from traditional Machine Learning to state-of-the-art Deep Learning, Federated Learning, and basic MLOps practices for classifying plant leaf diseases. Working with the **PlantVillage dataset**, the goal is to build a robust, scalable, privacy-aware, and deployment-ready model capable of assisting in disease diagnosis for crops such as Apple, Corn, Potato, Tomato, and others.

Beyond model training, the project also demonstrates **local inference**, **metric visualization**, and **deployment-style workflows**, bridging the gap between experimentation and real-world usability.

All code is organized across multiple Kaggle notebooks and supporting scripts, each linked within this repository.

The project follows a **First Principles approach** over a structured **5-week timeline**:

- **Week 1:** Understanding the data through rigorous Exploratory Data Analysis (EDA).
- **Week 2:** Establishing performance baselines using Classical Machine Learning techniques (SVMs, Random Forests).
- **Week 3:** Developing deep learning models using CNNs and Transfer Learning.
- **Week 4:** Simulating decentralized training using Federated Learning (Flower + FedAvg).
- **Week 5:** Persisting models and metrics, visualizing training behavior, enabling local inference, and introducing deployment-oriented MLOps workflows.

---

## 📂 Repository Structure

| File / Notebook | Description |
|----------------|------------|
| `eda-week1.ipynb` | Exploratory Data Analysis: data inspection, visualization, and statistical profiling |
| `ClassicalML_Week2.ipynb` | Classical ML baselines: feature engineering, PCA, SVM vs Random Forest |
| `CNN_Week3.ipynb` | Deep Learning: custom CNN and Transfer Learning with MobileNetV2 |
| `FederatedLearning_Week4.ipynb` | Federated Learning simulation using Flower and Federated Averaging |
| `MLOps_Week5.ipynb` | Model persistence, metric logging, Streamlit visualization, and deployment packaging |
| `app.py` | Streamlit application for visualizing federated training metrics and inspecting model behavior |
| `predict.py` | Local inference script for testing individual leaf images using the trained global model |

---

## 🗓️ Weekly Methodology & Technical Details

### Week 1: Exploratory Data Analysis (EDA)

**Goal**  
Peform EDA to get at least class distribution visualization, sample image grid (multiple images per class), Image resolution/size analysis and 2-3 meaningful insights about the dataset.

**Key Decisions & Analysis**

- **Data Selection**:
Chose the *color* folder in the dataset to preserve diagnostic features such as chlorosis and rust spores that are lost in grayscale images.
- **Class Distribution Analysis**:
Various different graphs & visualisations were plotted like bar graphs, pie chart, sun burst chart, stacked bar graphs, treemap to assess the distribution pattern in the dataset.

- **Data Quality Audit**:
   - Aspect ratio was checked using PIL, where all the images were found to be of the standard 256×256 size.
   - Background consistency check was done using corner-pixel color sampling.
   - Blurs were detected using the concept of Laplacian blur for various classes & plotted into a candlestick chart.
   - Lighting variation analysis was done using HSV-based brightness (V-channel) distribution across classes.
   - Leaf-to-background ratio check was done using grayscale conversion and Otsu’s thresholding to estimate leaf pixel coverage.
   
- **Visual inspection**:
  This was performed using class-wise galleries and side-by-side comparisons.
  
- **Biological Feature Analysis**:
   - Chlorophyll variation was analyzed using green-channel intensity distributions across biologically diverse classes.
   - Biological color separability was evaluated using mean HSV hue and saturation features across healthy, viral, and fungal classes.
   - Leaf texture differences were analyzed using GLCM-based contrast and homogeneity features.
   - Full RGB color distribution was analyzed using mean channel intensities across disease classes.
   - Model feasibility was evaluated using PCA on flattened pixel intensities to assess class separability.
   - Non-linear class separability was analyzed using t-SNE embeddings of scaled pixel features.

**Key Insight**  
- Massive class imbalance observed:
  - `Orange___Huanglongbing`: ~5.5K images  
  - `Corn_(maize)___healthy`: 21 images
- Background: Corner pixel analysis confirms a somehwat uniform background.
- Lighting: Brightness distribution is more or less consistent.
- Several disease classes are visually similar, increasing the difficulty of fine-grained classification.

---

### Week 2: Classical Machine Learning Baselines

**Goal**  
Evaluate the limits of non-deep-learning approaches & find the "ceiling" of classical Machine Learning. Build a system that classifies plant diseases using "Old School" methods (SVMs, Random Forests) on raw pixel data. This score will serve as the benchmark that our Deep Learning models must beat in the week that follows.

#### Feature Engineering
- Images were resized from `(224, 224, 3)` to `(64, 64, 3)` and flattened into a feature matrix with corresponding class labels. This was done to ease the load on RAM & also faster & efficient training. 
- Data was split into training and testing sets and standardized using z-score normalization via StandardScaler.
- Dimensionality reduction was performed using fast randomized **Principal Component Analysis (PCA)**, reducing the feature space from `12,288` to `150` components.

#### Models Evaluated
- **DummyClassifier** (Chance baseline): 10.58% accuracy
- **Nearest Centroid**: 37.28% accuracy
- **Linear SVM (PCA+Linear)**: 66.18% accuracy
- **Random Forest Classifier**: 68.53% accuracy (best classical model)

#### Failure Analysis
- Random Forest performs well when diseases have distinct visual signatures or consistent texture/color cues. On raw pixels, it handles coarse texture and strong color cues well.
- Several classes exhibit near-zero recall due to strong visual similarity with related diseases and class imbalance, highlighting the inability of pixel-based Random Forests to separate subtle disease variations.
- Healthy leaves are frequently misclassified because “healthy” varies significantly across plant species, making a single healthy label biologically inconsistent in raw pixel feature space.
- Diseases within the same plant species are heavily confused as they differ primarily in fine-grained lesion morphology rather than global color, which flattened pixel representations fail to capture.

**Conclusion**  
Classical machine learning models on flattened pixel features achieve moderate performance (~69% accuracy) and capture coarse color and texture cues but fail on fine-grained, visually similar diseases and several minority classes. The confusion patterns, especially within the same plant species and between healthy vs. diseased leaves highlight fundamental limitations of non-spatial models. These results establish a solid baseline and clearly motivate the need for CNN-based approaches to learn localized, disease-specific visual features for meaningful performance gains. 

---

### Week 3: Deep Learning & Transfer Learning

**Goal**  
Build a CNN that learns features directly from images and outperform classical ML, then apply transfer learning to achieve ~85–90% accuracy, marking the transition from traditional methods to modern, powerful deep learning.

---

#### Stage 1: Simple CNN (From Scratch)
The dataset was split into augmented training and clean validation subsets using an 80–20 split. DataLoaders were created for batched training and validation, and augmented samples were visually inspected.
**Architecture**
- A deliberately *minimal* 2-layer CNN:
  - `Conv2D (3 → 16 filters, 3×3) → ReLU → MaxPool (2×2)`
  - `Conv2D (16 → 32 filters, 3×3) → ReLU → MaxPool (2×2)`
  - `Flatten → Dense (32 × 56 × 56 → 38)`

**The Overfitting Experiment**
- The model was trained using mini-batch gradient descent with backpropagation for 10 epochs and evaluated on a validation set each epoch.
- Using Adam and cross-entropy loss, the CNN learned effective visual features.
- Results:
  - Training Accuracy: **~99.5%**
  - Validation Accuracy: **~88%**

**Key Observation**
- Loss curves showed classic divergence:
  - Training loss ↓
  - Validation loss ↑  
  → Clear evidence of overfitting

---

#### Stage 2: Transfer Learning (The Solution)

**Architecture**
- A **MobileNetV2** model pretrained on ImageNet was initialized, its backbone frozen, and the classifier head replaced to adapt the network for plant disease classification.
**Technique**
- Frozen Backbone reuses pretrained feature extractors.
- Custom Classification Head is a linear layer for 38 classes.
- Data Augmentation is still present in the form of random rotations, horizontal/vertical flips, affine zoom transformations.

**Result**
- **97% validation accuracy** in just **5 epochs**.
- Stable convergence with no train–validation gap.

---

### Week 4: Federated Learning Simulation

**Goal**  
Evaluate whether high-performing centralized models can maintain accuracy when trained in a decentralized, privacy-preserving setup.

---

**Methodology**

- **Environment Setup:**  
  The Flower federated learning framework was installed programmatically with system and backend logs suppressed to ensure a clean execution environment.

- **Runtime Configuration:**  
  Core libraries were imported, runtime warnings and internal logs were silenced, and GPU availability was configured for efficient federated training.

- **Data Partitioning:**  
  The dataset was normalized and partitioned across **3 simulated clients**, with each client independently split into **80% training / 20% validation** sets to emulate decentralized data ownership.

- **Federated Client Design:**  
  A **MobileNetV2 model wrapped into a Flower `NumPyClient`** implementing local training, parameter exchange, and evaluation logic was defined.

- **Federated Training:**  
  Federated training was executed for **three rounds using FedAvg across three clients**, achieving stable convergence (**~92% → ~96% accuracy**). Runtime warnings originated from multi-process GPU initialization and dependency mismatches and did not impact training correctness.

---

**Results**

- **Distributed Accuracy Trend:**  
  **92.4% → 94.9% → 95.8%**

- **Centralized vs Federated Gap:**  
  Performance drop relative to centralized baseline ≈ **1.5%**, well within acceptable privacy tradeoff limits.

- **Deployment Insight:**  
  Confirms feasibility of privacy-preserving deployment without major accuracy degradation.

---

### Week 5: Federated Model Persistence & MLOps Foundations

**Goal**  
Transition from experimental federated training to a production-oriented workflow by persisting models and metrics, enabling reproducibility, inspection, and lightweight monitoring.

---

**Methodology**

- **Environment Hardening**
  - Installed Flower and supporting libraries programmatically with backend logs suppressed for a clean MLOps workspace.
  - Configured structured output directories for artifacts:
    - `outputs/models/` → Saved global model weights  
    - `outputs/logs/` → Federated training metrics  

- **Data & Model Setup**
  - Reused the normalized PlantVillage dataset and 3-client partitioning strategy from Week 4.
  - Continued using transfer-learned **MobileNetV2** with a frozen backbone and custom classification head.

- **Federated Training Enhancements**
  - Increased federated rounds to **10** for smoother convergence.
  - Added **multiple local epochs per client (3 epochs)** to improve local optimization.
  - Reduced learning rate for improved training stability.
  - Used weighted accuracy aggregation under **FedAvg**.

- **Artifact Persistence**
  - Persisted federated metrics per round into a CSV file (`federated_metrics.csv`).
  - Extracted and saved the final global model weights (`global_plant_model.pth`).
  - Packaged all outputs into a deployable archive for portability and reuse.

- **Visualization & Monitoring**
  - Loaded saved metrics locally and visualized training behavior using **Streamlit**.
  - Enabled post-training analysis without rerunning federated simulations.
  - Simulated lightweight model monitoring and inspection workflows.

---

**Results**

- Federated accuracy steadily improved over 10 rounds:  
  **93.8% → 97.4%**
- Final accuracy closely matches centralized performance, confirming stability under longer federated training.
- All training artifacts were successfully persisted and reused locally via Streamlit visualization.

---

**Key Insight**

Week 5 demonstrates the shift from *model experimentation* to *system engineering*.  
Persisting models and metrics enables reproducibility, debuggability, and real-world deployment workflows — a critical requirement for federated systems where training behavior is inherently distributed and harder to observe.

---

## 📊 Results Summary

| Model Architecture | Accuracy | Key Observation |
| :--- | :---: | :--- |
| Random Forest (Week 2) | 68.53% | Failed to detect fine-grained disease patterns (near-zero recall on some classes) |
| Simple CNN (Week 3) | ~88% | High overfitting; unstable validation loss and limited generalization |
| Transfer Learning – MobileNetV2 (Week 3) | **97.34%** | Robust convergence; solved previously “invisible” classes |
| Federated Averaging (3 Clients) – MobileNetV2 (Week 4) | **95.87%** | Minimal accuracy drop (~1.5%) while preserving privacy and stability |
| Federated Model Deployment & Monitoring (Week 5) | **97.38%** | Persisted model and metrics; enabled reproducible evaluation and live visualization |

---

## 📈 Performance Visualizations

### 1. Limits of Classical ML & Naive Learning (Baseline Problems)
Classical machine learning baselines (Random Forests, SVMs) plateau around ~65–70% accuracy and fail to separate visually similar disease classes due to the absence of spatial feature learning. A naive CNN trained from scratch further demonstrates instability: training accuracy rapidly approaches 100% while validation accuracy plateaus around ~87–88%, accompanied by decreasing training loss and increasing validation loss. This behavior reflects memorization rather than robust feature extraction, leading to poor generalization on unseen samples.

---

### 2. Robustness in Transfer Learning (Centralized Solution)
MobileNetV2 exhibits stable convergence with rapidly aligning training and validation curves, steadily decreasing loss, and consistently high validation accuracy. The confusion matrix demonstrates near-perfect diagonal dominance, confirming strong class-wise separability across all 38 diseases.

---

### 3. Stability in Federated Learning (Decentralized Extension)
Federated training maintains stable convergence across communication rounds (**~92% → ~96% accuracy**) with only a minor performance drop relative to centralized training. This demonstrates that privacy-preserving decentralized learning retains strong predictive performance and model robustness.

---

### 4. Observability & Reproducibility in MLOps (Deployment Layer)
Federated training metrics were persisted to disk and visualized using a lightweight Streamlit dashboard, enabling post-training inspection without rerunning simulations. Model weights were saved and packaged for reuse, demonstrating reproducibility, deployment readiness, and real-world monitoring workflows.

---

## ✅ Key Takeaways

- EDA-driven insights strongly influence downstream modeling decisions
- Classical ML models are insufficient for high-dimensional image data
- CNNs learn spatial hierarchies but require regularization
- Transfer Learning provides both **performance and efficiency**, even with limited training epochs
- A slight accuracy drop (“privacy tax”) is observed in Federated Learning, enabling decentralized training with minimal performance compromise.
- Persisting models and metrics enables reproducibility, post-training analysis, and deployment readiness — bridging experimentation with real-world MLOps practices.
