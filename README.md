# 🤖 Ali Yasser — AI & Machine Learning Projects

A collection of projects built during my **AI department** studies, covering Neural Networks, Deep Learning, Intelligent Programming, and Recommender Systems.

---

## 📁 Projects

---

### 🧠 Brain Tumor Classifier (Neural Networks)
**Files:** `Neural project.ipynb` · `neural app.py`

Trains and compares two neural network architectures to classify **brain MRI scans** into 4 tumor categories using PyTorch. Comes with a full Streamlit web app for interactive inference.

**Classes:** Glioma · Meningioma · No Tumor · Pituitary

**Models built from scratch:**
| Model | Architecture |
|---|---|
| FFNN | Flatten → Linear(49152→1024) → BN → ReLU → Dropout → Linear(→512) → Linear(→4) |
| CNN | 4× [Conv2d → BatchNorm → ReLU → MaxPool] → FC layers |

**Training details:**
- Input: 128×128 RGB images
- Augmentation: Random flip, rotation, random crop
- Optimizer: Adam (lr=1e-3, weight_decay=1e-4)
- Scheduler: ReduceLROnPlateau
- Epochs: 30 · Batch size: 32

**Streamlit app features:**
- Upload any MRI image and get predictions from FFNN and/or CNN
- Side-by-side confidence bar charts
- Model agreement indicator
- Configurable checkpoint paths and class names via sidebar

**Tech:** `PyTorch` · `torchvision` · `Streamlit` · `NumPy` · `Matplotlib` · `Pillow`

---

### 🎬 Hybrid Movie Recommender System — IP Final Project
**Files:** `IP Final project.ipynb` · `app.py`

A hybrid recommendation system combining **Content-Based Filtering** and **Collaborative Filtering** (via SVD) to recommend movies — with an Arabic-language Streamlit interface.

**How it works:**
- **Content-Based:** TF-IDF on movie genres + Cosine Similarity
- **Collaborative:** SVD (12 components) on user-movie rating matrix → predicted ratings
- **Hybrid score:** `0.5 × collaborative_score + 0.5 × content_similarity`

**Tech:** `Pandas` · `Scikit-learn` · `SciPy` · `NumPy` · `Streamlit`

**Data:** MovieLens-style `movies.csv` + `ratings.csv`

---

### ❤️ Heart Disease Detection — IP Project 1
**Files:** `IP Project 1.ipynb`

Detects heart disease risk using two approaches on the **Heart Disease UCI dataset**, then compares them head-to-head.

**Approaches:**
| Approach | Description |
|---|---|
| Expert System | Rule-based scoring on age, cholesterol, blood pressure, exercise angina, ST depression |
| Decision Tree | GridSearchCV-tuned classifier (depth, min_samples_split, 5-fold CV) |

**Pipeline:** EDA → Missing value imputation → MinMaxScaler → Train/Test split (70/30) → GridSearchCV → Evaluation → Model export (`joblib`)

**Metrics reported:** Accuracy · Precision · Recall · F1-Score · Feature Importances

**Tech:** `Pandas` · `Scikit-learn` · `Matplotlib` · `Seaborn` · `Joblib`

---

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| Deep Learning | PyTorch · torchvision |
| Machine Learning | Scikit-learn · SciPy |
| Data | Pandas · NumPy |
| Visualization | Matplotlib · Seaborn |
| Web Apps | Streamlit |
| Environment | Jupyter Notebook · Python 3.x |

---



> **Note for Neural App:** Save trained model weights first from the notebook:
> ```python
> torch.save(ffnn_model.state_dict(), "ffnn_model.pth")
> torch.save(cnn_model.state_dict(), "cnn_model.pth")
> ```

---

## 📦 Requirements

```
torch
torchvision
numpy
pandas
matplotlib
seaborn
scikit-learn
scipy
streamlit
pillow
joblib
jupyter
```

Install all: `pip install -r requirements.txt`

---

## 👤 About Me

**Ali Yasser** — AI Department Student

- 🎯 Interests: Neural Networks · Deep Learning · Computer Vision · Recommender Systems · Intelligent Systems
- 🔗 GitHub: [@omaralimona05-maker](https://github.com/omaralimona05-maker)
- 📧 Email:omaralimona05@gmail.com
- 💼 LinkedIn:https://www.linkedin.com/in/ali-yasser-350799328/

