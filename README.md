# 🤖 AI Code Detection System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Flask](https://img.shields.io/badge/Flask-3.0+-green.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-orange.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

**A comprehensive machine learning system for detecting AI-generated code and identifying code authorship**

[Features](#-features) • [Installation](#-installation) • [Quick Start](#-quick-start) • [Documentation](#-documentation) • [Contributing](#-contributing)

</div>

---

## 📖 Overview

This project implements **three distinct machine learning tasks** for code analysis using Random Forest classifiers:

| Task | Description | Output Classes | Accuracy |
|------|-------------|----------------|----------|
| **Task A** | Binary Machine-Generated Detection | Human, Machine | ~90% |
| **Task B** | Multi-Class Authorship Detection | Human + 10 LLM Families | ~82% |
| **Task C** | Hybrid Code Detection | Human, Machine, Hybrid, Adversarial | ~85% |

### 🎯 Task B: LLM Families
DeepSeek-AI • Qwen • 01-ai • BigCode • Gemma • Phi • Meta-LLaMA • IBM-Granite • Mistral • OpenAI

---

## ✨ Features

### 🧠 Machine Learning
- ✅ **Random Forest Classifiers** with 400-500 trees for maximum accuracy
- ✅ **Feature Engineering** - 35-50 code-specific features per task
- ✅ **Class Balancing** - Handles imbalanced datasets effectively
- ✅ **Model Compression** - Efficient storage with lossless compression
- ✅ **Out-of-Bag Scoring** - Built-in cross-validation

### 🌐 Web Interface
- ✅ **Beautiful UI** - Gradient purple theme with smooth animations
- ✅ **Three Interactive Tabs** - One for each task
- ✅ **Real-time Analysis** - Get predictions in seconds
- ✅ **Probability Visualization** - Horizontal bar charts for all classes
- ✅ **Responsive Design** - Works on desktop and mobile

### 🛠️ Developer Tools
- ✅ **Training Scripts** - Easy model training and evaluation
- ✅ **REST API** - Flask server with CORS support
- ✅ **Submission Generators** - Create competition-ready CSV files
- ✅ **Comprehensive Logging** - Track model performance

---

## 🏗️ Project Structure
```
project/
│
├── Task A/                      # Binary Detection (Human vs Machine)
│   ├── train_subtask_a.py          # Train and evaluate model
│   ├── generate_submission_a.py     # Generate competition files
│   ├── binary_code_model.pkl       # Trained model
│   └── task_a_*.parquet            # Datasets
│
├── Task B/                      # Authorship Detection (11 classes)
│   ├── train_subtask_b.py          # Train and evaluate model
│   ├── generate_submission_b.py     # Generate competition files
│   ├── authorship_model.pkl        # Trained model
│   └── task_b_*.parquet            # Datasets
│
├── Task C/                      # Hybrid Detection (4 classes)
│   ├── train_subtask_c.py          # Train and evaluate model
│   ├── generate_submission_subtask_C.py
│   ├── hybrid_code_model.pkl       # Trained model
│   └── task_c_*.parquet            # Datasets
│
├── website/                         # Web Application
│   ├── app.py                      # Flask backend server
│   └── index.html                   # Frontend interface
│
├── README.md                        # This file
└── requirements.txt                 # Python dependencies
```

---

## 📦 Installation

### Prerequisites
- Python 3.8 or higher
- 8 GB RAM minimum (16 GB recommended for Task B)
- 5 GB free disk space

### Step 1: Clone Repository
```bash
git clone ttps://github.com/yourusername/ai-code-detection.git](https://github.com/gnandeep22/SemEval-Task13-GenAI.git
cd SemEval-Task13-GenAI
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Verify Installation
```bash
python -c "import sklearn, pandas, flask; print('✓ All dependencies installed!')"
```

---

## 🚀 Quick Start

### 1️⃣ Train Models
```bash
# Task A: Binary Detection (~15-20 minutes)
cd Task A
python train_subtask_a.py --train

# Task B: Authorship Detection (~30-40 minutes)
cd ../Task B
python train_subtask_b.py --train

# Task C: Hybrid Detection (~10-15 minutes)
cd ../Task C
python train_subtask_c.py --train
```

### 2️⃣ Start Web Application
```bash
# Navigate to website folder
cd website

# Start Flask server
python app.py
```

### 3️⃣ Open in Browser

- **Option 1:** Double-click `website/index.html`
- **Option 2:** Visit `http://localhost:5000`

### 4️⃣ Analyze Code

1. Paste your code in the text area
2. Click "🔍 Analyze Code"
3. View results with confidence scores and probability charts

---

## 🎯 How It Works

### Feature Extraction Pipeline
```
Code Input → Feature Extraction → 40-50 Features → Random Forest → Prediction
```

### Extracted Features

Each code snippet is analyzed across multiple dimensions:

| Category | Features | Examples |
|----------|----------|----------|
| **Basic Metrics** | 4-6 | Lines, characters, avg length |
| **Whitespace** | 4-6 | Indentation style, spaces, tabs |
| **Comments** | 5-6 | Comment density, docstrings, TODO markers |
| **Naming** | 5-7 | camelCase, snake_case, variable lengths |
| **Type Hints** | 3-4 | Type annotations, return types |
| **Structure** | 5-6 | Functions, classes, imports, error handling |
| **Control Flow** | 4-5 | Nesting depth, conditionals, loops |
| **LLM Patterns** | 6 | "Example usage", "Here's...", markdown |

**Total:** 35-50 features depending on task

---

## 🌐 REST API

### Endpoints

#### Health Check
```bash
GET http://localhost:5000/
```

**Response:**
```json
{
  "status": "running",
  "available_models": ["A", "B", "C"]
}
```

#### Make Prediction
```bash
POST http://localhost:5000/predict_a
Content-Type: application/json

{
  "code": "def hello():\n    print('Hello!')"
}
```

**Response:**
```json
{
  "prediction": "human",
  "confidence": 87.45,
  "probabilities": {
    "human": 87.45,
    "machine": 12.55
  }
}
```

### Available Endpoints
- `POST /predict_a` - Task A (Binary Detection)
- `POST /predict_b` - Task B (Authorship)
- `POST /predict_c` - Task C (Hybrid Detection)

---

## 📊 Model Performance

### Task A: Binary Detection
- **Macro F1:** ~90%
- **Features:** 40
- **Classes:** 2
- **Model Size:** ~1.2 GB

### Task B: Authorship Detection
- **Macro F1:** ~82%
- **Features:** 50
- **Classes:** 11
- **Model Size:** ~1.5 GB

### Task C: Hybrid Detection
- **Macro F1:** ~85%
- **Features:** 35
- **Classes:** 4
- **Model Size:** ~800 MB

---

## 💻 Usage Examples

### Command Line Prediction
```bash
# Interactive mode
cd Task A
python train_subtask_a.py --predict

# Paste code and press Enter twice
```

### Generate Competition Submissions
```bash
cd Task A
python generate_submission_a.py

# Creates:
# - submission_subtask_a.csv (predictions)
# - gold_subtask_a.csv (validation labels)
```

### Validate Model
```bash
python train_subtask_a.py --validate
```

### Test Model
```bash
python train_subtask_a.py --test
```

---

## 🔧 Configuration

### High Accuracy Settings

**Task A & B:**
```python
N_ESTIMATORS = 500      # Number of trees
MAX_DEPTH = 40          # Tree depth
MIN_SAMPLES_SPLIT = 2   # Minimum samples to split
MIN_SAMPLES_LEAF = 1    # Minimum samples in leaf
```

**Task C:**
```python
N_ESTIMATORS = 400
MAX_DEPTH = 35
```

### Model Compression

All models use `compress=3` for efficient storage:
```python
joblib.dump(model_data, MODEL_OUTPUT, compress=3)
```

This reduces file size by ~60-70% with **zero accuracy loss**.

---

## 🔍 Internal Architecture

### Data Flow
```
User Input (Code)
    ↓
Website Frontend (JavaScript)
    ↓
Flask Server (app.py)
    ↓
Feature Extraction (extract_features_a/b/c)
    ↓
Load Model (.pkl file)
    ↓
Random Forest Prediction
    ↓
JSON Response {prediction, confidence, probabilities}
    ↓
Website Display (Bar Charts)
```

### Component Interaction
```
┌─────────────────┐
│  Training Data  │
│   (.parquet)    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ train_subtask   │
│   _x.py         │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Trained Model  │
│     (.pkl)      │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Flask Server   │
│    (app.py)     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Web UI        │
│ (index.html)    │
└─────────────────┘
```

---

## 🐛 Troubleshooting

### Model Won't Load
**Error:** `Unable to allocate memory`

**Solution:** Model file too large. Retrain with compression:
```bash
cd Task B
del authorship_model.pkl
python train_subtask_b.py --train
```

### Port Already in Use
**Error:** `Port 5000 already in use`

**Solution (Windows):**
```bash
netstat -ano | findstr :5000
taskkill /PID <PID> /F
```

**Solution (Linux/Mac):**
```bash
lsof -ti:5000 | xargs kill -9
```

### Import Errors
```bash
pip install --upgrade -r requirements.txt
```

### Slow First Prediction
**Normal behavior** - First prediction loads the model (~5-10 seconds). Subsequent predictions are fast (<1 second).

---

## 📚 Documentation

### Training Guide

Each task has similar training options:
```bash
python train_subtask_x.py --help

Options:
  --train       Train the model
  --test        Test on test set
  --validate    Validate on validation set
  --predict     Interactive prediction mode
```

### Feature Extraction

Features are extracted in `extract_features()` function:
- **Task A:** 40 features - `extract_features_a()`
- **Task B:** 50 features - `extract_features_b()`
- **Task C:** 35 features - `extract_features_c()`

### Model Configuration

Edit these parameters in training scripts:
```python
# In train_subtask_x.py

N_ESTIMATORS = 500      # More = better accuracy (slower)
MAX_DEPTH = 40          # Deeper = more complex patterns
MIN_SAMPLES_SPLIT = 2   # Fine-grained splitting
```

---

## 🤝 Contributing

Contributions are welcome! Here's how:

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **Commit** changes (`git commit -m 'Add amazing feature'`)
4. **Push** to branch (`git push origin feature/amazing-feature`)
5. **Open** a Pull Request

### Areas for Improvement
- Add XGBoost/LightGBM models
- Implement model explainability
- Add code syntax highlighting
- Create Docker container
- Improve mobile UI
- Add batch prediction API



## 🙏 Acknowledgments

- **scikit-learn** - Machine learning framework
- **Flask** - Web server framework
- **Pandas** - Data manipulation
- **NumPy** - Numerical computing

---

## 📞 Contact

- **GitHub Issues:** [Report bugs or request features](https://github.com/gnandeep22/SemEval-Task13-GenAI/issues)
- **Email:** gnandeep.padala@gmail.com
- **Project Link:** [https://github.com/gnandeep22/SemEval-Task13-GenAI](https://github.com/gnandeep22/SemEval-Task13-GenAI)

---

---

## ⭐ Star History

If you find this project useful, please consider giving it a star! ⭐

---
