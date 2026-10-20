# AI-Based Phishing Website Detection Using Machine Learning

An end-to-end cybersecurity machine learning system that classifies website URLs as **Legitimate** or **Phishing** using **54 structural and lexical features** evaluated by a **Tuned Random Forest Classifier**.

---

## 📌 Project Overview & Objective
Phishing remains the most pervasive vector for identity theft, credential harvesting, and financial fraud. Attackers frequently generate deceptive domain names, abuse subdomains, impersonate trusted brands, or inject misleading path tokens to trick victims.

The objective of this project is to provide a fast, non-intrusive, and resilient URL security analyzer capable of evaluating any raw URL string in milliseconds without downloading heavy webpage content or executing potentially hazardous JavaScript.

---

## 🛠️ Technologies & Frameworks
- **Programming Language**: Python 3.12
- **Machine Learning**: `scikit-learn` (Random Forest, SVM, Decision Tree, Logistic Regression)
- **Data Manipulation**: `pandas`, `numpy`
- **Model Serialization**: `joblib`
- **Web Application**: `Streamlit`
- **URL & Network Parsing**: `urllib.parse`, `tldextract`, `requests`

---

## 📂 Project Structure
```text
AI-Phishing-Detection/
│
├── app.py                           # Production Streamlit Web Application
├── requirements.txt                 # Project dependencies
├── README.md                        # Documentation & setup guide
│
├── data/
│   ├── raw/
│   │   └── dataset_phishing.csv     # Benchmark raw dataset (11,430 samples)
│   └── processed/                   # Prepared train/test splits (scaled & unscaled)
│
├── models/
│   ├── random_forest_tuned.joblib   # Production Random Forest classifier
│   ├── svm_tuned.joblib             # Tuned SVM classifier
│   ├── scaler.joblib                # Fitted StandardScaler (for linear/SVM models)
│   ├── feature_names.json           # Canonical list of 54 features in exact order
│   ├── model_evaluation_report.json # Phase 4 baseline evaluation metrics
│   └── hyperparameter_tuning_report.json # Phase 5 cross-validation tuning report
│
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py        # Data pipeline & stratified 80/20 splitting
│   ├── model_training.py            # Baseline model training and comparison
│   ├── model_tuning.py              # 5-fold CV hyperparameter tuning
│   ├── feature_extraction.py        # Live 54-feature URL extraction engine
│   ├── predict.py                   # End-to-end inference pipeline
│   ├── validate_features.py         # Feature vector integrity checker
│   └── validate_prediction.py       # Inference validation utility
│
└── tests/
    ├── test_feature_extraction.py   # Extractor unit tests on benchmark URLs
    ├── test_prediction.py           # Pipeline integration tests
    └── test_audit_urls.py           # Pre-deployment edge case & error tests
```

---

## 📊 Dataset Description
The project utilizes the benchmark phishing dataset published by **Abdelhakim Hannousse and Salima Yahiouche (2021)** (*Engineering Applications of Artificial Intelligence*, Vol. 104, DOI: [10.17632/c2gw7fy2j4.2](https://doi.org/10.17632/c2gw7fy2j4.2)):
- **Total Samples**: 11,430 URLs
- **Class Balance**: Exactly **5,715 Legitimate (50.0%)** and **5,715 Phishing (50.0%)**
- **Missing Values**: 0 nulls across all records
- **Features**: 54 numerical features (after removing raw URL string, target label, and constant zero-variance column `nb_or`)

---

## 🔍 The 54 Extracted URL Features
The system extracts 54 numerical properties across three operational categories:

1. **Local Deterministic Lexical & Structural Features (41)**:
   - Lengths: `length_url`, `length_hostname`, `shortest_words_raw`, `longest_words_raw`, `avg_words_raw`, etc.
   - Special Characters: `nb_dots`, `nb_hyphens`, `nb_at`, `nb_qm`, `nb_and`, `nb_eq`, `nb_slash`, `nb_colon`, `nb_semicolumn`, etc.
   - Domain Tokens: `nb_www`, `nb_com`, `http_in_path`, `https_token`, `punycode`, `port`, `nb_subdomains`, `prefix_suffix`.
   - Statistical Ratios: `ratio_digits_url`, `ratio_digits_host`, `char_repeat`.

2. **Dataset-Specific Heuristics (11)**:
   - `phish_hints`: Frequency of sensitive credential keywords (`login`, `signin`, `account`, `banking`, `verify`, etc.).
   - Brand Spoofing: `domain_in_brand`, `brand_in_subdomain`, `brand_in_path` across 25+ trusted institutions.
   - Anomaly Indicators: `abnormal_subdomain`, `random_domain`, `shortening_service`, `path_extension`, `suspecious_tld`.

3. **Network-Dependent Redirection Features (2)**:
   - `nb_redirection`: Count of HTTP redirect hops.
   - `nb_external_redirection`: Count of HTTP redirect hops landing on an external root domain.
   - *Resilience*: Uses an HTTP redirect check with a 1.5-second timeout and safely defaults to `0` if offline or unreachable.

---

## 🏆 Machine Learning Evaluation & Tuning

### 1. Baseline Model Comparison (Evaluated on Unseen Test Set: $N = 2,286$)
Four classifiers were trained on the training partition ($N = 9,144$) and tested on the held-out test partition:

| Model | Train Accuracy | Held-Out Test Accuracy | Precision | Recall (Phishing) | F1-Score |
|---|:---:|:---:|:---:|:---:|:---:|
| **Random Forest (Baseline)** | 99.69% | 91.25% | 91.11% | 91.43% | 91.27% |
| **SVM (Baseline)** | 90.14% | 89.37% | 89.40% | 89.33% | 89.37% |
| **Decision Tree** | 99.69% | 87.53% | 87.18% | 88.01% | 87.59% |
| **Logistic Regression** | 85.24% | 84.43% | 86.67% | 81.36% | 83.94% |

### 2. Controlled Hyperparameter Tuning
Using **Stratified 5-Fold Cross-Validation** strictly on the training set with **F1-score** as the primary selection criterion:
- **Selected Hyperparameters**:
  - `n_estimators`: `200`
  - `max_features`: `'log2'` (decorrelates trees effectively across 54 features)
  - `max_depth`: `None`
  - `min_samples_split`: `2`
  - `min_samples_leaf`: `1`
  - `random_state`: `42`

### 3. Final Production Model Benchmarks
- **Accuracy**: **91.29% accuracy on the held-out test set**
- **Phishing F1-Score**: **91.37%**
- **Phishing Recall**: **92.13%** (Caught 1,053 of 1,143 phishing sites; missed only 90)
- **Input Representation**: **Un-scaled numerical features** (StandardScaler is not applied to tree ensembles)

> **Notice**: These metrics represent performance evaluated strictly on the benchmark held-out test partition ($N = 2,286$). They do not constitute an absolute guarantee of safety for arbitrary real-world websites.

---

## 🚀 Installation & Local Execution

### 1. Clone the Repository
```bash
git clone https://github.com/USER/AI-Phishing-Detection.git
cd AI-Phishing-Detection
```

### 2. Set Up Virtual Environment
```powershell
python -m venv venv
.\venv\Scripts\activate
```

### 3. Install Dependencies
```powershell
pip install -r requirements.txt
```

### 4. Run the Streamlit Application
```powershell
streamlit run app.py
```
Open your browser to: `http://localhost:8501`

---

## 🧪 Testing & Verification Suites
The project includes three automated testing utilities:

1. **Feature Extractor Validation**:
   ```powershell
   python tests/test_feature_extraction.py
   ```
2. **Prediction Pipeline Integration**:
   ```powershell
   python tests/test_prediction.py
   ```
3. **Pre-Deployment Edge Case & Error Audit**:
   ```powershell
   python tests/test_audit_urls.py
   ```

---

## 🔒 Private Deployment Guide (Private GitHub + Private Streamlit Cloud)

To deploy the application securely as a **Private Application** accessible only to authorized evaluators:

### Step 1: Create a PRIVATE GitHub Repository
1. Log in to [GitHub](https://github.com).
2. Click **New Repository**.
3. Choose a name (e.g., `ai-phishing-detection`).
4. **CRITICAL**: Select **Private** (do NOT choose Public).
5. Do not initialize with README, `.gitignore`, or license (they already exist locally).
6. Click **Create repository**.

### Step 2: Push Local Code to the Private Repository
Run the following commands in PowerShell from the project root:
```powershell
git init
git add .
git commit -m "feat: complete production phishing detection system"
git branch -M main
git remote add origin https://github.com/<YOUR_GITHUB_USERNAME>/<YOUR_REPO_NAME>.git
git push -u origin main
```
*(Because `models/random_forest_tuned.joblib` is ~38.5 MB, it pushes directly without requiring Git LFS).*

### Step 3: Connect to Streamlit Community Cloud
1. Sign in to [share.streamlit.io](https://share.streamlit.io/) with your GitHub account.
2. If prompted, grant Streamlit access to your **Private repositories**.

### Step 4: Deploy the App
1. Click **New app**.
2. Select your **Private repository** (`<YOUR_GITHUB_USERNAME>/ai-phishing-detection`).
3. Set **Branch**: `main`.
4. Set **Main file path**: `app.py`.
5. Click **Deploy**.

### Step 5: Enforce Private Access & Invite Authorized Viewers
1. By default, apps deployed from a private GitHub repository on Streamlit Cloud are **Private** to your account.
2. In the Streamlit Cloud dashboard, click the **Settings (⚙️)** or **Share** icon on your app card.
3. Under **Viewer Access**, ensure the app is set to **Private** (restricted to you and invited viewers).
4. Under **Manage Viewers**, enter the email addresses of your professors, project evaluators, or viva examiners.
5. Authorized recipients will receive a secure magic login link to access the live app. Unauthenticated public users will see an access-denied screen.

---

## ⚠️ Limitations
1. **Lexical Focus**: The model analyzes structural and lexical patterns in the URL. Phishing pages hosted on high-reputation domains (e.g., compromised cloud storage or public forms) may require supplementary DOM content analysis.
2. **Dynamic Redirections**: Live HTTP redirect counts rely on server responsiveness. If a website is offline, unreachable, or times out past 1.5 seconds, redirect counts safely default to `0`.
3. **Probabilistic Estimation**: Outputs represent statistical model probabilities rather than a formal cryptographic or antivirus guarantee.

---

## 🔮 Future Scope
- **Browser Extension Integration**: Package the Python inference engine into a lightweight WebExtension for instant navigation warnings.
- **Selective DOM Scanning**: Ingest page title and form action attributes as secondary verification when URL risk is in the medium boundary (30%–60%).
- **Continuous Threat Ingestion**: Retrain the ensemble periodically against emerging daily phishing feeds (e.g., PhishTank, OpenPhish).
