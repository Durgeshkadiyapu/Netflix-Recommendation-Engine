# 🎬 Netflix Recommendation Engine

A collaborative filtering-based movie recommendation system built using **SVD (Singular Value Decomposition)** on the real Netflix Prize dataset. The system predicts personalized movie ratings for any user and ranks movies they haven't seen yet by estimated enjoyment.

---

## 📌 Project Overview

This project replicates the core of Netflix's recommendation engine using **matrix factorization** (SVD). Given a user's past ratings, the model learns latent preference patterns and predicts how much that user would enjoy any unwatched movie — then ranks and recommends the top picks.

---

## 🚀 Key Highlights

| Metric | Value |
|--------|-------|
| Dataset | Netflix Prize Dataset (`combined_data_1.txt`) |
| Ratings Processed | **100,000+ user-movie ratings** |
| Algorithm | **SVD (Singular Value Decomposition)** via scikit-surprise |
| Validation | **3-Fold Cross-Validation** with RMSE |
| Filtering | Quantile-based (60th percentile) noise removal |
| Output | Ranked personalized movie recommendations per user |

---

## 🛠️ Tech Stack

| Category | Tools |
|----------|-------|
| Language | Python |
| Data Processing | Pandas, NumPy |
| Recommendation | scikit-surprise (SVD, Reader, Dataset) |
| Validation | Cross-validation (RMSE) |
| Visualization | Matplotlib, Seaborn |
| Environment | Google Colab / Jupyter Notebook |

---

## 📂 Project Structure

```
Netflix-Recommendation-Engine/
│
├── Netflix_project.ipynb       # Main notebook — full pipeline
├── Netflix_Dataset_.txt        # Raw Netflix Prize ratings data
├── movie_titles.csv            # Movie metadata (ID, Year, Title)
└── README.md                   # Project documentation
```

---

## ⚙️ How It Works

### 1. Data Loading & Parsing
- Loaded the Netflix Prize dataset (`combined_data_1.txt`) with custom parsing
- Extracted Movie IDs from row separators (e.g., `1:` → Movie ID 1)
- Merged with `movie_titles.csv` for human-readable movie names

### 2. Exploratory Data Analysis
- Counted total movies, unique customers, and total ratings
- Analyzed rating distribution (1–5 stars) using `value_counts()`
- Identified data sparsity patterns in the user-movie matrix

### 3. Noise Filtering (Quantile-Based)
- Removed movies with fewer ratings than the **60th percentile** threshold
- Removed customers who rated fewer movies than the **60th percentile** threshold
- This step significantly reduces noise and improves model accuracy

### 4. SVD Model (Matrix Factorization)
- Used `scikit-surprise`'s `SVD` algorithm — the same class of technique used by winning teams in the Netflix Prize competition
- Loaded filtered data into the Surprise `Dataset` format using a `Reader`
- Trained SVD to learn latent factors for both users and movies

### 5. Cross-Validation
- Applied **3-fold cross-validation** with **RMSE** as the evaluation metric
- Ensures the model generalizes well beyond the training data

### 6. Personalized Recommendations
- For a given user, predicted estimated ratings for all movies they haven't seen
- Sorted predictions in descending order to surface top recommendations
- Example: Generated top movie picks for User ID `2643029`

---

## 📊 Pipeline Summary

```
Raw Data → Parse Movie IDs → EDA → Quantile Filtering
        → SVD Training → Cross-Validation → Predict Ratings
        → Sort by Estimated Rating → Top-N Recommendations
```

---

## 📈 Sample Output

For **User 2643029**, the model predicts and ranks all eligible movies by estimated rating, returning the top unwatched recommendations sorted by predicted enjoyment score.

---

## 🔧 How to Run

```bash
# Clone the repository
git clone https://github.com/Durgeshkadiyapu/Netflix-Recommendation-Engine.git
cd Netflix-Recommendation-Engine

# Install dependencies
pip install pandas numpy matplotlib seaborn scikit-surprise

# Launch the notebook
jupyter notebook Netflix_project.ipynb
```

> **Note:** The dataset file is large. You may need to mount Google Drive if running on Colab, or download the Netflix Prize dataset from Kaggle.

---

## 🧠 What I Learned

- How to parse and clean real-world messy datasets with custom logic
- Quantile-based filtering for removing sparse, low-signal data
- Matrix factorization using SVD for collaborative filtering
- Model evaluation using RMSE and cross-validation with scikit-surprise
- End-to-end recommendation system design from raw data to ranked output

---

## 👤 Author

**Kadiyapu Durgesh Kumar**  
B.Tech CSE (AI & ML Minor) | VIT-AP University  
AI & Data Science Trainee | DRISHTI CPS, IIT Indore

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue)](https://linkedin.com/in/durgesh-kumar-37a46a31b)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-black)](https://github.com/Durgeshkadiyapu)

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).
