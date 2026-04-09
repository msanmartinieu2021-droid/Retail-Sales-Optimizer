# Retail Sales Optimizer

An integrated retail analytics platform combining **customer segmentation**, **demand prediction**, and **recommendation systems** into a single, transparent decision-support workflow.

This project was developed as part of a Final Degree Thesis (TFG/DBA) focused on building a unified analytics system that transforms raw retail data into actionable, profitability-oriented insights.

---

## Project Overview

Retail and e-commerce companies generate large volumes of transactional and behavioral data. However, many existing analytics solutions treat segmentation, predictive modeling, and profitability analysis as separate modules.

This project integrates:

- Customer segmentation (K-Means + PCA)
- Segment-level discount optimization (LightGBM)
- Implicit recommendation system (ALS)
- Interactive decision-support web application (Streamlit)

The goal is to create a transparent and adaptable system that supports real-world retail decision-making.

---

## Key Features

### 1. Automatic Data Processing
- Schema auto-detection
- Flexible column mapping
- Missing value handling
- Feature engineering
- Synthetic discount generation (when needed)

### 2. Customer Segmentation
- Aggregated behavioral and transactional features
- PCA for dimensionality reduction
- K-Means clustering
- Silhouette-based K selection
- Balanced cluster assignment (optional)

### 3. Discount Optimization
- Segment-level demand modeling
- LightGBM regression
- Revenue simulation across candidate discount levels
- Optimal discount selection per segment

### 4. Recommendation System
- Implicit feedback matrix (purchase frequency)
- Alternating Least Squares (ALS)
- Personalized category suggestions
- Affinity score normalization

### 5. Interactive Web App
- Built with Streamlit
- Upload custom datasets
- Adjust segmentation parameters
- Compute discount policies
- Generate recommendations

---

## Methodology Summary

The analytical pipeline integrates:

- **Unsupervised Learning** → K-Means segmentation  
- **Supervised Learning** → LightGBM demand prediction  
- **Matrix Factorization** → ALS recommendation system  

The system emphasizes:
- Interpretability
- Generalization across datasets
- Transparency (non–black-box logic)
- Business-oriented outputs (discount policies & segment profiles)

---

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- LightGBM
- Implicit (ALS)
- SciPy (Sparse matrices)
- Streamlit

---

## Project Structure

Retail-Sales-Optimizer/
│
├── retail_models.py        
├── app_retail.py           
├── retail_pipeline.ipynb   
├── requirements.txt
└── README.md

## License
This project is for academic and research purposes.
