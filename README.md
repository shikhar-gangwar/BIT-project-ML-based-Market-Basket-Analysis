# BIT Project: Machine Learning-Based Market Basket Analysis

[![Python Version](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.14-blue.svg)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.67.0-FF4B4B.svg)](https://streamlit.io)
[![MLxtend](https://img.shields.io/badge/MLxtend-0.25.0-brightgreen.svg)](http://rasbt.github.io/mlxtend/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> Academic Project developed at **Birla Institute of Technology (BIT)** focusing on Machine Learning and Unsupervised Pattern Mining for Retail Affinity Discovery.

---

## 📌 Project Overview
Market Basket Analysis is a core data mining technique used by retailers to uncover associations between items purchased together. This project delivers an end-to-end machine learning pipeline that:
1. Cleans and audits over **1,041,000 retail transaction records**.
2. Encodes shopping baskets into a high-dimensional one-hot transaction matrix.
3. Implements and compares **Apriori** and **FP-Growth** frequent itemset mining algorithms.
4. Generates and filters robust association rules using statistical metrics: **Support, Confidence, Lift, Leverage, and Conviction**.
5. Deploys an interactive multi-page **Streamlit Web Application** for executive KPI tracking, dynamic rule search, and real-time visualization.

---

## 🚀 Quickstart & Installation

```bash
git clone https://github.com/shikhar-gangwar/BIT-project-ML-based-Market-Basket-Analysis.git
cd Market_Basket_Analysis
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run Market_Basket_Analysis/app.py
```

---

## 📊 Benchmark Summary: Apriori vs. FP-Growth

| Metric | Apriori Algorithm | FP-Growth Algorithm | Comparison |
|---|---|---|---|
| **Execution Time (min_sup = 0.01)** | 72.82 s | **4.07 s** | **FP-Growth is 17.9× faster** |
| **Frequent Itemsets Discovered** | 712 | 712 | 100% Exact Equivalence |
| **Association Rules Generated** | 143 | 143 | 100% Identical Rules |
| **Average Rule Confidence** | 33.21% | 33.21% | Identical |
| **Average Rule Lift** | 8.27 | 8.27 | Identical |
| **Scalability (min_sup = 0.005)** | Crashed (MemoryError >59 GiB) | **5.62 s (2,175 itemsets)** | **FP-Growth scales seamlessly** |
