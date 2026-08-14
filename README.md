# **Fraud Detection — BankSim Transaction Data**

End-to-end fraud detection project on 594,643 banking transactions, built to demonstrate a full analytics workflow: data cleaning, exploratory analysis, feature engineering, predictive modeling, and business translation — not just a classification exercise.

**Result**: XGBoost model catching 98.8% of fraud cases (PR AUC 0.927), translated into a tiered review system and merchant/category risk watchlist with an estimated net benefit of \~3.19M under moderate cost assumptions.

---

## **Project Structure**

fraud-detection-banksim/  
├── data  
│   ├── data.csv          \# raw dataset (download separately, see below)  
│   ├── X\_train\_final.csv            \# engineered features, output of notebook 03  
│   ├── X\_test\_final.csv  
│   ├── y\_train.csv  
│   └── y\_test.csv  
├── notebooks/  
│   ├── 01\_Cleaning\_Fraud.ipynb  
│   ├── 02\_EDA\_Fraud.ipynb  
│   ├── 03\_Feature\_Engineering\_Fraud.ipynb  
│   ├── 04\_Modeling\_Evaluation\_Fraud.ipynb  
│   └── 05\_Business\_Recommendation\_Fraud.ipynb  
├── outputs/  
│   ├── figures/                     
│   ├── model\_results.csv            \# row-level predictions from the final model  
│   └── xgboost\_fraud\_model.pkl      \# saved trained model  
└── README.md

---

## **Dataset**

**BankSim** — synthetic banking transaction data with real-world-like fraud patterns. Source: [kaggle.com/datasets/ntnu-testimon/banksim1](https://www.kaggle.com/datasets/ntnu-testimon/banksim1) File used: `bs140513_032310.csv` (the flat transaction file, not the network/graph version)

|  |  |
| ----- | ----- |
| Rows | 594,643 |
| Fraud rate | 1.21% (7,200 fraud cases) |
| Features | step, customer, age, gender, merchant, category, amount |

*Not included in this repo due to size — download directly from Kaggle and place in `data/`.*

---

## **How to Run**

1. Download the dataset from Kaggle (link above) into `data/bs140513_032310.csv`

Install dependencies:  
 pip install pandas numpy scikit-learn xgboost lightgbm imbalanced-learn matplotlib seaborn joblib

2.   
3. Run notebooks in order — each depends on the previous notebook's saved output: `01 → 02 → 03 → 04 → 05`

---

## **Methodology Summary**

### **1\. Data Cleaning**

Validated structure, dtypes, missing values (explicit and disguised), duplication, categorical consistency (whitespace, case-sensitivity, ID formatting), and relational consistency (e.g., customer age/gender stable across transactions). One anomaly (52 transactions with `amount = 0`) was investigated using fraud-rate comparison rather than dropped blindly — confirmed as a legitimate business pattern (0% fraud rate vs. 1.21% baseline) and retained.

### **2\. Exploratory Data Analysis**

Key findings:

* **Category** is the strongest fraud signal — fraud rate ranges from 0% (`es_transportation`, `es_food`, `es_contents`, 89.5% of volume) to 95% (`es_leisure`).  
* **Merchant** reinforces this — a small set of merchants show 32–96% fraud rates.  
* **Amount** is strongly discriminative — fraud transactions average \~525 vs. \~32 for legitimate ones (\~16x higher).  
* **Demographics** (age, gender) are weak predictors.  
* **Time** (`step`) shows no meaningful pattern over the simulation period.

### **3\. Feature Engineering**

* Target-encoded `category_risk_score` and `merchant_risk_score` (computed from training set only, to prevent leakage)  
* Log-transformed `amount` to address extreme right-skew (skewness \= 32.37)  
* Customer-relative features (`customer_avg_amount`, `amount_vs_avg_ratio`, `customer_txn_count`)  
* One-hot encoded `age`, `gender`

### **4\. Modeling**

Compared Logistic Regression (baseline) against Random Forest, XGBoost, and LightGBM, all trained on SMOTE-resampled data and evaluated on the untouched test distribution.

| Model | ROC AUC | PR AUC | Train Time |
| ----- | ----- | ----- | ----- |
| **XGBoost** | **0.998** | **0.927** | 19.8s |
| Random Forest | 0.996 | 0.908 | 413.4s |
| LightGBM | 0.991 | 0.905 | 15.2s |
| Logistic Regression | 0.994 | 0.817 | — |

Threshold selected via cost-based analysis (estimated cost of missed fraud ≈ 2,314 vs. manual review cost scenarios of 10/25/75), rather than an arbitrary 0.5 cutoff.

**Final result at selected threshold (0.0035)**: 98.8% recall, 33.7% precision — catching 1,423 of 1,440 fraud cases while flagging 2.4% of legitimate transactions.

### **5\. Business Recommendation**

Full write-up in [`Business_Recommendation_Fraud_Detection_Final.md`](https://claude.ai/chat/Business_Recommendation_Fraud_Detection_Final.md), including:

* Cost-benefit analysis (net benefit \~3.19M, robust across review-cost assumptions)  
* Merchant and category risk watchlist  
* 4-tier review system (auto-block / manual review / light monitoring / no action) — captures 84.7% of fraud from just 1.2% of transaction volume in Tier 1 alone  
* Documented model limitations

---

## **Key Results at a Glance**

* **99% fraud capture rate**, missing only 17 of 1,440 fraud cases in the test set  
* **Estimated net benefit ≈ 3.19M** (moderate cost scenario), staying positive even under conservative assumptions  
* **Tiered review system** reduces review burden by concentrating effort on ambiguous cases rather than treating all flagged transactions identically

---

## **Limitations**

* Model relies heavily on `merchant_risk_score` (88.9% of feature importance) — limited effectiveness for new/infrequent merchants without transaction history. Mitigated with a category-tier fallback, but flagged as the primary limitation.  
* Dataset reflects a simulated 2017 environment — fraud patterns evolve, so periodic retraining on live data would be required before production use.  
* Cost assumptions (4.41x fraud cost multiplier, review cost scenarios) are industry-informed estimates, not this specific institution's actual figures.

Full details in the [business recommendation document](https://claude.ai/chat/Business_Recommendation_Fraud_Detection_Final.md), Section 6\.

---

## **Tools & Libraries**

Python · Pandas · NumPy · Scikit-learn · XGBoost · LightGBM · imbalanced-learn (SMOTE) · Matplotlib · Seaborn

