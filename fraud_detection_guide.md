# End-to-End Credit Card Fraud Detection: A Complete ML Project Guide

This guide walks through building a full fraud detection system, using your uploaded `creditcard.csv` as the running example. Quick facts about your file, confirmed directly:

- **284,807 transactions**, 31 columns: `Time`, `Amount`, `Class` (0 = legit, 1 = fraud), and `V1`–`V28`
- **Only 492 frauds — 0.17% of the data.** This is the single most important fact about the project. Every design decision below exists because of this imbalance.
- `V1`–`V28` are already **PCA-transformed** (this is the real, well-known Kaggle "Credit Card Fraud Detection" dataset from a 2013 European cardholder sample). That means the original feature names (merchant, location, card type, etc.) were anonymized for privacy — you're working with anonymized principal components, not raw business features. `Time` and `Amount` are the only two untouched, human-readable columns.

Keep that imbalance number in your head throughout: a model that predicts "not fraud" for every single transaction would be **99.83% accurate** and completely useless. Accuracy is basically a meaningless metric here — most of this guide is really about working around that one fact.

---

## 1. Data Collection / Sourcing

You already have data, but here's the context for how projects like this normally start, and what to do if you want to extend it:

- **Your dataset**: this is the Kaggle "Credit Card Fraud Detection" dataset, originally released by the Machine Learning Group at Université Libre de Bruxelles (ULB). It covers transactions by European cardholders over two days in September 2013.
- **`Time`** is seconds elapsed since the first transaction in the dataset (not a timestamp/clock time).
- **`Amount`** is the transaction amount in the original currency.
- Because the original features (merchant category, IP, device, location, etc.) were sensitive, they were transformed with **PCA** into `V1`–`V28`. This is common in publicly released fraud datasets — you'll see this pattern often. The tradeoff: you get privacy-safe features, but you lose the ability to do intuitive feature engineering ("was this a foreign transaction?") on those columns, since they're abstract components, not real-world concepts.
- If you wanted to extend this project with real, non-anonymized data: Kaggle also has an "IEEE-CIS Fraud Detection" dataset (larger, more raw features) and a synthetic "PaySim" mobile-money dataset that's good for learning feature engineering since its columns are human-readable.
- In an actual industry setting, data comes from a company's own transaction logs, joined with device/session data, and often labeled by chargebacks or manual investigator review (with a lag of days-to-weeks before a "confirmed fraud" label exists — a nuance that matters a lot for production systems, more on that in Deployment).

**Practical tip:** always note *how* labels were generated. A label of "fraud" based on customer chargeback disputes is different from one based on bank-confirmed fraud — the former can include some false disputes. This dataset's labels are ground truth, so you don't need to worry about label noise here, but you should ask this question of any dataset you use in the future.

---

## 2. Environment & Tools Setup

Recommended stack for this project:

| Purpose | Tool |
|---|---|
| Data manipulation | `pandas`, `numpy` |
| Visualization | `matplotlib`, `seaborn`, `plotly` (optional, for interactive EDA) |
| Modeling | `scikit-learn` (logistic regression, random forest, isolation forest), `xgboost` or `lightgbm` (gradient boosting) |
| Imbalanced data handling | `imbalanced-learn` (SMOTE, undersampling) |
| Model explainability | `shap` |
| Experiment tracking | `mlflow` (optional but good practice) |
| Deployment | `FastAPI` or `Flask` for serving, `Docker` for containerization |
| Notebook environment | Jupyter Lab or Google Colab |

```bash
pip install pandas numpy scikit-learn matplotlib seaborn imbalanced-learn xgboost shap mlflow fastapi uvicorn --break-system-packages
```

**Tip:** pin your versions in a `requirements.txt` from day one — reproducibility matters more in fraud detection than most domains, because you'll retrain often as fraud patterns drift.

---

## 3. Exploratory Data Analysis (EDA)

Goals of this stage: understand the imbalance, understand the distributions of `Time` and `Amount`, and sanity-check that `V1`–`V28` don't have obvious data-quality problems.

Key things to check, in order:

1. **Class balance** (you already know this: 284,315 vs. 492). Always compute and print this first in the notebook — it should visually anchor every later modeling decision.
2. **Missing values** — check `df.isnull().sum()`. This dataset happens to be clean (no missing values), which is unusual for real-world fraud data; call this out in your write-up as a simplification versus production data.
3. **Distribution of `Amount`** — fraud transactions are often (but not always) clustered at particular amount ranges (e.g., unusually small "test" transactions or unusually large ones). Plot `Amount` separately for fraud vs. non-fraud using overlaid histograms or box plots (log-scale the axis — `Amount` is heavily right-skewed).
4. **Distribution of `Time`** — plot transaction volume over time, and again split by class. Fraud sometimes clusters at odd hours (e.g., overnight) when legitimate volume is low.
5. **Correlation with `Class`** — even though `V1`–`V28` are anonymized, you can still rank them by correlation (or mutual information) with `Class` to see which ones separate fraud most cleanly. In this dataset, `V14`, `V4`, `V11`, `V12`, and `V17` are typically among the most discriminative — worth confirming on your copy rather than taking this as given.
6. **Pairwise visuals** — a 2D scatter or density plot of the top 2 most-correlated `V` features, colored by class, is often the single most convincing EDA chart for a fraud project because it visually shows how separable the classes are (or aren't).

**Best practice:** Always do EDA on a **held-out-aware** basis — meaning, glance at overall distributions freely, but don't use statistics *computed from the full dataset* (like scaling parameters) inside your feature engineering pipeline. Save that for after you've split into train/test (see Section 5). It's an easy and common mistake to compute a global mean/std for scaling before splitting, which quietly leaks information from the test set into training.

---

## 4. Data Preprocessing

For this dataset specifically:

- **Scaling**: `V1`–`V28` are already roughly on a similar scale (PCA outputs typically are), but `Time` and `Amount` are not — `Amount` can range from 0 to tens of thousands, while `Time` is in seconds over ~172,800 seconds (48 hours). Apply `StandardScaler` or `RobustScaler` to `Time` and `Amount` only. `RobustScaler` (uses median/IQR) is a good choice here since `Amount` has extreme outliers that would distort a mean-based scaler.
- **Duplicates**: check for and consider dropping exact duplicate rows — this dataset is known to have some (~1,000), and duplicate fraud rows in particular can artificially inflate a model's apparent performance if some duplicates leak across your train/test split.
- **Feature `Time`**: consider transforming it into "seconds since midnight" (`Time % 86400`) to capture time-of-day cyclicality instead of raw elapsed seconds, which mostly just encodes "which of the two days did this happen on."
- **No categorical encoding needed** here since everything is already numeric — but note that in most real fraud datasets, you'd have categorical fields (merchant category, country, device type) needing one-hot or target encoding at this stage.

---

## 5. Train/Test Split — the Step Most People Get Wrong

This is the most important preprocessing decision in the whole project.

- Use a **stratified split** (`train_test_split(..., stratify=y)`) so both your train and test sets preserve the ~0.17% fraud ratio. Without stratification, a random split could easily leave your test set with too few frauds to evaluate meaningfully, or your train set with almost none to learn from.
- Because `Time` exists, consider a **time-based split** instead of a random one for a more realistic evaluation: train on the first ~70% of transactions by time, test on the later ~30%. This simulates how the model will actually be used — predicting on *future* transactions — and avoids the subtly optimistic bias you get from randomly shuffling time-ordered data (fraud patterns can look similar between adjacent transactions).
- **Never touch the test set again** until final evaluation. Set aside a third split (validation) for tuning, or use cross-validation on the training set only.
- If you use resampling techniques like SMOTE (next section), **apply them only to the training data, after the split** — never to the test set, and never before splitting. Otherwise you leak synthetic copies of test-adjacent fraud patterns into training.

---

## 6. Handling Class Imbalance

This is the core technical challenge. Approaches, roughly from simplest to most sophisticated:

1. **Class weighting** (easiest, often surprisingly effective): most `sklearn` models accept `class_weight='balanced'`, which tells the model to penalize mistakes on the minority class more heavily. Start here — it's a one-line change and a strong baseline.
2. **Undersampling** the majority class: randomly drop most of the "not fraud" rows so the classes are closer to balanced. Simple, but you throw away a lot of legitimate data and information about what "normal" looks like.
3. **Oversampling / SMOTE** (Synthetic Minority Oversampling Technique): generates synthetic fraud examples by interpolating between existing fraud points in feature space, rather than just duplicating them. Available via `imbalanced-learn`. Good middle ground, but can create unrealistic synthetic points if the minority class is very sparse or has outliers — inspect the synthetic points, don't apply blindly.
4. **Anomaly detection framing**: instead of treating this as a two-class classification problem, treat "fraud" as a rare anomaly and use unsupervised/semi-supervised methods like `IsolationForest` or `AutoEncoder`-based reconstruction error, trained mostly on normal transactions. This is a legitimate alternative paradigm worth trying and comparing against supervised approaches, especially since fraud patterns evolve and a model trained only on past fraud examples can miss new fraud "shapes."
5. **Combining approaches**: e.g., moderate undersampling of the majority class combined with SMOTE on the minority class, or class weighting combined with a boosting algorithm's own imbalance handling (XGBoost's `scale_pos_weight` parameter).

**Practical tip:** try more than one of these and compare — there's no universally "correct" choice, and the best one depends on which error type (missed fraud vs. false alarm) matters more for your use case (see Section 8).

---

## 7. Feature Engineering

Given the PCA columns are already abstracted, most of the engineering value here comes from `Time` and `Amount`, plus interactions:

- `Hour_of_day = (Time % 86400) // 3600` — captures time-of-day pattern.
- `Amount_log = log1p(Amount)` — compresses the long right tail, often helps linear/logistic models.
- Interaction features between top-correlated `V` columns (e.g., `V14 * V4`) — tree-based models can find these on their own, but linear models benefit from you engineering them explicitly.
- **In a real production system** (worth mentioning even though not applicable to this static file), the highest-value features are usually **behavioral/aggregate** ones computed per card or per user: transaction velocity (number of transactions in the last hour/day), deviation from the cardholder's typical spend amount, distance from the cardholder's usual location, time since the card's last transaction, etc. These require a stateful feature pipeline (a "feature store") rather than static columns — mention this if you extend the project or discuss it in an interview/report context, since it's a common follow-up question.

**Best practice:** always fit any feature-engineering transformer (scalers, encoders) on the training set only, then apply (`.transform()`, not `.fit_transform()`) to validation/test — this is the same leakage principle as Section 5.

---

## 8. Choosing Evaluation Metrics (before you choose a model)

Decide this *before* modeling, because it determines which model and threshold you'll pick later.

- **Accuracy: don't use it.** As established, 99.83% accuracy is trivial and meaningless here.
- **Precision**: of the transactions you flagged as fraud, what fraction actually were fraud? Low precision = you're annoying customers and burning investigator time on false alarms.
- **Recall (a.k.a. sensitivity)**: of all actual frauds, what fraction did you catch? Low recall = fraud slips through, direct financial loss.
- **There is an unavoidable trade-off** between precision and recall, controlled by your decision threshold. A bank blocking a legitimate transaction is annoying; a bank missing a fraudulent one is costly — the right balance is a business decision, not a purely technical one. Make sure to state explicitly, in whatever write-up you produce, which side you're optimizing for and why.
- **F1 score**: harmonic mean of precision and recall — useful as one summary number, but can hide which side of the trade-off you're on, so report precision and recall separately too.
- **PR-AUC (Precision-Recall Area Under Curve)**: more informative than ROC-AUC for heavily imbalanced problems, because ROC-AUC can look deceptively good even for a weak model when negatives vastly outnumber positives. **Use PR-AUC as your primary ranking metric for model comparison in this project.**
- **Confusion matrix**: always look at raw counts (true positives, false positives, false negatives, true negatives), not just derived percentages — with only 492 frauds, a handful of misclassifications swings your recall a lot, and the raw numbers keep that visible.

---

## 9. Model Selection & Training

Suggested progression — build a simple baseline first, then increase complexity, comparing each against the last:

1. **Logistic Regression** with `class_weight='balanced'` — fast, interpretable, a genuinely reasonable production baseline for fraud detection in many companies (interpretability matters when you need to explain to a regulator or a customer why a transaction was blocked).
2. **Random Forest** — handles nonlinearity and feature interactions with minimal tuning, robust to outliers, gives you feature importances for free.
3. **Gradient Boosting (XGBoost or LightGBM)** — typically the strongest performer on tabular data like this; use `scale_pos_weight` (ratio of negative to positive class count) to handle imbalance natively.
4. **Isolation Forest / anomaly detection** — as an unsupervised comparison point (Section 6, approach 5).
5. *(Optional, more advanced)* a shallow **neural network (autoencoder)** trained to reconstruct normal transactions, flagging high reconstruction-error transactions as anomalies.

**Practical workflow tips:**
- Use `Pipeline` from `sklearn` to chain scaling → resampling → model into one object — this prevents leakage automatically and makes cross-validation clean.
- Use **stratified k-fold cross-validation** on the training set for model comparison and hyperparameter search, not a single train/validation split — with only ~350 frauds in a training set, a single split is noisy.
- Log every experiment (model type, hyperparameters, metrics) — even a simple CSV log works, or use `mlflow` for a more structured approach. You'll run many variants, and losing track of what you tried is a common time-waster.

---

## 10. Hyperparameter Tuning

- Use `RandomizedSearchCV` or `Optuna` rather than exhaustive `GridSearchCV` for tree-based models — the hyperparameter space (tree depth, learning rate, number of estimators, min samples per leaf, regularization) is large enough that grid search becomes very slow for little extra benefit.
- **Tune the decision threshold separately from the model's hyperparameters.** Most classifiers output a probability, and the default 0.5 cutoff is rarely optimal for a problem this imbalanced. After training, plot precision and recall across thresholds (a precision-recall curve) and pick the threshold that matches your business trade-off from Section 8 — this is often a bigger performance lever than further model tuning.
- Optimize your search around **PR-AUC**, not accuracy, as the scoring metric passed to your search tool.

---

## 11. Model Evaluation

On your held-out test set (only look at this once, at the end):

- Report precision, recall, F1, PR-AUC, and the confusion matrix.
- Plot the precision-recall curve, not just the ROC curve (include ROC too, but caveat it given the imbalance).
- **Segment your errors**: look specifically at the false negatives (missed frauds) — are they clustered in a particular amount range or time window? This often reveals a blind spot worth addressing with more feature engineering or a different resampling strategy.
- Use **SHAP** values to explain a handful of individual predictions, especially any false positives/negatives — useful both for debugging the model and for producing human-readable justifications, which matter a lot in fraud (you may need to explain to a customer or regulator why a transaction was flagged).

---

## 12. Deployment

A few real-world considerations that go beyond just "export the model":

- **Serving**: wrap the trained model in a lightweight API (FastAPI is a good, modern choice) that accepts a transaction's features and returns a fraud probability plus a flag based on your chosen threshold.
- **Latency**: fraud scoring for card transactions typically needs to happen in well under a second, since it can block the transaction in real time — keep the model lightweight enough (or serve it efficiently) to meet this.
- **Monitoring for drift**: fraud patterns change constantly as fraudsters adapt. Track your model's precision/recall on a rolling basis against newly confirmed labels, and set up alerts if performance degrades — this is arguably more important in fraud than in most ML domains.
- **Label delay problem**: in production, you often don't know if a transaction was truly fraudulent until days or weeks later (after a chargeback or investigation). This means your "ground truth" for monitoring lags behind your predictions — a nuance worth mentioning if you write this project up, since it's a step beyond what a static Kaggle CSV can teach you.
- **Retraining cadence**: decide upfront (e.g., weekly or monthly) how often the model gets retrained on fresh data, and version your models so you can roll back a regression.
- **Containerize** with Docker for reproducible deployment, and consider a simple CI check that reruns your evaluation metrics against a fixed test set before any new model version is promoted.

---

## Suggested Project Structure

```
fraud-detection/
├── data/
│   └── creditcard.csv
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_preprocessing.ipynb
│   └── 03_modeling.ipynb
├── src/
│   ├── preprocessing.py
│   ├── train.py
│   ├── evaluate.py
│   └── serve.py          # FastAPI app
├── models/
│   └── model_v1.pkl
├── requirements.txt
└── README.md
```

## Summary Checklist

- [ ] Confirmed class imbalance (0.17% fraud) and chose PR-AUC as primary metric
- [ ] Did EDA on `Time`, `Amount`, and per-class distributions
- [ ] Scaled `Time`/`Amount`, checked for duplicates
- [ ] Split data with stratification (or time-based split) **before** any resampling
- [ ] Tried class weighting, SMOTE, and/or anomaly detection framing
- [ ] Trained baseline (logistic regression) → stronger model (XGBoost/LightGBM)
- [ ] Tuned decision threshold based on precision/recall trade-off, not just default 0.5
- [ ] Evaluated on untouched test set with confusion matrix + PR curve
- [ ] Used SHAP to explain key predictions/errors
- [ ] Planned for latency, monitoring, and label-delay in a deployment scenario

Would you like me to turn any specific stage — e.g., the EDA, the SMOTE + XGBoost training pipeline, or the FastAPI serving code — into a runnable Python script against your actual `creditcard.csv`?
