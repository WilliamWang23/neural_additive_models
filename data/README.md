# 数据集说明

本目录用于存放本地数据集，因体积较大不纳入 Git。请按需自行下载并放置到对应子目录：

- **Credit Card Fraud Detection**：`creditcard.csv`，来自 [Kaggle Credit Card Fraud](https://www.kaggle.com/mlg-ulb/creditcardfraud)
- **California Housing**：可用 `sklearn.datasets.fetch_california_housing`，或放置 `california_housing.csv`
- **COMPAS**：`compas-analysis-master/compas-scores-two-years.csv`，来自 [ProPublica](https://www.propublica.org/datastore/dataset/compas-recidivism-risk-score-data-and-analysis)
- **FICO**：`FICO-Explainable-ML-Challenge-HELOC-Dataset-master/HelocData.csv`，来自 [FICO Challenge](https://community.fico.com/s/explainable-machine-learning-challenge)

详见 `data_utils.py` 中的路径与加载逻辑。
