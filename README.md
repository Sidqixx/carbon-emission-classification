# Classifying CO₂ Emission Levels Using Machine Learning

## Project Overview

This project explores whether economic and demographic indicators can be used to classify countries into predefined CO₂ emission categories.

Two supervised machine learning models are evaluated:

- Decision Tree
- Random Forest

The project focuses on a leakage-aware machine learning workflow, where direct CO₂ emission measurements are excluded from the model features because the target category is derived from CO₂ emissions.

---

## Objectives

- Explore the relationship between economic and demographic indicators and CO₂ emission levels.
- Classify observations into Low, Medium, and High emission categories.
- Compare Decision Tree and Random Forest performance.
- Identify which economic and demographic variables contribute most to the model.
- Evaluate model performance using accuracy, precision, recall, F1-score, and confusion matrices.
- Demonstrate a machine learning workflow that avoids direct target leakage.

---

## Dataset

The project combines two datasets:

### 1. CO₂ Emissions Dataset

Contains country-level CO₂ emission measurements, including:

- Country
- Date
- Kilotons of CO₂
- Metric Tons Per Capita
- Other emission-related information

### 2. Economic Impact on Emissions Dataset

Contains economic and demographic indicators, including:

- Country
- GDP per Capita
- Fuel Price
- Population Density
- Urbanization Rate
- CO₂ Emissions

The datasets are merged using the `Country` column.

---

## Target Definition

The target variable is `Emission_Category`, created from the `CO2_Emissions` value using the following predefined thresholds:

| CO₂ Emissions | Category |
|---|---|
| `< 150` | Low |
| `150 – < 300` | Medium |
| `≥ 300` | High |

The resulting target distribution contains:

| Category | Records | Proportion |
|---|---:|---:|
| High | 12,240 | 44.98% |
| Medium | 9,060 | 33.30% |
| Low | 5,910 | 21.72% |
| **Total** | **27,210** | **100%** |

---

## Feature Selection and Target Leakage

A key methodological issue identified during the project was target leakage.

Originally, `CO2_Emissions` was included as a model feature even though `Emission_Category` was directly created from the same variable. This allowed the model to access information that was directly related to the target.

To address this issue, the final model excludes:

- `CO2_Emissions`
- `Kilotons of Co2`
- `Metric Tons Per Capita`

These variables are either directly used to construct the target or represent direct CO₂ emission measurements that could act as proxies for the target.

`Country`, `Region`, and `Date` are also excluded from the final feature set so that the models focus specifically on economic and demographic indicators.

### Final Features

The final models use four predictors:

1. `GDP_per_Capita`
2. `Fuel_Price`
3. `Population_Density`
4. `Urbanization_Rate`

This results in a more defensible modeling question:

> Can economic and demographic indicators alone classify observations into predefined CO₂ emission categories?

---

## Data Preparation

The data preparation workflow consists of:

1. Loading the two datasets.
2. Creating the `Emission_Category` target.
3. Merging the datasets using `Country`.
4. Checking for duplicate rows.
5. Checking missing values.
6. Filling numeric missing values using the median.
7. Filling categorical missing values using the mode.
8. Selecting only non-emission economic and demographic predictors.
9. Splitting the dataset into training and testing sets.

The dataset is divided using an 80/20 train-test split with `random_state=42` and stratification based on the target variable.

---

## Machine Learning Workflow

```text
Raw Datasets
     │
     ▼
Data Loading
     │
     ▼
Target Creation
     │
     ▼
Dataset Merge
     │
     ▼
Data Cleaning
     │
     ▼
Feature Selection
     │
     ├── GDP per Capita
     ├── Fuel Price
     ├── Population Density
     └── Urbanization Rate
     │
     ▼
Train-Test Split
     │
     ├───────────────┐
     ▼               ▼
Decision Tree   Random Forest
     │               │
     └───────┬───────┘
             ▼
       Model Evaluation
             │
             ├── Accuracy
             ├── Precision
             ├── Recall
             ├── F1-Score
             └── Confusion Matrix
```

---

## Models

### Decision Tree

Configuration:

```python
DecisionTreeClassifier(
    max_depth=2,
    min_samples_split=20,
    random_state=42
)
```

### Random Forest

Configuration:

```python
RandomForestClassifier(
    n_estimators=50,
    max_depth=5,
    min_samples_split=10,
    random_state=42
)
```

The models are intentionally kept relatively simple because the purpose of the project is to compare baseline tree-based classification approaches rather than perform extensive hyperparameter optimization.

---

## Model Evaluation

The final evaluation considers:

- Accuracy
- Weighted Precision
- Weighted Recall
- Weighted F1-Score
- Per-class Precision
- Per-class Recall
- Per-class F1-Score
- Confusion Matrix

This is important because accuracy alone does not fully describe the model behavior when predictions are unevenly distributed across classes.

---

## Results

After removing direct emission variables from the model features, model performance decreased substantially compared with the original experiment.

### Final Model Performance

| Model | Accuracy | Weighted Precision | Weighted Recall | Weighted F1 |
|---|---:|---:|---:|---:|
| Decision Tree | ~0.46 | ~0.67 | ~0.46 | ~0.30 |
| Random Forest | ~0.49 | ~0.72 | ~0.49 | ~0.36 |

The Random Forest model produced approximately 49% accuracy in the final experiment.

However, the confusion matrix showed that the model heavily favored the High category. For the Random Forest experiment:

```text
                 Predicted
                 High  Low  Medium
Actual High      2433   0     15
Actual Low       1095  87      0
Actual Medium    1685   0    127
```

This indicates that the overall accuracy is largely driven by the model's ability to identify the High category, while its ability to distinguish Low and Medium observations remains limited.

---

## Feature Importance

The Random Forest model identified the following features as the most influential in the final experiment:

| Feature | Importance |
|---|---:|
| Population Density | 0.203 |
| Urbanization Rate | 0.180 |
| Fuel Price | 0.153 |
| GDP per Capita | 0.147 |

These four economic and demographic indicators collectively account for the majority of the model's feature importance in the experiment.

The result suggests that population structure, urbanization, fuel pricing, and economic conditions contain some information related to the predefined emission categories. However, the classification results indicate that these variables alone do not provide strong enough separation between all three categories.

---

## Key Findings

### 1. Direct emission variables caused target leakage

The original experiment achieved approximately 92% accuracy because `CO2_Emissions` was included among the model features while also being used to construct `Emission_Category`.

After removing this variable and other direct emission measurements, the performance dropped substantially.

This demonstrates the importance of checking whether model predictors contain information derived directly from the target.

### 2. Economic and demographic indicators provide limited class separation

Using only:

- GDP per Capita
- Fuel Price
- Population Density
- Urbanization Rate

resulted in approximately 49% Random Forest accuracy.

The model was able to identify High-emission observations relatively well but struggled to distinguish Low and Medium categories.

### 3. Accuracy alone can be misleading

Although the Random Forest achieved approximately 49% accuracy, the class-level results showed very low recall for Low and Medium observations.

Therefore, the model should not be interpreted as a strong three-class classifier based on accuracy alone.

---

## Limitations

### 1. Predefined emission thresholds

The Low, Medium, and High categories are created using fixed thresholds of 150 and 300. These thresholds may not represent a universally applicable classification standard.

### 2. Limited predictor set

The final model uses only four economic and demographic indicators. Other potentially relevant factors, such as energy consumption structure, industrial activity, transportation patterns, energy sources, and environmental policies, are not included in the final model.

### 3. Classification performance

After removing direct emission measurements, the models show limited ability to distinguish between Low, Medium, and High categories.

The Random Forest tends to predict the High category for a large proportion of observations, resulting in poor recall for the Low and Medium classes.

### 4. Dataset structure

The datasets are merged using country names, and the resulting dataset contains repeated observations across countries and dates. The final experiment uses a standard stratified random train-test split rather than a country-level or time-based validation strategy.

Therefore, the results should be interpreted as an exploratory machine learning experiment rather than a production-ready predictive system.

---

## Project Structure

```text
carbon-emission-classification/
│
├── data/
│       ├── Carbon_(CO2)_Emissions_by_Country.csv
│       └── economic_impact_on_emissions_modified.csv
│
├── notebooks/
│   └── Carbon_Emission_Classification_Using_Random_Forest_and_Decision_Tree.ipynb
│
├── README.md
├── requirements.txt
└── .gitignore
```

---

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Jupyter Notebook

### Machine Learning

- Decision Tree Classifier
- Random Forest Classifier

---

## Project Type

Exploratory Machine Learning / Classification

The project emphasizes data preparation, leakage detection, feature selection, model evaluation, and interpretation rather than maximizing predictive performance.
