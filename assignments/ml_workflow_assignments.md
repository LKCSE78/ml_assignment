# ML Workflow Assignment 

## Task 1

### **Label Column**

**`repeat_purchase_flag`**

This column is the **target variable (dependent variable)** because it directly captures the business objective: predicting whether a customer will make a repeat purchase within 30 days. In supervised machine learning, the label is the outcome the model is trained to predict, and here it is clearly defined as a binary classification problem (1 = repeat purchase, 0 = no repeat purchase). All other features should be used to predict this variable without including any information that is only available after the event occurs.

---

### **Data Leakage Column**

**`discount_used_on_repeat_order`**

This column introduces **data leakage** because it contains information that is only available *after* the repeat purchase has already occurred. Specifically, a discount can only be applied if a repeat order actually happens, which makes this feature indirectly reveal the target variable. Including this column during training would allow the model to “cheat” by learning patterns that would not exist in real-world prediction scenarios. As a result, the model may show artificially high accuracy during training and testing but will fail in production when such future information is not available at prediction time.

---

## Task 2

Before directly training a **gradient boosting model**, several foundational steps in the machine learning workflow should have been completed to ensure robustness, reliability, and interpretability of the model.

---

### **Step 1: Exploratory Data Analysis (EDA)**

EDA is a critical first step that involves deeply understanding the dataset before applying any modeling techniques. This includes:

* **Understanding distributions:** Analyze how features like `order_count_last_90d`, `avg_order_value`, and `days_since_last_order` are distributed (e.g., skewness, spread).
* **Identifying outliers:** Detect abnormal values that could negatively impact model performance.
* **Checking class imbalance:** Determine whether the target variable (`repeat_purchase_flag`) is balanced or skewed toward one class, which may require resampling techniques.
* **Feature relationships:** Use correlation analysis and visualization (heatmaps, scatter plots) to understand how input variables relate to the target.
* **Business insights:** Gain intuition about customer behavior patterns (e.g., customers with recent purchases may be more likely to repeat).

**Why this matters:**
Skipping EDA can lead to poor feature selection, unnoticed data quality issues, and inappropriate model choices. It ensures that the model is built on a solid understanding of the data rather than assumptions.

---

### **Step 2: Data Preprocessing (Cleaning & Feature Engineering)**

Raw data is rarely suitable for direct input into machine learning models. Preprocessing ensures the data is clean, consistent, and meaningful. Key activities include:

* **Handling missing values:** Fill, remove, or impute missing data appropriately.
* **Removing data leakage features:** Exclude columns like `discount_used_on_repeat_order` to prevent unrealistic model performance.
* **Feature scaling:** Normalize or standardize numerical features if required by the algorithm (though tree-based models are less sensitive).
* **Feature engineering:** Create new meaningful features (e.g., customer recency-frequency metrics) that improve predictive power.
* **Train-test split:** Separate data into training and testing sets to evaluate model generalization.

**Why this matters:**
Without proper preprocessing, the model may learn incorrect patterns, overfit the training data, or fail to generalize to new data. It also ensures that the input data reflects real-world conditions where predictions will be made.

---


