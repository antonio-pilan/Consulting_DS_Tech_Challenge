# Predictive Maintenance Cost Optimization using Machine Learning

This portfolio project demonstrates the application of data science techniques to optimize the maintenance planning for a fleet of trucks, aiming to significantly reduce operational costs.  The core of the project is the development of a predictive model to identify potential failures in the vehicles' Air Pressure System (APS) before they become critical.

## 1. Business Problem

A third-party transportation company experienced a significant rise in maintenance costs for its truck fleet's air system over the last three years, despite the fleet size remaining relatively constant.  Maintenance costs are structured as follows:

* **Inspection with no defect found:** A cost of **$10** is incurred for the specialized team's inspection time. 
* **Preventive repair (defect caught in time):** A cost of **$25** is charged. 
* **Corrective maintenance (in-operation failure):** A cost of **$500** is incurred, which includes labor, replacement parts, and other logistical issues like a truck breaking down on the road. 

The primary objective is to develop an AI-based system to predict maintenance needs, thereby minimizing expensive corrective actions and optimizing overall maintenance expenditure.

## 2. Data Analysis & Preprocessing

The dataset was provided in two files: `air_system_previous_years.csv` containing historical maintenance data, and `air_system_present_year.csv` with current year data, used for the final model validation.  For contractual reasons, all column names were encoded. 

The target variable, `class`, identifies trucks with defects in the air system (`pos`) versus those with defects in other systems (`neg`). 

### Key Preprocessing Steps:

1.  **Handling Missing Data:** Missing values, denoted as `na` in the dataset,  were imputed with `0`. This decision was based on the assumption that a missing reading implies a truck did not operate on a specific route or a sensor failed to record, making zero a plausible placeholder. This approach was favored over row deletion, which would have led to a significant data loss of approximately 8.28%. Imputation with the mean was deemed unsuitable due to the high standard deviation across features.

2.  **Type Conversion:** All numeric columns, initially interpreted as strings, were converted to their correct numeric types to enable statistical analysis and modeling.

3.  **Feature Scaling (Z-Score Standardization):** To prevent features with larger magnitudes from disproportionately influencing the models, all predictor variables were standardized using the Z-Score method:
    $$z = \frac{x-\mu}{\sigma}$$
    Where, $z$ is the standardized value, $x$ is the original value, $\mu$ is the feature's mean, and $\sigma$ is its standard deviation.

## 3. Modeling and Validation

### Custom Business Metric
Given that the primary goal is cost reduction, a standard technical metric like accuracy would be misleading due to the severe class imbalance (only 1.66% of historical cases are positive for air system failure). Therefore, a custom cost function was developed to evaluate model performance based on its direct financial impact:

* **Cost of a False Positive (FP):** $25 (A conservative estimate, assuming a truck flagged incorrectly still undergoes a preventive repair).
* **Cost of a False Negative (FN):** $500 (The cost of a failure that was not predicted).

The total cost function is: `Total Cost = (FP * $25) + (FN * $500)`. This metric was used as the primary scorer for model selection and hyperparameter tuning.

### Predictive Models Tested
Several classification models were trained and evaluated:
* Decision Trees
* Random Forest
* Extra Trees Classifier
* **XGBoost (Extreme Gradient Boosting)**

### Hyperparameter Optimization
`GridSearchCV` was employed to find the optimal set of hyperparameters for each model, using the custom cost function to identify the most financially sound configuration.

## 4. Results & Conclusion

The **XGBoost** model delivered the best performance, yielding the lowest total maintenance cost.

### Final XGBoost Model Optimization
A standard classification threshold of 0.5 is suboptimal when the cost of a false negative is significantly higher than a false positive. To minimize costly in-operation failures, a more conservative approach was adopted.

The decision **threshold was adjusted to 0.02**. This means any truck with a predicted failure probability greater than 2% is sent for maintenance. This strategy drastically reduces the number of false negatives at the expense of a manageable increase in false positives, leading to substantial overall savings.

With this optimized threshold, the projected maintenance cost on the test data was **$17,900**, a figure significantly lower than any of the previous years' actual costs.

### Predictive Factors (Feature Importance)
The XGBoost model also provides insights into which variables are most predictive of system failure, addressing a key stakeholder question. The feature importance plot below highlights the top 20 factors that contribute most to predicting an air system failure.

## 5. Conclusion & Next Steps

This project successfully demonstrates that a machine learning approach, specifically a fine-tuned XGBoost model, can be a highly effective tool for reducing truck maintenance costs by predicting failures before they occur. 

### Deployment and Monitoring
* **Deployment:** If approved, the model could be deployed in several ways, such as through a Django-based data product or a Streamlit dashboard for management oversight. A simpler, automated solution could involve triggering email alerts to the maintenance team when a truck is flagged. 
* **Monitoring:** Once in production, the model's performance must be continuously monitored against new, incoming data to detect any performance degradation or model drift. 
* **Retraining:** A retraining schedule should be established to ensure the model remains accurate as the fleet's operational patterns and data distributions evolve over time. 
* **Risks and Precautions:** Before full implementation, it is crucial to communicate potential risks to the client, including the possibility of model drift and the critical importance of maintaining high-quality input data for sustained performance. 

This proof-of-concept not only solves a specific business problem but also provides a scalable framework that can be adapted for other maintenance systems within the company.
