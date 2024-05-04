# Wildfire Prediction Using Machine Learning

## Project Overview

This project aims to develop a predictive model for wildfire occurrences in the Bay Area using machine learning techniques. By integrating Forest Fire data from NASA and weather data from the OpenMateo API, our model aspires to accurately predict wildfire likelihood, enhancing efforts in disaster preparedness and environmental protection.

## Data Collection

### Datasets Used
- **NASA FIRMS:** Provides historical wildfire occurrence data, including confidence scores and timestamps.
- **OpenMateo API:** Offers historical weather conditions leading to wildfires and geographical data of forests.

### Initial Findings
- **San Jose Fire Dataset (2018-2023):** Initially considered, this dataset was eventually excluded due to its focus on vegetation fires, which did not align with our forest fire prediction objectives.
- **NASA FIRMS vs. San Jose Data:** The NASA dataset proved to be more comprehensive and relevant for our study compared to the San Jose dataset.

## Methodology

### Approach
- **Binary Classification:** Our initial model version utilizes 5-6 significant weather parameters over one county. We are working towards a multi-classification model to provide nuanced probability scores for wildfire occurrences.
- **Regression:** Predict severity of the wildfire. Severity = 0 for negative samples (not wildfire).

### Model Selection and Insights
Our exploration of machine learning models for predicting wildfire occurrences includes:

- **Logistic Regression:** Initial tests suggested that logistic regression might not be suitable due to its inability to effectively recognize positive class instances.

- **SGD Classifier:** This model showed significant misclassification of positive instances, indicating it was not a good fit for our dataset.

- **Random Forest Classifier:** Showed improved accuracy over previous models but still lacked optimal performance, especially in identifying positive class instances.

- **AdaBoost Classifier:** Demonstrated a significantly higher recall, indicating better performance in catching true positives. It also had a higher F1 Score, suggesting a better balance between precision and recall, which is crucial for our imbalanced dataset.

- **XGBClassifier:** This model showed promise with better F1 and Recall scores compared to AdaBoost, indicating a more balanced performance in classifying both positive and negative classes.

- **TabNet:** State-of-the-art, attention based neural network, for Tabular datasets. Best used on larger data.

### Strategies for Enhancement
1. **Refining Target Value Calculations:** Focus on data from the VIIRS sensor, aiming to include additional sensors and extend the analysis timeframe.
2. **Expanding the Weather Dataset:** Integrate more data on forest cover, canopy characteristics, and tree density. Utilize a centroid-based method for precise weather data collection on targeted forest areas.

## Future Steps

We aim to refine our model's predictive capabilities further, exploring different classification models and integrating a broader set of parameters to achieve a comprehensive prediction tool.
