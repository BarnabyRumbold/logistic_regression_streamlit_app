# Logistic Regression App for Predicting Patient Attendances

## Overview
This application leverages logistic regression to predict patient attendances based on warehouse data for Kent and Medway ICB. The model analyzes historical trends and patient demographics to provide insightful predictions that assist healthcare professionals in resource planning and operational efficiency.

## Features
- **Predict patient attendances** based on structured warehouse data.
- **Leverage logistic regression**, a robust statistical model for classification tasks.
- **Use Markdown for documentation and reporting**, ensuring transparency and readability.
- **Optimized for Kent and Medway ICB**, utilizing region-specific data for accurate forecasting.

## Data Sources
The model uses data extracted from the Kent and Medway Integrated Care Board's warehouse, including:
- **Demographic data** (age, gender, location, etc.).
- **Historical attendance records** (date, time, and frequency of visits).
- **Medical history indicators** (chronic conditions, prior hospitalizations, etc.).
- **External factors** (weather, public holidays, etc.).

## Installation
Ensure you have Python installed along with the required dependencies. Clone the repository and install dependencies via pip:

```bash
# Clone the repository
git clone https://github.com/your-repo/logistic-regression-app.git
cd logistic-regression-app

# Install dependencies
pip install -r requirements.txt
```

## Usage
Run the application using the command below:

```bash
python app.py
```

You can provide new datasets for inference by modifying the `data/input.csv` file.

## Model Explanation
Logistic regression is a statistical method used for binary classification. In this application, it predicts whether a patient will attend an appointment (1) or not (0) based on historical data.

### Formula:
\[
P(Y=1) = \frac{e^{(\beta_0 + \beta_1X_1 + \beta_2X_2 + ... + \beta_nX_n)}}{1 + e^{(\beta_0 + \beta_1X_1 + \beta_2X_2 + ... + \beta_nX_n)}}
\]

Where:
- \( Y \) is the attendance outcome (1 = attended, 0 = missed).
- \( X_1, X_2, ..., X_n \) are input features from the dataset.
- \( \beta_0 \) is the intercept, and \( \beta_1, \beta_2, ..., \beta_n \) are coefficients learned during training.

## Results & Reporting
After running the model, the app generates:
- **Visualisations in a Streamlit App**, summarizing key insights, dependent on different demographics.


## Contributing
Contributions are welcome! To contribute:
1. Fork the repository.
2. Create a feature branch (`git checkout -b feature-name`).
3. Commit your changes (`git commit -m 'Add feature'`).
4. Push to the branch (`git push origin feature-name`).
5. Create a pull request.


## Contact
For inquiries or collaboration, please contact barnabyrumbold@hotmail.com.
