# Jamboree Education - Linear Regression

This project develops a linear regression model to predict the probability of graduate admission for students applying to top Ivy League programs. Leveraging applicant data such as GRE and TOEFL scores, university rating, SOP/LOR strength, undergraduate GPA, and research experience, the model aims to provide actionable insights to improve admission chances.

---

## Table of Contents

- [Jamboree Education - Linear Regression](#jamboree-education---linear-regression)
  - [Table of Contents](#table-of-contents)
  - [Project Overview](#project-overview)
  - [Problem Definition](#problem-definition)
  - [Dataset Description](#dataset-description)
  - [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
  - [Data Preprocessing](#data-preprocessing)
  - [Model Building](#model-building)
  - [Assumptions \& Model Diagnostics](#assumptions--model-diagnostics)
  - [Model Evaluation](#model-evaluation)
  - [Actionable Insights \& Recommendations](#actionable-insights--recommendations)
    - [Insights:](#insights)
    - [Recommendations:](#recommendations)
  - [Repository Structure](#repository-structure)
  - [How to Run the Project](#how-to-run-the-project)
  - [Tools \& Technologies](#tools--technologies)
  - [License](#license)

---

## Project Overview

Jamboree Education has launched an online feature that estimates students' chances of admission to Ivy League graduate programs. This project:
- Identifies the key factors influencing graduate admission.
- Builds a linear regression model (with alternatives such as Ridge and Lasso) to predict the Chance of Admit.
- Provides insights and recommendations to help students enhance their profiles and guide Jamboree in refining the prediction feature.

---

## Problem Definition

The primary goal is to determine how various academic and qualitative features affect a student's chance of admission. By understanding these relationships, we can:
- Guide students on where to focus (e.g., improving GRE/TOEFL scores or engaging in research).
- Assist Jamboree in offering personalized advice and refining their online prediction tool.

---

## Dataset Description

The dataset (`jamboree_admission.csv`) contains the following columns:

- **Serial No.**: Unique identifier for each record.
- **GRE Score**: GRE score (out of 340).
- **TOEFL Score**: TOEFL score (out of 120).
- **University Rating**: Rating of the university (scale 1–5).
- **SOP**: Statement of Purpose strength (scale 1–5).
- **LOR**: Letter of Recommendation strength (scale 1–5).
- **CGPA**: Undergraduate GPA (scale 1–10).
- **Research**: Research experience (binary: 0 for no, 1 for yes).
- **Chance of Admit**: Target variable (continuous value from 0 to 1).

---

## Exploratory Data Analysis (EDA)

The EDA phase includes:

- **Data Inspection**: Checking the structure, data types, and summary statistics.
  
  ```python
  df = pd.read_csv("data/jamboree_admission.csv")
  print(df.info())
  print(df.describe())
  print(df.isnull().sum())
```

- **Univariate Analysis**: Visualizing distributions using histograms and boxplots for key variables such as GRE, TOEFL, CGPA, and Chance of Admit.
  
  ```python
  sns.histplot(df['GRE Score'], kde=True)
  plt.title('GRE Score Distribution')
  plt.show()
  ```

- **Bivariate Analysis**: Exploring relationships between predictors and the target variable through scatter plots and a correlation heatmap.
  
  ```python
  sns.scatterplot(x=df['GRE Score'], y=df['Chance of Admit'])
  plt.title('GRE Score vs. Chance of Admit')
  plt.show()

  correlation_matrix = df.corr()
  sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
  plt.title('Correlation Matrix')
  plt.show()
  ```

**Key Insights:**
- **High Correlations**: GRE, TOEFL, and CGPA exhibit strong positive correlations with Chance of Admit.
- **Moderate Influence**: Research experience, University Rating, and SOP/LOR strength have moderate correlations.
- **Distribution Observations**: GRE and TOEFL scores follow a near-normal distribution; CGPA is evenly distributed; Chance of Admit is right-skewed.

---

## Data Preprocessing

Steps taken include:

1. **Duplicate Check & Missing Values**: Confirmed no duplicates or missing values exist.
2. **Outlier Treatment**: Outliers for GRE and TOEFL were assessed using boxplots, Z-scores, and IQR methods. As a precaution, capping was applied at the 95th percentile.
3. **Feature Scaling**: Numerical features (GRE, TOEFL, CGPA) were normalized using `StandardScaler`.
  
   ```python
   from sklearn.preprocessing import StandardScaler
   scaler = StandardScaler()
   df[['GRE Score', 'TOEFL Score', 'CGPA']] = scaler.fit_transform(df[['GRE Score', 'TOEFL Score', 'CGPA']])
   ```

4. **Feature Engineering**: Interaction terms (e.g., GRE * Research) were created to capture complex relationships.

---

## Model Building

A linear regression model was built using the Statsmodels library. The dataset was split into training and testing sets, with a constant term added to the predictors.

```python
import statsmodels.api as sm
from sklearn.model_selection import train_test_split

X = df[['GRE Score', 'TOEFL Score', 'University Rating', 'SOP', 'CGPA', 'Research']]
y = df['Chance of Admit']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train = sm.add_constant(X_train)
X_test = sm.add_constant(X_test)

model = sm.OLS(y_train, X_train).fit()
print(model.summary())
```

**Model Summary:**
- Adjusted R² ≈ 0.812 (81.2% of the variance explained).
- Significant predictors: GRE Score, TOEFL Score, CGPA, and Research.
- University Rating and SOP were not statistically significant at the 0.05 level.

**Alternative Models:**  
Ridge and Lasso regressions were also evaluated to handle multicollinearity and feature selection.

```python
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import mean_squared_error
import numpy as np

# Ridge Regression
ridge = Ridge(alpha=1.0)
ridge.fit(X_train, y_train)
y_pred_ridge = ridge.predict(X_test)
ridge_rmse = np.sqrt(mean_squared_error(y_test, y_pred_ridge))

# Lasso Regression
lasso = Lasso(alpha=0.1)
lasso.fit(X_train, y_train)
y_pred_lasso = lasso.predict(X_test)
lasso_rmse = np.sqrt(mean_squared_error(y_test, y_pred_lasso))

print(f"Ridge RMSE: {ridge_rmse}, Lasso RMSE: {lasso_rmse}")
```

---

## Assumptions & Model Diagnostics

1. **Multicollinearity (VIF Check):**
   ```python
   from statsmodels.stats.outliers_influence import variance_inflation_factor

   vif_data = pd.DataFrame()
   vif_data["feature"] = X_train.columns
   vif_data["VIF"] = [variance_inflation_factor(X_train.values, i) for i in range(X_train.shape[1])]
   print(vif_data)
   ```
   All VIF values were below 5 after iterative removal of high-VIF predictors.

2. **Residual Analysis:**
   - **Mean of Residuals:** Approximately zero.
   - **Residual Plot:** No distinct pattern, supporting linearity.
   - **Homoscedasticity Check:** Scatter plot of fitted values vs. √|residuals| showed no funnel-shaped pattern.
   - **Normality of Residuals:** Q-Q plot indicates that residuals follow a normal distribution.

   ```python
   residuals = model.resid
   print(f"Mean of residuals: {np.mean(residuals)}")

   sns.scatterplot(x=model.fittedvalues, y=residuals)
   plt.title('Residual Plot')
   plt.show()

   import scipy.stats as stats
   stats.probplot(residuals, dist="norm", plot=plt)
   plt.show()
   ```

---

## Model Evaluation

Using the test set, the following metrics were obtained:
- **MAE:** ~0.0431
- **RMSE:** ~0.0616
- **R²:** ~0.8144

These metrics indicate that the model explains approximately 81.44% of the variance in the target variable and performs well in predicting admission chances.

---

## Actionable Insights & Recommendations

### Insights:
- **Key Predictors:** GRE, TOEFL, and CGPA have the strongest influence on admission chances.
- **Research Experience:** Provides a moderate boost to admission probability.
- **Other Factors:** University Rating and SOP/LOR strength, while less significant, still contribute to the overall profile.
- **Alternative Models:** Ridge regression shows improved performance under multicollinearity, and Lasso aids in feature selection.

### Recommendations:
1. **For Students:**
   - **Improve GRE/TOEFL Scores:** Focus on test preparation as these scores are critical predictors.
   - **Enhance Academic Profile:** A strong CGPA is highly correlated with admission success.
   - **Gain Research Experience:** Engage in research projects to boost admission probabilities.
   - **Strengthen SOP/LOR:** While not the top predictors, well-crafted SOPs and LORs can provide an edge.

2. **For Jamboree Education:**
   - **Refine the Admission Prediction Tool:** Incorporate additional features (e.g., extracurricular activities) to enhance predictive accuracy.
   - **Personalized Guidance:** Utilize the model to provide tailored advice for students based on their profiles.
   - **Dashboard Integration:** Develop an interactive dashboard for real-time predictions and actionable recommendations.

---

## Repository Structure

```
Jamboree_Education_Linear_Regression/
│
├── README.md                            # Project overview and documentation
├── data/
│   └── jamboree_admission.csv           # Dataset for the project
├── notebooks/
│   └── jamboree_linear_regression.ipynb   # Jupyter Notebook containing EDA, model building, and diagnostics
├── outputs/
│   └── images/                          # Visualizations and plots generated during analysis
├── requirements.txt                     # List of Python dependencies
└── LICENSE                              # Project license (MIT License)
```

---

## How to Run the Project

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/GopalGB/Jamboree_Education_Linear_Regression.git
   ```
2. **Navigate to the Project Directory:**
   ```bash
   cd Jamboree_Education_Linear_Regression
   ```
3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
4. **Open the Notebook:**
   - Launch Jupyter Notebook or open the notebook on Google Colab.
   - Open `notebooks/jamboree_linear_regression.ipynb` and run the cells sequentially.
5. **Review Outputs:**
   - Check the `outputs/images/` folder for visualizations.
   - Export the notebook as a PDF if needed for reporting.

---

## Tools & Technologies

- **Python:** Core programming language.
- **Pandas & NumPy:** Data manipulation and numerical operations.
- **Matplotlib & Seaborn:** Data visualization.
- **Statsmodels & Scikit-learn:** Model building and evaluation.
- **Jupyter Notebook/Google Colab:** Interactive development environment.

---

## License

This project is licensed under the MIT License. See the [LICENSE](./LICENSE) file for more details.
```


