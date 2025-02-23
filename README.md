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