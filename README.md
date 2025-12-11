# Project Description

This project demonstrates a complete, end-to-end **Credit Risk Modelling** pipeline built using statistical methods and machine learning. It uses two real-world credit datasets consisting of over **51,000 customer records** and more than **80 credit bureau and behavioral features**. The goal is to classify customers into distinct credit risk categories (P1–P4) and help financial institutions make data-driven lending decisions.

The data includes critical credit attributes such as **trade-line history**, **delinquencies**, **enquiries**, **credit utilization**, **product usage flags**, and **time-based credit performance indicators**. These features enable robust modelling of borrower behaviour and accurate prediction of future credit risk.


---

## Dataset Overview

### **1. case_study1.csv (df1)**  
This dataset contains **customer-level credit bureau summary information** with **26 columns**.

**Key fields include:**

- `Total_TL`, `Tot_Active_TL`, `Tot_Closed_TL`  
- `Total_TL_opened_L6M`, `Total_TL_opened_L12M`  
- `pct_active_tl`, `pct_tl_open_L6M`, `pct_tl_open_L12M`  

**Product-level trade lines:**  
- `Auto_TL`, `CC_TL`, `PL_TL`, `Home_TL`, etc.

**Trade line ageing:**  
- `Age_Oldest_TL`, `Age_Newest_TL`

**Purpose:**  
These features describe the customer’s **historical credit behaviour**, total portfolio exposure, and trade-line distribution.

---

### **2. case_study2.csv (df2)**  
This dataset contains **detailed behavioural indicators, delinquency metrics, enquiry counts, utilization measures, and demographic information**.

**Key fields include:**

#### **Delinquency & Behaviour**
- `time_since_recent_payment`, `time_since_recent_deliquency`  
- `num_times_delinquent`, `max_delinquency_level`  
- `num_deliq_6mts`, `num_std`, `num_sub`, `num_dbt`, `num_lss`  

#### **Credit Enquiries**
- `enq_L3m`, `enq_L6m`, `enq_L12m`  
- `CC_enq`, `PL_enq`  

#### **Utilization Metrics**
- `CC_utilization`, `PL_utilization`  

#### **Demographic & Income Fields**
- `MARITALSTATUS`, `GENDER`, `EDUCATION`  
- `NETMONTHLYINCOME`, `Time_With_Curr_Empr`  

#### **Target Variable**
- `Approved_Flag`

**Purpose:**  
These features capture **current credit behaviour**, enquiry patterns, repayment performance, demographic details, and income stability — all of which are essential for predicting credit risk.

## Data Preprocessing and Feature Engineering

A crucial part of the workflow involves rigorous data cleaning and preparation. This includes handling missing or invalid values, correcting data inconsistencies, and merging datasets from multiple sources. The project implements advanced statistical feature selection techniques such as:

- **Chi-Square tests** for categorical variables  
- **Sequential VIF** to control multicollinearity  
- **ANOVA** to identify numerical predictors that differentiate risk classes  

These steps ensure that only meaningful, stable, and predictive features are included in the modelling stage.

---

## Building Machine Learning Models for Credit Risk

The modelling phase incorporates multiple algorithms—**Random Forest**, **XGBoost**, and **Decision Trees**—to compare performance and identify the most effective model. The pipeline-based architecture ensures clean separation of preprocessing, encoding, scaling, and model training, making the workflow scalable and production-ready.

---

## Evaluation and Interpretation of Credit Risk Models

The project demonstrates comprehensive model evaluation using metrics such as **accuracy**, **precision**, **recall**, and **F1-scores** across risk classes.

