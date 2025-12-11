Project Description

This project demonstrates a complete, end-to-end Credit Risk Modelling pipeline built using statistical methods and machine learning. It uses two structured datasets, case_study1.csv and case_study2.csv, containing over 51,000 customer records and more than 80 credit bureau and behavioural variables. Together, these datasets offer a comprehensive view of customer credit profiles, allowing the model to classify borrowers into risk categories (P1–P4) and support data-driven lending decisions.

Dataset Overview
1. case_study1.csv (df1)

This dataset contains customer-level credit bureau summary information with 26 columns.

Key fields include:

Total_TL, Tot_Active_TL, Tot_Closed_TL

Total_TL_opened_L6M, Total_TL_opened_L12M

pct_active_tl, pct_tl_open_L6M, pct_tl_open_L12M

Product-level trade lines:

Auto_TL, CC_TL, PL_TL, Home_TL, etc.

Trade line ageing:

Age_Oldest_TL, Age_Newest_TL

Purpose:
These features describe the customer’s historical credit behaviour, portfolio exposure, and trade-line distribution.

2. case_study2.csv (df2)

This dataset contains detailed behavioural indicators, delinquency metrics, enquiries, utilization measures, and demographics.

Key fields include:

Delinquency & Behaviour:

time_since_recent_payment, time_since_recent_deliquency

num_times_delinquent, max_delinquency_level

num_deliq_6mts, num_std, num_sub, num_dbt, num_lss

Credit Enquiries:

enq_L3m, enq_L6m, enq_L12m

CC_enq, PL_enq

Utilization Metrics:

CC_utilization, PL_utilization

Demographic & Income Fields:

MARITALSTATUS, GENDER, EDUCATION

NETMONTHLYINCOME, Time_With_Curr_Empr

Target Variable:

Approved_Flag

Purpose:
These features capture current credit behaviour, enquiry patterns, repayment performance, demographic attributes, and income stability, all of which are critical for predicting credit risk.
