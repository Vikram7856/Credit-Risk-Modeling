# **Credit Risk Modelling – End-to-End Machine Learning Project**

This project demonstrates a complete, end-to-end **Credit Risk Modelling** pipeline built using statistical techniques and machine learning. It processes and models two real-world credit datasets containing **51,000+ customers** and more than **80 credit bureau, enquiry, and behavioural attributes**.  
The goal is to classify each customer into **risk segments (P1–P4)** and enable lending teams to make **data-driven credit decisions**.

---

## **1. Project Objectives**

- Build a robust credit risk scoring model using real-world structured credit data.  
- Engineer, clean, filter, and select features using statistical methods.  
- Compare performance of multiple machine learning algorithms.  
- Generate interpretable results for credit decisioning teams.  
- Provide an analysis-ready, production-friendly workflow.  

---

## **2. Dataset Summary**

### **Dataset 1: case_study1.csv (df1)**  
Contains **bureau summary and trade-line information** with **26 columns**.

**Key attributes:**
- `Total_TL`, `Tot_Active_TL`, `Tot_Closed_TL`  
- `Total_TL_opened_L6M`, `Total_TL_opened_L12M`  
- `pct_active_tl`, `pct_tl_open_L6M`, `pct_tl_open_L12M`  
- Product-level exposures: `Auto_TL`, `CC_TL`, `PL_TL`, `Home_TL`  
- `Age_Oldest_TL`, `Age_Newest_TL`  

**Purpose:** Represents historical credit behavior, portfolio mix, and repayment footprint.

---

### **Dataset 2: case_study2.csv (df2)**  
Contains **behavioural variables, delinquencies, enquiries, income, and demographics** with **62 columns**.

**Key fields:**

**Delinquency**
- `num_times_delinquent`, `max_recent_level_of_deliq`, `num_deliq_6/12 months`

**Enquiries**
- `enq_L3m`, `enq_L6m`, `enq_L12m`, `CC_enq`, `PL_enq`

**Utilization**
- `CC_utilization`, `PL_utilization`

**Demographics**
- `MARITALSTATUS`, `EDUCATION`, `GENDER`

**Income**
- `NETMONTHLYINCOME`, `Time_With_Curr_Empr`

**Target Variable**
- `Approved_Flag` (P1–P4)

**Purpose:** Captures recent credit behavior, borrower intent, financial stability, and risk indicators.

---

## **3. Data Cleaning & Preprocessing**

### **Handling invalid flags (-99999)**
- Columns with extreme missing values (>10,000 occurrences) were dropped.  
- Remaining rows with `-99999` were filtered out for each column.

### **Merging datasets**
Merged `df1` and `df2` using the key: PROSPECTID


Final merged dataset size: **42,064 rows × 79 columns**

### **Categorical Identification**
Detected categorical features:
- `MARITALSTATUS`  
- `EDUCATION`  
- `GENDER`  
- `last_prod_enq2`  
- `first_prod_enq2`  
- `Approved_Flag` (target)

---

## **4. Feature Selection**

Multiple statistical feature selection techniques were applied:

### **1. Chi-Square Test (Categorical vs Target)**
- All categorical variables had **p ≤ 0.05**, indicating strong influence on outcomes.

### **2. Variance Inflation Factor (VIF)**
Performed in two modes:
- Standard VIF  
- Sequential VIF (iteratively removing multicollinear variables)

**Result:** Numerical features reduced **from 72 → 45**

### **3. One-Way ANOVA (Numerical vs Approved_Flag)**
- Identified features that differentiate P1–P4 groups.
- Final selected numerical features: **37**

### **Final feature set used for modelling:**
- **37 numerical features**  
- **5 categorical features**  
**Total: 42 predictors**

---

## **5. Feature Engineering**

### **1. Ordinal Encoding for EDUCATION**
Mapped according to academic hierarchy:

SSC → 1
12TH → 2
GRADUATE / UNDER GRADUATE / PROFESSIONAL → 3
POST-GRADUATE → 4
OTHERS → 1


### **2. One-Hot Encoding**
Applied to:
- `MARITALSTATUS`  
- `GENDER`  
- `last_prod_enq2`  
- `first_prod_enq2`  

### **3. Scaling (StandardScaler)**
Applied selectively to:
- `Age_Oldest_TL`  
- `Age_Newest_TL`  
- `time_since_recent_payment`  
- `max_recent_level_of_deliq`  
- `NETMONTHLYINCOME`  
- `Time_With_Curr_Empr`  


