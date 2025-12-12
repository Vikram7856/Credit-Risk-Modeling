# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

# import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from scipy.stats import chi2_contingency
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,classification_report,precision_recall_fscore_support
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    precision_score,
    recall_score,
    f1_score
)

import warnings
import os
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
# load dataset

a1 = pd.read_csv("C:\Pandas project\Credit Risk Modelling\case_study1.xlsx - case_study1.csv")
a2 = pd.read_csv("C:\Pandas project\Credit Risk Modelling\case_study2.xlsx - case_study2.csv")

df1 = a1.copy()
df2 = a2.copy()

df1.head()
df1.shape
df1.columns

df2.head()
df2.shape
df2.columns

#remove nulls
df1.loc[df1['Age_Oldest_TL'] != -99999].shape

for col in df1.columns:
    total = len(df1)  # total number of rows
    count = df1.loc[df1[col] == -99999].shape[0] # count of -99999
    percent = (count / total) * 100  # calculate percentage
    print(f"{col}: {count} ({percent:.2f}%)")


df1 = df1.loc[df1['Age_Oldest_TL'] != -99999]




columns_to_be_removed = []

df2.dtypes

for i in df2.columns:
    print(i, df2.loc[df2[i] == -99999].shape[0])


for col in df2.columns:
    total = len(df2)  # total number of rows
    count = df2.loc[df2[col] == -99999].shape[0] # count of -99999
    percent = (count / total) * 100  # calculate percentage
    print(f"{col}: {count} ({percent:.2f}%)")


for i in df2.columns:
    if df2.loc[df2[i] == -99999].shape[0] > 10000:
        columns_to_be_removed.append(i)
        
print(columns_to_be_removed)
 

# dropping columns which have high null values     
df2 = df2.drop(columns_to_be_removed,axis = 1)

df2.shape

df2.isnull().sum()

# df2 with all non null values
for i in df2.columns:
    df2 = df2.loc[df2[i] != -99999]
    

# finding common columns to merge    
for i in list(df1.columns):
    if i in list(df2.columns):
        print(i)
        
df = pd.merge(df1, df2, how = 'inner', left_on = 'PROSPECTID' , right_on = 'PROSPECTID')

df.shape

df.info()
df.isnull().sum().sum()

# check how many columns are ctegoricals
for i in df.columns:
    if df[i].dtype == 'object':
        print(i)
        

df['MARITALSTATUS'].value_counts()
df['EDUCATION'].value_counts()
df['GENDER'].value_counts()
df['last_prod_enq2'].value_counts()
df['first_prod_enq2'].value_counts()
df['Approved_Flag'].value_counts()

# chi_square test
for i in ['MARITALSTATUS','EDUCATION','GENDER','last_prod_enq2','first_prod_enq2']:
    chi2,pval,_,_ = chi2_contingency(pd.crosstab(df[i], df["Approved_Flag"]))
    print(i,'---',pval)
    
# since all columns featres have p_value <= 0.05, we will accept all  








# Sequential vif
# Step 1: take only numeric columns
s_df = df.copy()
s_df.shape
# Step 1: Select only numeric columns from s_df
s_numeric_cols = []

for col in s_df.columns:
    if s_df[col].dtype != 'object' and col not in ['PROSPECTID', 'Approved_Flag']:
        s_numeric_cols.append(col)

# Step 2: Prepare VIF DataFrame
s_vif_df = s_df[s_numeric_cols].copy()
s_kept_cols = s_vif_df.columns.tolist()

# Step 3: Sequential VIF Loop
dropped = True

while dropped:
    dropped = False
    s_vif_list = []

    # Compute VIF for all remaining columns
    for idx in range(len(s_kept_cols)):
        vif_val = variance_inflation_factor(s_vif_df[s_kept_cols].values, idx)
        s_vif_list.append(vif_val)

    # Find highest VIF
    max_vif = max(s_vif_list)
    max_idx = s_vif_list.index(max_vif)
    s_col_to_drop = s_kept_cols[max_idx]

    print(s_col_to_drop, "----", max_vif)

    # Drop if above threshold
    if max_vif > 6:
        print("Dropping:", s_col_to_drop)
        s_vif_df = s_vif_df.drop(columns=[s_col_to_drop],axis=1)
        s_kept_cols = s_vif_df.columns.tolist()
        dropped = True

# Final set of columns after Sequential VIF
s_kept_cols
len(s_kept_cols)
 # After sequential vif out of 72 left columns to s_kept_cols is 45
 
 

 
 

# check Annova for columns to be kept

from scipy.stats import f_oneway

columns_to_be_kept_numerical = []

for i in s_kept_cols:
    a=list(df[i])
    b=list(df['Approved_Flag'])
    
    group_P1 = [value for value , group in zip(a,b) if group == 'P1']
    group_P2 = [value for value , group in zip(a,b) if group == 'P2']
    group_P3 = [value for value , group in zip(a,b) if group == 'P3']
    group_P4 = [value for value , group in zip(a,b) if group == 'P4']
    
    f_statistic,p_value = f_oneway(group_P1,group_P2,group_P3,group_P4)
    
    if p_value <= 0.05:
        columns_to_be_kept_numerical.append(i)
        
# columns to be kept after annova test is 37 out of 72
columns_to_be_kept_numerical
# feature selection is done for cat and num features




# listing all the final features
features = columns_to_be_kept_numerical + ['MARITALSTATUS', 'EDUCATION', 'GENDER', 'last_prod_enq2', 'first_prod_enq2']
df = df[features + ['Approved_Flag']]






# Label encoding for the categorical features
['MARITALSTATUS', 'EDUCATION', 'GENDER' , 'last_prod_enq2' ,'first_prod_enq2']



df['MARITALSTATUS'].unique()    
df['EDUCATION'].unique()
df['GENDER'].unique()
df['last_prod_enq2'].unique()
df['first_prod_enq2'].unique()



# Ordinal feature -- EDUCATION
# SSC            : 1
# 12TH           : 2
# GRADUATE       : 3
# UNDER GRADUATE : 3
# POST-GRADUATE  : 4
# OTHERS         : 1
# PROFESSIONAL   : 3


# Others has to be verified by the business end user 




df_encoded = pd.get_dummies(df, columns=['MARITALSTATUS','GENDER', 'last_prod_enq2' ,'first_prod_enq2'])



df_encoded.info()
ck = df_encoded.describe()













# Machine Learing model fitting

# Data processing

# 1. Random Forest

y = df_encoded['Approved_Flag']
x = df_encoded. drop ( ['Approved_Flag'], axis = 1 )




x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)





rf_classifier = RandomForestClassifier(n_estimators = 200, random_state=42)





rf_classifier.fit(x_train, y_train)



y_pred = rf_classifier.predict(x_test)





accuracy = accuracy_score(y_test, y_pred)
print ()
print(f'Accuracy: {accuracy}')
print ()
precision, recall, f1_score, _ = precision_recall_fscore_support(y_test, y_pred)


for i, v in enumerate(['p1', 'p2', 'p3', 'p4']):
    print(f"Class {v}:")
    print(f"Precision: {precision[i]}")
    print(f"Recall: {recall[i]}")
    print(f"F1 Score: {f1_score[i]}")
    print()
    


# =====================================
# 1. Clean EDUCATION Column
# =====================================

education_map = {
    'SSC': 1,
    '12TH': 2,
    'OTHERS': 1,
    'GRADUATE': 3,
    'UNDER GRADUATE': 3,
    'PROFESSIONAL': 3,
    'POST-GRADUATE': 4
}

df['EDUCATION'] = df['EDUCATION'].map(education_map)
df['EDUCATION'].value_counts()
df.info()

# =====================================
# 2. Prepare X and y
# =====================================

y = df['Approved_Flag']
X = df.drop(['Approved_Flag'], axis=1)

# Identify categorical & numerical columns  
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
numeric_cols = X.select_dtypes(exclude=['object']).columns.tolist()


print("Categorical Columns:", categorical_cols)
print("Numeric Columns:", numeric_cols)

# =====================================
# 3. Build Preprocessor
# =====================================

preprocess = ColumnTransformer(
    transformers=[
        ('categorical', OneHotEncoder(handle_unknown='ignore'), categorical_cols),
        ('numeric', 'passthrough', numeric_cols)
    ]
)

# =====================================
# 4. Build Pipeline (Preprocess + Model)
# =====================================

rf_pipeline = Pipeline(steps=[
    ('preprocessing', preprocess),
    ('model', RandomForestClassifier(n_estimators=200, random_state=42))
])

# =====================================
# 5. Train-Test Split
# =====================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

# =====================================
# 6. Fit Model
# =====================================

rf_pipeline.fit(X_train, y_train)

# =====================================
# 7. Predict
# =====================================

y_pred = rf_pipeline.predict(X_test)

# =====================================
# 8. Evaluation
# =====================================

accuracy = accuracy_score(y_test, y_pred)
print("\nModel Accuracy:", accuracy)

precision, recall, f1_score, support = precision_recall_fscore_support(y_test, y_pred)
unique_classes = y.unique()

for i, cls in enumerate(unique_classes):
    print(f"\nClass: {cls}")
    print("Precision:", precision[i])
    print("Recall:", recall[i])
    print("F1 Score:", f1_score[i])




# 2. xgboost

import xgboost as xgb

xgb_classifier = xgb.XGBClassifier(objective='multi:softmax',  num_class=4)



y = df_encoded['Approved_Flag']
x = df_encoded. drop ( ['Approved_Flag'], axis = 1 )


label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)


x_train, x_test, y_train, y_test = train_test_split(x, y_encoded, test_size=0.2, random_state=42)




xgb_classifier.fit(x_train, y_train)
y_pred = xgb_classifier.predict(x_test)

accuracy = accuracy_score(y_test, y_pred)
print ()
print(f'Accuracy: {accuracy:.2f}')
print ()

precision, recall, f1_score, _ = precision_recall_fscore_support(y_test, y_pred)

for i, v in enumerate(['p1', 'p2', 'p3', 'p4']):
    print(f"Class {v}:")
    print(f"Precision: {precision[i]}")
    print(f"Recall: {recall[i]}")
    print(f"F1 Score: {f1_score[i]}")
    print()





# 3. Decision Tree
from sklearn.tree import DecisionTreeClassifier


y = df_encoded['Approved_Flag']
x = df_encoded. drop ( ['Approved_Flag'], axis = 1 )

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)


dt_model = DecisionTreeClassifier(max_depth=20, min_samples_split=10)
dt_model.fit(x_train, y_train)
y_pred = dt_model.predict(x_test)

accuracy = accuracy_score(y_test, y_pred)
print ()
print(f"Accuracy: {accuracy:.2f}")
print ()

precision, recall, f1_score, _ = precision_recall_fscore_support(y_test, y_pred)

for i, v in enumerate(['p1', 'p2', 'p3', 'p4']):
    print(f"Class {v}:")
    print(f"Precision: {precision[i]}")
    print(f"Recall: {recall[i]}")
    print(f"F1 Score: {f1_score[i]}")
    print()



        
len(columns_to_be_kept_numerical)
# after performing annova one way we left with 37 columns out of 72    

# Random forest = 0.76
# Xgboost = 0.72
#  Decision tree = 0.71





# HP TUNNING
# FEATURE ENGINNERING -- SCALING,GRAPHS,FEATURE ENGG



# xgboost is giving me best results
# We will further finetune it


# Apply standard scaler 

from sklearn.preprocessing import StandardScaler

columns_to_be_scaled = ['Age_Oldest_TL','Age_Newest_TL','time_since_recent_payment',
'max_recent_level_of_deliq','recent_level_of_deliq',
'time_since_recent_enq','NETMONTHLYINCOME','Time_With_Curr_Empr']

for i in columns_to_be_scaled:
    column_data = df_encoded[i].values.reshape(-1, 1)
    scaler = StandardScaler()
    scaled_column = scaler.fit_transform(column_data)
    df_encoded[i] = scaled_column





    


# ------------------------------------------------------------
# 2. PREPARE FEATURES AND TARGET
# ------------------------------------------------------------
y = df_encoded['Approved_Flag']                        # Target
X = df_encoded.drop(['Approved_Flag'], axis=1)         # Features

# Encode target labels (P1, P2, P3, P4 -> 0,1,2,3)
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42
)


# ------------------------------------------------------------
# 3. DEFINE BASE MODEL
# ------------------------------------------------------------
model = xgb.XGBClassifier(
    objective='multi:softmax',
    num_class=4,
    eval_metric='mlogloss'
)


# ------------------------------------------------------------
# 4. HYPERPARAMETER GRID
# ------------------------------------------------------------
param_grid = {
    'colsample_bytree': [0.1, 0.3, 0.5, 0.7, 0.9],
    'learning_rate'   : [0.001, 0.01, 0.1, 1],
    'max_depth'       : [3, 5, 8, 10],
    'alpha'           : [1, 10, 100],
    'n_estimators'    : [10, 50, 100]
}


# ------------------------------------------------------------
# 5. APPLY GRIDSEARCHCV
# ------------------------------------------------------------
grid_search = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    cv=3,
    scoring='accuracy',
    verbose=2,
    n_jobs=-1
)

grid_search.fit(X_train, y_train)


# ------------------------------------------------------------
# 6. BEST PARAMETERS & ACCURACY
# ------------------------------------------------------------
print("Best Parameters:")
print(grid_search.best_params_)

print("\nBest CV Accuracy:")
print(grid_search.best_score_)


# ------------------------------------------------------------
# 7. TEST SET PREDICTION
# ------------------------------------------------------------
best_model = grid_search.best_estimator_

y_pred_test = best_model.predict(X_test)

test_accuracy = accuracy_score(y_test, y_pred_test)

print("\nTest Accuracy:", test_accuracy)


# ------------------------------------------------------------
# 8. PRECISION, RECALL, F1 SCORE
# ------------------------------------------------------------
precision = precision_score(y_test, y_pred_test, average='weighted')
recall    = recall_score(y_test, y_pred_test, average='weighted')
f1        = f1_score(y_test, y_pred_test, average='weighted')

print("\nPrecision (Weighted):", precision)
print("Recall (Weighted):", recall)
print("F1 Score (Weighted):", f1)



# Best CV Accuracy: 0.7787287153427833

# Test Accuracy: 0.7795079044336146

# Precision (Weighted): 0.7594568388291515
# Recall (Weighted): 0.7795079044336146
# F1 Score (Weighted): 0.7647677446658347







