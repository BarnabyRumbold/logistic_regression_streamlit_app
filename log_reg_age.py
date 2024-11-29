# Import packages

import streamlit as st
import pandas as pd
import pyodbc
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.linear_model import LogisticRegression
import warnings
from sklearn.preprocessing import OneHotEncoder
warnings.filterwarnings('ignore')
import plotly.express as px

# Connect to DW
st.set_page_config(layout="wide")
col1,col2,col3 = st.columns(3)
with col3:
    st.image(r"https://github.com/BarnabyRumbold/HSMA/blob/master/NHS.jpg")
st.title('Age')
col1,col2 = st.columns(2)

cnxn = pyodbc.connect(driver='{ODBC Driver 17 for SQL Server}',
                               server='10.118.254.94\LIVE,1435',
                               database='MRT_KMCCG',
                               uid='PROD.KMCCG.Barnaby_Rumbold',pwd='SJ6N3obot5Qh4fWtcxPf')



# Get access to data
@st.cache_data
def get_data() -> pd.DataFrame:
    return pd.read_sql(
        
        "SELECT TOP (50000)  [DER_AGE_AT_CDS_ACTIVITY_DATE], ATTENDANCE_STATUS\
        FROM [MRT_KMCCG].[dbo].[FSUS_OPA_MAIN]\
        WHERE [DER_AGE_AT_CDS_ACTIVITY_DATE] IS NOT NULL AND ATTENDANCE_STATUS IS NOT NULL AND [DER_AGE_AT_CDS_ACTIVITY_DATE] > 0 AND [DER_AGE_AT_CDS_ACTIVITY_DATE] < 116 AND REP_DATE >= '20190101' AND ATTENDANCE_STATUS IN (3,7,5,6)", 
        cnxn)

# Call function to get dataframe
df = get_data() 

# Normalise attendance column
norm_list=[]
for i in df['ATTENDANCE_STATUS']:
    if i in ('5','6'):
        norm_list.append(1)
    else:
        norm_list.append(0)
df['ATTENDED'] = norm_list
df = df[['DER_AGE_AT_CDS_ACTIVITY_DATE','ATTENDED']]

# Create bins for age
df["AGE"] = pd.to_numeric(df["DER_AGE_AT_CDS_ACTIVITY_DATE"])
df['BIN_AGE'] = pd.cut(df.AGE, [0,10,20,30,40,50,60,70,80,90,100,110,120], include_lowest=False)

df = df[['BIN_AGE','ATTENDED']]

# One Hot Encoding - allow prediction for categorical variables
categorical_cols = ['BIN_AGE'] 
df = pd.get_dummies(df, columns = categorical_cols)

# Split data
X = df.drop('ATTENDED',axis=1)
y = df['ATTENDED']

# Prepare data set for logistic regression - decide weights and clean data set (potentially in query)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

# Fit model 
model = LogisticRegression()
model.fit(X_train, y_train)

# Predict
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

# Predictions and accuracy test
accuracy_train = np.mean(y_pred_train == y_train)
accuracy_test = np.mean(y_pred_test == y_test)
co_eff = model.coef_[0]
co_eff_df = pd.DataFrame() # create empty DataFrame
co_eff_df['feature'] = list(X) # Get feature names from X
co_eff_df['co_eff'] = co_eff
co_eff_df['abs_co_eff'] = np.abs(co_eff)
co_eff_df.sort_values(by='abs_co_eff', ascending=False, inplace=True)
co_eff_df['feature'] = co_eff_df['feature'].str.replace("BIN_AGE_","")
co_eff_df = co_eff_df.round(3)
co_eff_df = co_eff_df.head(20)

# def Z(pay_amt1, age):
#     return logreg.intercept_[0]+logreg.coef_[0][0]*pay_amt1+logreg.coef_[0][1]*age
# def prob_default(limit_balance, age):
#     z = Z(limit_balance, age)
#     return 1/(1+np.exp(-z))

# prob_default(limit_balance=1.6,age=40)

# Create positive co-eff dataframe and plot
co_eff_df_pos = co_eff_df[co_eff_df['co_eff']>0]
fig = px.bar(co_eff_df_pos, 
                y='feature', 
                x='abs_co_eff',
                opacity = 0.65,
                text_auto=True)
fig.update_layout(yaxis_title=None)


with col1:
    st.write('Positive Impact On Attendance')
    st.plotly_chart(fig, use_container_width=True,color ="#284D76")

# Create negative co-eff dataframe and plot
co_eff_df_neg = co_eff_df[co_eff_df['co_eff']<0]
fig = px.bar(co_eff_df_neg,
            y='feature', 
            x='abs_co_eff',
            opacity = 0.65,
            text_auto=True)
fig.update_layout(yaxis_title=None)
with col2:
    st.write('Negative Impact On Attendance')
    st.plotly_chart(fig, use_container_width=True,color ="#284D76")


