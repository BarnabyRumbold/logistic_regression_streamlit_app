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
    st.image(r"C:\Users\Barnaby.Rumbold\Desktop\NHS2.jpeg")
st.title('Provider')
col1,col2 = st.columns(2)


cnxn = pyodbc.connect(driver='{ODBC Driver 17 for SQL Server}',
                               server='10.118.254.94\LIVE,1435',
                               database='MRT_KMCCG_SANDBOX',
                               uid='PROD.KMCCG.Barnaby_Rumbold',pwd='SJ6N3obot5Qh4fWtcxPf')



# Get access to data
@st.cache_data
def get_data() -> pd.DataFrame:
    return pd.read_sql(
        
        "SELECT TOP (1000000) ATTENDANCE_STATUS, [Name]\
        FROM [MRT_KMCCG].[dbo].[FSUS_OPA_MAIN] AS OP\
		LEFT JOIN [LOOKUPS].[ODS].[All_Codes] PROV\
		ON PROV.Code = OP.PROVIDER_CODE\
        WHERE REP_DATE >= '20190101' AND [Name] IS NOT NULL AND ATTENDANCE_STATUS IS NOT NULL AND ATTENDANCE_STATUS IN (3,7,5,6)",
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
df = df[['Name','ATTENDED']]

# One Hot Encoding - allow prediction for categorical variables
categorical_cols = ['Name'] 
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
co_eff_df['feature'] = co_eff_df['feature'].str.replace("Name_","")
co_eff_df = co_eff_df.round(3)





# Create positive co-eff dataframe and plot

co_eff_df_pos = co_eff_df[co_eff_df['co_eff']>0]
fig = px.bar(co_eff_df_pos.head(10), 
                y='feature', 
                x='abs_co_eff',
                opacity = 0.65,
                text_auto=True)
fig.update_layout(yaxis_title=None)
fig.update_layout(barmode='stack', xaxis={'categoryorder':'total descending'})

with col1:
    st.write('Positive Impact On Attendance')
    st.plotly_chart(fig, use_container_width=True,color ="#284D76")

# Create negative co-eff dataframe and plot


co_eff_df_neg = co_eff_df[co_eff_df['co_eff']<0]
fig = px.bar(co_eff_df_neg.head(10),
            y='feature', 
            x='abs_co_eff',
            opacity = 0.65,
            text_auto=True)
fig.update_layout(yaxis_title=None)
fig.update_layout(barmode='stack', xaxis={'categoryorder':'total descending'})
with col2:
    st.write('Negative Impact On Attendance')
    st.plotly_chart(fig, use_container_width=True,color ="#284D76")

# # Add selected provider

# with col3:
#     st.write("Selected Provider Score")
#     specialties = co_eff_df['feature'].drop_duplicates().sort_values()
#     specialty_choice = st.sidebar.selectbox(label = 'Select Provider:', options=specialties, placeholder='Select Specialty')
#     co_eff_df = co_eff_df[co_eff_df['feature'] == specialty_choice]
#     st.metric(value = co_eff_df['co_eff'], label = specialty_choice)

