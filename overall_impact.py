import streamlit as st
import pandas as pd
import pyodbc
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.linear_model import LogisticRegression
import warnings
from sklearn.preprocessing import OneHotEncoder
import plotly.express as px
import altair as alt
import seaborn as sns
import matplotlib.pyplot as plt
# Set up page

warnings.filterwarnings('ignore')

col1,col2,col3 = st.columns(3)
with col3:
    st.image(r"https://www.kentandmedway.icb.nhs.uk/application/files/cache/thumbnails/b9a92ca7b10d21ae1cdf056d95e99659.png")

st.title('Overall Assessment Of Impact Variables')


# https://docs.google.com/presentation/d/1MdLRe4-089MsEWttK9OgFvgBX8EqJ6LFCRC72zyz1jI/edit#slide=id.g2c0882fecc0_0_132

# conn = pyodbc.connect('DRIVER={ODBC Driver 18 for SQL Server};SERVER=test;DATABASE=test;UID=user;PWD=password')

# cnxn_str = ('DRIVER={ODBC Driver 17 for SQL Server};SERVER=10.118.254.94,1435;DATABASE=MRT_KMCCG_SANDBOX;USER=Temp_KMCCG;PASSWORD=JFhy3AVU62o3ZKf4')
#             # "Trusted_Connection=Yes"

# cnxn = pyodbc.connect(cnxn_str)

cnxn = pyodbc.connect(driver='{ODBC Driver 17 for SQL Server}',
                               server='10.118.254.94\LIVE,1435',
                               database='MRT_KMCCG_SANDBOX',
                               uid='',pwd='')         
 
#--Features -- variables that affect outcome
#--Weights -- what we are trying to predict
@st.cache_data
def get_data() -> pd.DataFrame:
    return pd.read_sql(
        
        "SELECT TOP (2000000) ATTENDANCE_STATUS, DER_AGE_AT_CDS_ACTIVITY_DATE, IMD_Decile, E.Main_Description, SEX, [Name], MSC.[Main_Description_60_Chars]\
        FROM [MRT_KMCCG].[dbo].[FSUS_OPA_MAIN] OP\
        JOIN [LOOKUPS].[Demography].[Index_Of_Multiple_Deprivation_By_LSOA1] LS  ON LS.[LSOA_Code]=OP.[DER_POSTCODE_LSOA_CODE]\
        JOIN [LOOKUPS].[Data_Dictionary].[Ethnic_Category_Code_SCD] E ON OP.ETHNIC_CATEGORY=E.Main_Code_Text\
        JOIN [LOOKUPS].[ODS].[All_Codes] PROV ON PROV.Code = OP.PROVIDER_CODE\
        JOIN [LOOKUPS].[Data_Dictionary].[Main_Specialty_Code_SCD]	MSC ON OP.[MAIN_SPECIALTY_CODE] = MSC.Main_Code_Text\
        WHERE IMD_Decile IS NOT NULL\
        AND [DER_AGE_AT_CDS_ACTIVITY_DATE] IS NOT NULL\
        AND E.Main_Description IS NOT NULL\
        AND SEX IS NOT NULL\
        AND [Name] IS NOT NULL\
        AND MSC.[Main_Description_60_Chars] IS NOT NULL\
        AND ATTENDANCE_STATUS IS NOT NULL\
        AND REP_DATE >= '20190101'\
        AND ATTENDANCE_STATUS IN (3,7,5,6)",
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

df["AGE"] = pd.to_numeric(df["DER_AGE_AT_CDS_ACTIVITY_DATE"])
df['BIN_AGE'] = pd.cut(df.AGE, [0,10,20,30,40,50,60,70,80,90,100,110,120], include_lowest=False)

df = df[['ATTENDED', 'BIN_AGE', 'IMD_Decile', 'Main_Description', 'SEX', 'Name', 'Main_Description_60_Chars']]

df = df.dropna()

# One Hot Encoding
categorical_cols = ['BIN_AGE', 'IMD_Decile', 'Main_Description', 'SEX', 'Name', 'Main_Description_60_Chars'] 

#import pandas as pd
df = pd.get_dummies(df, columns = categorical_cols)


# Split Data
X = df.drop('ATTENDED',axis=1)
y = df['ATTENDED']

# Prepare data set for logistic regression - decide weights and clean data set (potentially in query)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

# Plot in streamlit or in jupyter notebook to allow sharing

# # standardisation
# scaler = StandardScaler()
# x_stand = scaler.fit_transform(x)

# fit model 
model = LogisticRegression()
model.fit(X_train, y_train)

# predict
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

print (y_pred_train == y_train)
accuracy_train = np.mean(y_pred_train == y_train)
accuracy_test = np.mean(y_pred_test == y_test)

print (f'Accuracy of predicting training data = {accuracy_train}')
print (f'Accuracy of predicting test data = {accuracy_test}')

co_eff = model.coef_[0]
co_eff_df = pd.DataFrame() # create empty DataFrame
co_eff_df['feature'] = list(X) # Get feature names from X
co_eff_df['co_eff'] = co_eff
co_eff_df['abs_co_eff'] = np.abs(co_eff)
co_eff_df.sort_values(by='abs_co_eff', ascending=False, inplace=True)
co_eff_df['feature'] = co_eff_df['feature'].str.replace("POSTCODE_CLEAN_FINAL_","Postcode ")
co_eff_df['feature'] = co_eff_df['feature'].str.replace("GENDER_DESC_1","Gender ")
co_eff_df['feature'] = co_eff_df['feature'].str.replace("GENDER_DESC_2","Gender ")
co_eff_df['feature'] = co_eff_df['feature'].str.replace("ETHNIC_CATEGORY_DESC_","Ethnicity ")
co_eff_df['feature'] = co_eff_df['feature'].str.replace("DERIVED_AGE_","Derived Age ")
co_eff_df['feature'] = co_eff_df['feature'].str.replace("IMD_Decile_","IMD Decile ")
co_eff_df['feature'] = co_eff_df['feature'].str.replace("MAIN_SPECIALTY_CODE_DESC_","Specialty ")

# Clean for graphs
co_eff_df['feature'] = co_eff_df['feature'].str.replace("Main_Description_","")
co_eff_df['feature'] = co_eff_df['feature'].str.replace("BIN_AGE_","")
co_eff_df['feature'] = co_eff_df['feature'].str.replace("60_Chars_","")
co_eff_df['feature'] = co_eff_df['feature'].str.replace('SEX_9','Indeterminate')
co_eff_df['feature'] = co_eff_df['feature'].str.replace('Name_','Indeterminate')
co_eff_df['feature'] = co_eff_df['feature'].str.replace('SEX_1','Male')
co_eff_df['feature'] = co_eff_df['feature'].str.replace('SEX_2','Female')
co_eff_df['feature'] = co_eff_df['feature'].str.replace('SEX_0','Not Known')

col1, col2 = st.columns(2)

# Create positive co-eff dataframe and plot
co_eff_df_pos = co_eff_df[co_eff_df['co_eff']>0]
fig = px.bar(co_eff_df_pos.head(10), 
                y='feature', 
                x='abs_co_eff',
                opacity = 0.65)
fig.update_layout(yaxis_title=None)
with col1:
    st.write('Positive Impact On Attendance')
    st.plotly_chart(fig, use_container_width=True,color ="#284D76")

# Create negative co-eff dataframe and plot
co_eff_df_neg = co_eff_df[co_eff_df['co_eff']<0]
fig = px.bar(co_eff_df_neg.head(10),
            y='feature', 
            x='abs_co_eff',
            opacity = 0.65)
fig.update_layout(yaxis_title=None)
with col2:
    st.write('Negative Impact On Attendance')
    st.plotly_chart(fig, use_container_width=True,color ="#284D76")
    
