# Import packages

import streamlit as st
import pandas as pd
import pyodbc
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.linear_model import LogisticRegression
import warnings
from sklearn.preprocessing import OneHotEncoder

# Set up page
warnings.filterwarnings('ignore')
st.set_page_config(layout="wide")
col1,col2,col3 = st.columns(3)
with col3:
    st.image(r"C:\Users\Barnaby.Rumbold\Desktop\NHS2.jpeg")
st.title('GP Postcode')
col1, col2 = st.columns(2)

# Connect to DW

cnxn = pyodbc.connect(driver='{ODBC Driver 17 for SQL Server}',
                               server='10.118.254.94\LIVE,1435',
                               database='MRT_KMCCG_SANDBOX',
                               uid='PROD.KMCCG.Barnaby_Rumbold',pwd='SJ6N3obot5Qh4fWtcxPf')         
 
#--Features -- variables that affect outcome
#--Weights -- what we are trying to predict

# Get access to data
@st.cache_data
def get_data() -> pd.DataFrame:
    return pd.read_sql(
        
        "SELECT TOP (500000) ATTENDANCE_STATUS, [REP_GP_PRACTICE_CODE]\
        FROM [MRT_KMCCG].[dbo].[FSUS_OPA_MAIN]\
        WHERE REP_DATE >= '20190101' AND [REP_GP_PRACTICE_CODE] IS NOT NULL AND ATTENDANCE_STATUS IS NOT NULL AND ATTENDANCE_STATUS IN (3,7,5,6)",
        cnxn)

# Call function to get dataframe
df = get_data()
df['GP_CODE'] = df['REP_GP_PRACTICE_CODE'] 


# Get connection for gp postcode data
def get_gp() -> pd.DataFrame:
    return pd.read_sql(
        "SELECT DISTINCT Organisation_Code, Postcode, Latitude_1m, Longitude_1m\
        FROM [LOOKUPS].[ODS].[GP_Practices_And_Prescribing_CCs_SCD] GP\
        JOIN [LOOKUPS].[ODS].[Postcode_Grid_Refs_Eng_Wal_Sco_And_NI_SCD] LL\
        ON LL.Postcode_8_chars=GP.Postcode\
        WHERE Latitude_1m IS NOT NULL",
        cnxn)
gp = get_gp()
gp['GP_CODE'] = gp['Organisation_Code']
print(gp)


df = df.merge(gp, on='GP_CODE')
print(df)


# Normalise attendance column
norm_list=[]
for i in df['ATTENDANCE_STATUS']:
    if i in ('5','6'):
        norm_list.append(1)
    else:
        norm_list.append(0)
df['ATTENDED'] = norm_list
df = df[['Postcode','ATTENDED']]
df = df.dropna()

# One Hot Encoding - allow prediction for categorical variables
categorical_cols = ['Postcode'] 
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
co_eff_df['feature'] = co_eff_df['feature'].str.replace("Postcode_","")
co_eff_df = co_eff_df.round(3)
# Write data to streamlit

with col1:
    co_eff_df.rename(columns={'feature': 'Postcode', 'co_eff': 'Coefficient Score'}, inplace=True)
    st.write(co_eff_df[['Postcode','Coefficient Score']])



# Join back to GP dataframe to get lat long
gp = gp[['Latitude_1m','Longitude_1m', 'Postcode']]
co_eff_df['Postcode'] = co_eff_df['Postcode'].str.replace("Postcode_", "")
co_eff_df = co_eff_df.merge(gp, on='Postcode')



 # Mapping data formatting
co_eff_df['lat'] = co_eff_df['Latitude_1m']
co_eff_df['lon'] = co_eff_df['Longitude_1m']
co_eff_df_map = co_eff_df[['Coefficient Score','lat','lon']]

# Add colour coding
colour = []
for i in co_eff_df_map['Coefficient Score']:
    if i > 0.5:
        colour.append("#009639")
    if i < -0.5: 
        colour.append("#DA291C")
    else:
        colour.append("#768692")

colour = pd.Series(colour)      
co_eff_df_map['colour'] = colour
co_eff_df_map['co_eff_correct'] = co_eff_df_map['Coefficient Score']*1500
co_eff_df_map = co_eff_df_map.dropna()




# Plot map 
with col2:
    st.write("Map Visual")
    st.write("The below map uses colour coding and size to highlight attendance coefficient score based on the registered GP practice postcode for the referral.")
    st.write("A coeffcient score of greater than 0.5 is given a green colour code")
    st.write("A coeffcient score of less than -0.5 is given a red colour code")
    st.write("A coeffcient score between 0.5 and -0.5 is given a grey colour code")
    st.write("The greater the size, the higher the score")
st.map(co_eff_df_map,
        size="co_eff_correct",
        color="colour",
        use_container_width=False)
    
# with col3:
#     st.write("Selected GP Score")
#     postcode = co_eff_df['Postcode'].drop_duplicates().sort_values()
#     specialty_choice = st.sidebar.selectbox(label = 'Select GP Postcode:', options=postcode, placeholder='Select Specialty')
#     co_eff_df = co_eff_df[co_eff_df['Postcode'] == specialty_choice]
#     st.metric(value = co_eff_df['Coefficient Score'], label = specialty_choice)

