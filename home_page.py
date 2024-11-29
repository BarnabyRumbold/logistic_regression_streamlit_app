# Import packages

import streamlit as st

# Set up page layout 
st.set_page_config(layout="wide",
                   initial_sidebar_state="expanded",
                   page_title="Attendance Predicator App")
col1,col2,col3 = st.columns(3)

with col3:
    st.image('https://github.com/BarnabyRumbold/HSMA/blob/master/NHS.jpg')

# Create title and layout info
st.title("Attendance Predicator App")
col1, col2, col3, col4 = st.columns(4)

# Add content
with col1:
    st.write("App Overview")
    st.write("This app looks at outpatient attendance data provided to the ICB and breaks this down using different demographic and geographical information. The app uses Logisic Regression to help understand the impact of these variables.")
    st.write("How Attendance Is Defined")
    st.write("The data is initially filtered to only include attendance codes of 3,7,5 and 6. These correspond to recorded instances of patient attendance and instances where patient's were recorded as either not attending or attended too late to be seen")

with col2:
    st.write("Logistic Regression Overview")
    st.write("Logistic regression is a statistical method that uses a mathematical model to estimate the probability of an event occurring based on a set of independent variables.")
    st.write("Common examples of logistic regression are prediciting if an email is spam or not, whether a person is likely to have a certain medical condition, or which product a customer is most likely to buy.")    

with col3:
    st.write("How To Interpret Results")
    st.write("Logistic regression creates a figure that helps us to understand the impact of variables an event happening in our data.")
    st.write("For example, if we are looking at incidences of a positive cancer diagnosis, and we had information on lifestyle, our data may be presented as:")
    st.write("Smoker - 2.432")
    st.write("Obese - 1.368")
    st.write("Alcohol Use - 0.983")
    st.write("This would mean that being a smoker means that you are 2.432 times more likely to have a cancer diagnosis, than those that are not a smoker (within our data set).")

with col4:
    st.write("Further Info")
    st.write("Potential uses for this app could be flagging individuals not currently served by existing DNA processes to improve outpatient attendance.")
    st.write("Contact: barnaby.rumbold@nhs.net")
    st.write("Click a page on the left to get started.")
