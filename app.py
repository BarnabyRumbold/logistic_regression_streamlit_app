# Import packages
import streamlit as st

# Set up pages, create titles and link to icons
pg = st.navigation([st.Page("home_page.py", title="Welcome", icon=":material/home:"),
        st.Page("overall_impact.py", title="Overall Impact", icon=":material/bar_chart:"),
        st.Page("log_reg_dep.py", title="Deprivation", icon=":material/person:"),
        st.Page("log_reg_ethnicity.py", title="Ethnicity", icon=":material/person:"),
        st.Page("log_reg_gender.py", title="Gender", icon=":material/person:"),
        st.Page("log_reg_age.py", title="Age", icon=":material/person:"),
        st.Page("log_reg_provider.py", title="Provider", icon=":material/local_hospital:"),
        st.Page("log_reg_specialty.py", title="Specialty", icon=":material/stethoscope:"),
        st.Page("log_reg_postcode.py", title="GP Postcode", icon=":material/map:")
        ])

pg.run()
