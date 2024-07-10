import streamlit as st
import numpy as np
descriptive_page = st.Page("pages/DAnalysis.py", title="Descriptive Analysis", icon=":material/add_circle:")
pre_process_page = st.Page("pages/Preprocessing.py", title="Preprocessing", icon=":material/delete:")
predict_page = st.Page("pages/PredictAnalysis.py", title="Predicitve Analysis", icon=":material/delete:")

pg = st.navigation([descriptive_page,pre_process_page, predict_page])

st.set_page_config(page_title="Data manager", page_icon=":material/edit:")
with st.container():
   st.write("This is inside the container")

   # You can call any Streamlit command, including custom components:
   st.bar_chart(np.random.randn(50, 3))

with st.container():
   st.write("This is inside the container")

   # You can call any Streamlit command, including custom components:
   st.bar_chart(np.random.randn(50, 3))


with st.container():
   st.write("This is inside the container")

   # You can call any Streamlit command, including custom components:
   st.bar_chart(np.random.randn(50, 3))

st.write("This is outside the container")

pg.run()


# import os
#
# import numpy as np
# import pandas as pd
# import streamlit as st
# from loguru import logger
#
# from pages.d_analysis import About
# from pages.pre_process import DataTable
# from utils.sidebar import sidebar_caption
#
# # Config the whole app
# st.set_page_config(
#     page_title="A Dashboard Template",
#     page_icon="🧊",
#     layout="wide",
#     initial_sidebar_state="expanded",
# )
#
#
# @st.cache()
# def fake_data():
#     """some fake data"""
#
#     dt = pd.date_range("2021-01-01", "2021-03-01")
#     df = pd.DataFrame(
#         {"datetime": dt, "values": np.random.randint(0, 10, size=len(dt))}
#     )
#
#     return df
#
#
# def main():
#     """A streamlit app template"""
#
#     st.sidebar.title("Tools")
#
#     PAGES = {
#         "Table": DataTable,
#         "About": About
#     }
#
#     # Select pages
#     # Use dropdown if you prefer
#     selection = st.sidebar.radio("Pages", list(PAGES.keys()))
#     sidebar_caption()
#
#     page = PAGES[selection]
#
#     DATA = {"base": fake_data()}
#
#     with st.spinner(f"Loading Page {selection} ..."):
#         page = page(DATA)
#         page()
#
#
# if __name__ == "__main__":
#     main()