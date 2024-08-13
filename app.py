#
# import numpy as np
# import pandas as pd
# import streamlit as st
# import wfdb
#
# # from loguru import logger
#
# from page.PreProccessing import Preprocess
# from page.Descriptive_Analysis import DescriptiveAnalysis
# from page.Predictive_Analysis import PredictiveAnalysis
# from utils.sidebar import sidebar_caption
#
# #
# # # Config the whole app
# # st.set_page_config(
# #     page_title="A Dashboard Template",
# #     page_icon="🧊",
# #     layout="wide",
# #     initial_sidebar_state="expanded",
# # )
# #
#
# # @st.cache()
# def fake_data():
#     """some fakest.chch data"""
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
#     # st.sidebar.title("Tools")
#     st.set_page_config(layout="wide")
#     PAGES = {
#         "Descriptive Analysis": DescriptiveAnalysis,
#         "PreProcessing": Preprocess,
#         "Predictive Analysis": PredictiveAnalysis
#     }
#
#     # Select page
#     # Use dropdown if you prefer
#     selection = st.sidebar.radio("Tools", list(PAGES.keys()))
#     sidebar_caption()
#     option = st.sidebar.selectbox(
#         "Select A patient Record",
#         ("100", "101", "102", "103", "104", "105", "106"))
#
#     data = f"mit-bih-arrhythmia-database-1.0.0/{option}"
#
#     record = wfdb.rdrecord(data, smooth_frames=True)
#
#     page = PAGES[selection]
#
#     DATA = {"record": record, }
#
#     with st.spinner(f"Loading Page {selection} ..."):
#
#
#
#         page = page(DATA)
#         page()
#
#
# if __name__ == "__main__":
#     main()

import streamlit as st
from page.PreProccessing import Preprocess
import wfdb

from page.Descriptive_Analysis import DescriptiveAnalysis
from page.Predictive_Analysis import PredictiveAnalysis
from page.arrythmia_detection import ArrhythmiaAnalysis
st.set_page_config(layout="wide")
# tab1, tab2 = st.tabs(["📈 Chart", "🗃 Data"])

tab1, tab2, tab3,tab4 = st.tabs(["📈 Preprocessing", "🗃 Descriptive Analysis", "📊 Predictive Analysis","Arrhythia Detection"])

data = f"mit-bih-arrhythmia-database-1.0.0/100"

record = wfdb.rdrecord(data, smooth_frames=True)
signal=[]
# page = PAGES[selection]

DATA = {"record": record, }
with tab1:
    option = st.selectbox(
        "Select A patient Record",
        ("100", "101", "102", "103", "104", "105", "106"))

    data = f"mit-bih-arrhythmia-database-1.0.0/{option}"

    record = wfdb.rdrecord(data, smooth_frames=True)
    DATA = {"record": record }

    page=Preprocess(DATA)
    signal=page.content()
    print(signal)
    # st.header("A cat")
    # st.image("https://static.streamlit.io/examples/cat.jpg", width=200)

with tab2:
    DATA = {"record": record, "signal":signal}

    page = DescriptiveAnalysis(DATA)
    page.content()
    st.header("A dog")
    st.image("https://static.streamlit.io/examples/dog.jpg", width=200)

with tab3:
   st.header("An owl")
   st.image("https://static.streamlit.io/examples/owl.jpg", width=200)

with tab4:
    DATA = {"record": record, "signal":signal}

    page = ArrhythmiaAnalysis(DATA)
    page.content()