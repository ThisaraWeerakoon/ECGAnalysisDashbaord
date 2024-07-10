
import numpy as np
import pandas as pd
import streamlit as st
import wfdb

# from loguru import logger

from page.PreProccessing import Preprocess
from page.Descriptive_Analysis import DescriptiveAnalysis
from page.Predictive_Analysis import PredictiveAnalysis
from utils.sidebar import sidebar_caption

#
# # Config the whole app
# st.set_page_config(
#     page_title="A Dashboard Template",
#     page_icon="🧊",
#     layout="wide",
#     initial_sidebar_state="expanded",
# )
#

# @st.cache()
def fake_data():
    """some fakest.chch data"""

    dt = pd.date_range("2021-01-01", "2021-03-01")
    df = pd.DataFrame(
        {"datetime": dt, "values": np.random.randint(0, 10, size=len(dt))}
    )

    return df


def main():
    """A streamlit app template"""

    # st.sidebar.title("Tools")

    PAGES = {
        "Descriptive Analysis": DescriptiveAnalysis,
        "PreProcessing": Preprocess,
        "Predictive Analysis": PredictiveAnalysis
    }

    # Select page
    # Use dropdown if you prefer
    selection = st.sidebar.radio("Tools", list(PAGES.keys()))
    sidebar_caption()
    option = st.sidebar.selectbox(
        "Select A patient Record",
        ("100", "101", "102"))

    data = f"mit-bih-arrhythmia-database-1.0.0/{option}"

    record = wfdb.rdrecord(data, smooth_frames=True)

    page = PAGES[selection]

    DATA = {"record": record}

    with st.spinner(f"Loading Page {selection} ..."):

        page = page(DATA)
        page()


if __name__ == "__main__":
    main()