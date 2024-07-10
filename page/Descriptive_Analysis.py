import os

import streamlit as st
import time
import numpy as np
import wfdb

from utils.page import Page

class DescriptiveAnalysis ( Page):
    def __init__(self, data, **kwargs):
        name = "About"
        super().__init__(name, data, **kwargs)

    def content(self):




        last_rows = self.data["signal"]
        chart = st.line_chart(last_rows)

        st.button("Re-run")

        with st.container():
            st.write("This is inside the container")

            # You can call any Streamlit command, including custom components:
            st.bar_chart(np.random.randn(50, 3))

        st.write("This is outside the container")

        with st.container():
            st.write("This is inside the container")

            # You can call any Streamlit command, including custom components:
            st.bar_chart(np.random.randn(50, 3))

        st.write("This is outside the container")


# st.set_page_config(page_title="Descriptive Analysis Demo", page_icon="📈")
