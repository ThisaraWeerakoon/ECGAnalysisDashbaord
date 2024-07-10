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
        st.markdown("# Plotting Demo")
        st.sidebar.header("Plotting Demo")
        st.write(
            """This demo illustrates a combination of plotting and animation with
        Streamlit. We're generating a bunch of random numbers in a loop for around
        5 seconds. Enjoy!"""
        )




        progress_bar = st.sidebar.progress(0)
        status_text = st.sidebar.empty()
        last_rows = self.data["record"].p_signal[:1000, 0]
        chart = st.line_chart(last_rows)

        # for i in range(1, 101):
        #     new_rows = last_rows[-1, :] + np.random.randn(5, 1).cumsum(axis=0)
        #     status_text.text("%i%% Complete" % i)
        #     chart.add_rows(new_rows)
        #     progress_bar.progress(i)
        #     last_rows = new_rows
        #     time.sleep(0.05)

        progress_bar.empty()

        # Streamlit widgets automatically run the script from top to bottom. Since
        # this button is not connected to any other logic, it just causes a plain
        # rerun.
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
