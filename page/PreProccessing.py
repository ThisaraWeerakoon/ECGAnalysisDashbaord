import streamlit as st
import pandas as pd
import pydeck as pdk
from urllib.error import URLError
from utils.page import Page
from sklearn.preprocessing import MinMaxScaler
from scipy.signal import butter, filtfilt

class Preprocess(Page):

    def band_pass_filter(self,data, cutoff, fs, btype='low', order=5):
        nyquist = 0.5 * fs

        # Check if cutoff is a list (indicating bandpass or bandstop filter)
        if isinstance(cutoff, list):
            normal_cutoff = [freq / nyquist for freq in cutoff]
        else:
            normal_cutoff = cutoff / nyquist

        b, a = butter(order, normal_cutoff, btype=btype, analog=False)
        filtered_data = filtfilt(b, a, data)

        return filtered_data
    def __init__(self, data, **kwargs):
        name = "About"
        super().__init__(name, data, **kwargs)

    def content(self):
        # st.set_page_config(page_title="Mapping Demo", page_icon="🌍")

        # st.markdown("# Mapping Demo")
        # st.sidebar.header("Mapping Demo")
        # st.write(self.data['record'])
        filtered_placeholder = st.empty()

        options = st.multiselect(
            "Choose countries", ["Normalization","Baseline Removal","HighPassFilter"],  ["Normalization","Baseline Removal","HighPassFilter"]
        )

        last_rows = self.data["record"].p_signal[:2000, 0]
        filtered_placeholder.line_chart(last_rows)



        if "HighPassFilter" in options:
            last_rows = self.band_pass_filter(self.data["record"].p_signal[:2000, 0], cutoff=[0.5, 50], fs=360, btype='band')

        if "Normalization" in options:
            # st.sidebar.progress(0)
            print("Normalization")
            scaler = MinMaxScaler()
            last_rows = scaler.fit_transform(last_rows.reshape(-1, 1))

        filtered_placeholder.line_chart(last_rows)

        return last_rows



        # def update_filtered_plot(lowcut, highcut):
        #     filtered_signal = apply_bandpass_filter(signal_data, lowcut, highcut, fs)
        #     filtered_placeholder.line_chart(filtered_signal, use_container_width=True)




