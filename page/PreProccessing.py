import streamlit as st
import pandas as pd
import pydeck as pdk
from urllib.error import URLError
from utils.page import Page
from sklearn.preprocessing import MinMaxScaler
from scipy.signal import butter, filtfilt, iirnotch
import numpy as np

class Preprocess(Page):

    def highpass_filter(self,data, cutoff, fs, order=5):
        nyquist = 0.5 * fs
        normal_cutoff = cutoff / nyquist
        b, a = butter(order, normal_cutoff, btype='high', analog=False)
        filtered_data = filtfilt(b, a, data)
        return filtered_data

    ## high pass filter
    def highpass_filter(self,data, cutoff, fs, order=5):
        nyquist = 0.5 * fs
        normal_cutoff = cutoff / nyquist
        b, a = butter(order, normal_cutoff, btype='high', analog=False)
        filtered_data = filtfilt(b, a, data)
        return filtered_data
    def lowpass_filter(self,data, cutoff, fs, order=5):
        nyquist = 0.5 * fs
        normal_cutoff = cutoff / nyquist
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        filtered_data = filtfilt(b, a, data)
        return filtered_data

    def notch_filter(self,data, cutoff, fs, Q=30):
        nyquist = 0.5 * fs
        normal_cutoff = cutoff / nyquist
        b, a = iirnotch(normal_cutoff, Q)
        filtered_data = filtfilt(b, a, data)
        return filtered_data



    def __init__(self, data, **kwargs):
        name = "About"
        super().__init__(name, data, **kwargs)

    def content(self):
        # st.set_page_config(page_title="Mapping Demo", page_icon="🌍")
        main_col1, main_col2 = st.columns([0.7, 0.3])
        # st.markdown("# Mapping Demo")
        # st.sidebar.header("Mapping Demo")
        # st.write(self.data['record'])

        last_rows = self.data["record"].p_signal[:4096, self.data["channel"]]
        with main_col1:
            filtered_placeholder = st.empty()


            # st.write( self.data["record"].p_signal.shape)

            fs = 360
            t = np.linspace(0, 4096/360, 4096)  # Time vector of 1 second
            # print(t)
            # print(len(ecg_signal))
            power_line_frequency = 50  # Power line frequency (Hz)
            pli_amplitude = 0.1  # Amplitude of the PLI
            pli_signal = pli_amplitude * np.sin(2 * np.pi * power_line_frequency * t)
            last_rows = last_rows + pli_signal
            filtered_placeholder.line_chart(last_rows)
        with main_col2:
            options = st.multiselect(
                "Select Processing" , ["Normalization", "Baseline Removal", "LowPassFilter", "Notch Filter" ,"HighPassFilter"],
                ["Normalization", "Baseline Removal", "LowPassFilter", "Notch Filter"]
            )
            # st.markdown("---")
            if "LowPassFilter" in options:
                st.markdown('<p style="font-size:20px;font-weight:bold;">Low Pass Filter</p>',
                            unsafe_allow_html=True)
                # st.write("Low Pass Filter")
                values = st.slider(
                    "Select a band of Frequncy Hz ",
                    0, 100, 25)
                st.write("Frequency band: 0", values)
                last_rows = self.lowpass_filter(self.data["record"].p_signal[:4096, self.data["channel"]], cutoff=values, fs=360)

            if "HighPassFilter" in options:
                st.markdown('<p style="font-size:20px;font-weight:bold;">High Pass Filter</p>',
                            unsafe_allow_html=True)
                # st.write("Low Pass Filter")
                high_pass = st.slider(
                    "Select a band of Frequency Hz ",
                    100, 1000, 25)
                st.write("Frequency band: 0", values)
                last_rows = self.highpass_filter(self.data["record"].p_signal[:4096, self.data["channel"]],
                                                cutoff=high_pass, fs=360)

                # You can call any Streamlit command, including custom components:
                # st.bar_chart(np.random.randn(50, 3))

            if "Notch Filter" in options:
                st.markdown('<p style="font-size:20px;font-weight:bold;">Notch Filter</p>',
                            unsafe_allow_html=True)

                # st.write("Notch  Filter")

                number = st.number_input("Set Notch frequency" ,50)
                # print(number)

                last_rows = self.notch_filter(last_rows, cutoff=int(number), fs=360)


            if "Normalization" in options:
                # st.sidebar.progress(0)
                # print("Normalization")
                scaler = MinMaxScaler()
                last_rows = scaler.fit_transform(last_rows.reshape(-1, 1))
                # st.write("This is inside the container")
                # normal = st.slider(
                #     "Select a range of values",
                #     0.0, 1.0, (25.0, 75.0))
                # st.write("Values:", normal)

            filtered_placeholder.line_chart(last_rows)

        return last_rows



        # def update_filtered_plot(lowcut, highcut):
        #     filtered_signal = apply_bandpass_filter(signal_data, lowcut, highcut, fs)
        #     filtered_placeholder.line_chart(filtered_signal, use_container_width=True)




