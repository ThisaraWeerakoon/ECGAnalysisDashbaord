import streamlit as st
import numpy as np
import neurokit2 as nk
import plotly.graph_objects as go
from utils.page import Page  
from streamlit_extras.metric_cards import style_metric_cards  

class ArrhythmiaAnalysis(Page):
    def __init__(self, data, **kwargs):
        name = "About"
        super().__init__(name, data, **kwargs)

    def content(self):
        last_rows = self.data["signal"]

        slider = self.data["signal"]

        if last_rows.ndim > 1:
            last_rows = last_rows.flatten()
        signals, info = nk.ecg_process(last_rows, sampling_rate=360)
        cleaned_ecg = signals["ECG_Clean"]
        epochs = nk.ecg_segment(cleaned_ecg, rpeaks=None, sampling_rate=250, show=True)
        for key in epochs.keys():
            print(epochs[key]["Index"].to_list()[-1])

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=list(cleaned_ecg.index), y=cleaned_ecg.to_list(), mode='lines', name='ECG Signal'))

        # Add vertical dotted lines
        for key in epochs.keys():
            fig.add_shape(
                dict(
                    type='line',
                    x0=epochs[key]["Index"].to_list()[-1],
                    y0=-0.2,
                    x1=epochs[key]["Index"].to_list()[-1],
                    y1=0.7,
                    line=dict(
                        color='red',
                        width=2,
                        dash='dot',
                    ),
                )
            )
            if int(epochs[key]["Index"].to_list()[-1]) % 2==0:
                fig.add_trace(go.Scatter(x=epochs[key]["Index"].to_list(), y=epochs[key]["Signal"].to_list(), mode='lines', name='ECG Signal',line=dict(color='red')))

        '''

        def add_peak_trace(peak_key, color, name):
            if peak_key in info and len(info[peak_key]) > 0:
                valid_indices = [i for i in info[peak_key] if i < len(last_rows)]
                fig.add_trace(go.Scatter(
                    x=valid_indices, y=last_rows[valid_indices], mode='markers', marker=dict(color=color),
                    name=name))

        add_peak_trace('ECG_P_Peaks', 'red', 'P Peaks')
        add_peak_trace('ECG_Q_Peaks', 'green', 'Q Peaks')
        add_peak_trace('ECG_R_Peaks', 'blue', 'R Peaks')
        add_peak_trace('ECG_S_Peaks', 'purple', 'S Peaks')
        add_peak_trace('ECG_T_Peaks', 'orange', 'T Peaks')

        '''


        fig.update_layout(
            title='ECG Signal with',
            xaxis_title='Sample',
            yaxis_title='Amplitude',
            xaxis=dict(
                rangeslider=dict(
                    visible=True,
                    thickness=0.07
                ),
                type='linear'
            ),
            yaxis=dict(
                fixedrange=True
            ),
            width=800,  
            height=700 
        )
    
        st.plotly_chart(fig, use_container_width=True)

        total1, total2, total3 = st.columns(3, gap='medium')

        with total1:
            if 'ECG_R_Peaks' in info:
                num_beats = len(info['ECG_R_Peaks'])
                duration_minutes = len(last_rows) / 360 / 60  
                heart_rate = num_beats / duration_minutes
                st.metric(label="Heart Rate (bpm)", value=f"{heart_rate:.2f}")

        with total2:
            if num_beats > 1:
                qrs_intervals = np.diff(info['ECG_R_Peaks'])
                average_qrs_interval = np.mean(qrs_intervals)
                st.metric(label="Average QRS Interval (ms)", value=f"{average_qrs_interval:.2f}")
            else:
                st.warning("Not enough data to calculate QRS interval.")

        with total3:
            if 'ECG_R_Peaks' in info:
                r_peaks = info['ECG_R_Peaks']
                rr_intervals = np.diff(r_peaks)  
                rr_std = np.std(rr_intervals)
                threshold = 0.1
                if rr_std < threshold:  
                    rhythm_regularity = "Regular"
                else:
                    rhythm_regularity = "Irregular"
                st.metric(label="Rhythm Regularity", value=rhythm_regularity)



        style_metric_cards(background_color="white", border_left_color="#89CFF0", border_color="#89CFF0",
                          box_shadow="#F71938")


