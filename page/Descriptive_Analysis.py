import streamlit as st
import numpy as np
import neurokit2 as nk
import plotly.graph_objects as go
from utils.page import Page
from streamlit_extras.metric_cards import style_metric_cards

class DescriptiveAnalysis(Page):
    def __init__(self, data, **kwargs):
        name = "About"
        super().__init__(name, data, **kwargs)

    def content(self):
        patient_ids= [s["patient_id"] for s in st.session_state.saved_signals]
        # patient_channels = [s["channel"] for s in st.session_state.saved_signals]
        # select_col1, select_col2 = st.columns([0.5, 0.5])

        option = st.selectbox("Select A patient Record", set(patient_ids))

        # with select_col2:
        #     channel = st.selectbox("Select A patient channel", set(patient_channels))
        # patient_ids = [s["patient_id"] for s in st.session_state.saved_signals]
        # patient_channels = [s["channel"] for s in st.session_state.saved_signals]
        # select_col1, select_col2 = st.columns([0.5, 0.5])

        # with select_col1:
        #     option = st.selectbox("Select A patient Record", set(patient_ids))

        # with select_col2:
        #     channel = st.selectbox("Select A patient Channel", set(patient_channels))

        st.markdown("ECG signal with peaks")
        # Get signal from saved signals
        for s in st.session_state.saved_signals:
            if s["patient_id"] == option :
                self.data["signal"] = s["signal"]
                break
        last_rows = self.data["signal"]

        if last_rows.ndim > 1:
            last_rows = last_rows.flatten()
        signals, info = nk.ecg_process(last_rows, sampling_rate=360)
        # st.write("ECG Signal Processing Information ")
        # st.write(info)
        # print(info)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=np.arange(len(last_rows)), y=last_rows, mode='lines', name='ECG Signal'))

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
        # Add red grid background
        for x in np.arange(0, len(last_rows), 50):
            fig.add_shape(type="line", x0=x, y0=min(last_rows), x1=x, y1=max(last_rows),
                          line=dict(color="red", width=0.5, dash="dash"))
        for y in np.arange(min(last_rows), max(last_rows), (max(last_rows) - min(last_rows)) / 10):
            fig.add_shape(type="line", x0=0, y0=y, x1=len(last_rows), y1=y,
                          line=dict(color="red", width=0.5, dash="dash"))

        fig.update_layout(
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
        #
        # total1, total2, total3 = st.columns(3, gap='medium')
        #
        # with total1:
        #     if 'ECG_R_Peaks' in info:
        #         num_beats = len(info['ECG_R_Peaks'])
        #         duration_minutes = len(last_rows) / 360 / 60
        #         heart_rate = num_beats / duration_minutes
        #         st.metric(label="Heart Rate (bpm)", value=f"{heart_rate:.2f}")
        total1, total2, total3,total4 = st.columns(4 ,gap='medium')

        with total2:
            if 'ECG_R_Peaks' in info:
                num_beats = len(info['ECG_R_Peaks'])
                duration_minutes = len(last_rows) / 360 / 60
                heart_rate = num_beats / duration_minutes
                st.metric(label="Heart Rate (bpm)", value=f"{heart_rate:.2f}")

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

        st.markdown("ECG Signal Time Intervals")
        col1, col2 = st.columns([0.7, 0.3])
        with col1:
            # fig_segment = go.Figure()
            p_start =info['ECG_P_Peaks'][0]-10
            t_start = info['ECG_T_Peaks'][0]+10

            fig_segment = fig.update_xaxes(range=[p_start, t_start],rangeslider_visible=False)

            p_start =info['ECG_P_Peaks'][0]
            t_start = info['ECG_T_Peaks'][0]

            def extract_peak_coordinates(peak_key, x_range_start, x_range_end):
                peak_coordinates = []
                if peak_key in info and len(info[peak_key]) > 0:
                    valid_indices = [i for i in info[peak_key] if i >= x_range_start and i <= x_range_end and i < len(last_rows)]
                    for idx in valid_indices:
                        peak_coordinates.append((idx, last_rows[idx]))
                return peak_coordinates

            p_peak_coordinates = extract_peak_coordinates('ECG_P_Peaks', p_start, t_start)
            q_peak_coordinates = extract_peak_coordinates('ECG_Q_Peaks', p_start, t_start)
            r_peak_coordinates = extract_peak_coordinates('ECG_R_Peaks', p_start, t_start)
            s_peak_coordinates = extract_peak_coordinates('ECG_S_Peaks', p_start, t_start)
            t_peak_coordinates = extract_peak_coordinates('ECG_T_Peaks', p_start, t_start)

            # print("P Peak Coordinates:", p_peak_coordinates)
            # print("Q Peak Coordinates:", q_peak_coordinates)
            # print("R Peak Coordinates:", r_peak_coordinates)
            # print("S Peak Coordinates:", s_peak_coordinates)
            # print("T Peak Coordinates:", t_peak_coordinates)

            fig_segment.add_shape(type="rect", xref="x", yref="paper", x0=p_peak_coordinates[0][0], y0=0, x1=q_peak_coordinates[0][0], y1=1, fillcolor="green", opacity=0.2)
            fig_segment.add_shape(type="rect", xref="x", yref="paper", x0=q_peak_coordinates[0][0], y0=0, x1=s_peak_coordinates[0][0], y1=1, fillcolor="yellow", opacity=0.2)
            fig_segment.add_shape(type="rect", xref="x", yref="paper", x0=s_peak_coordinates[0][0], y0=0, x1=t_peak_coordinates[0][0], y1=1, fillcolor="red", opacity=0.2)

            fig_segment.add_annotation(x=(s_peak_coordinates[0][0] + t_peak_coordinates[0][0]) / 2, y=1.1, text="ST Interval", showarrow=False, font=dict(color="black", size=12))
            fig_segment.add_annotation(x=(p_peak_coordinates[0][0] + q_peak_coordinates[0][0]) / 2, y=1.1, text="PQ Interval", showarrow=False, font=dict(color="black", size=12))
            fig_segment.add_annotation(x=(q_peak_coordinates[0][0] + s_peak_coordinates[0][0]) / 2, y=1.1, text="QRS Complex", showarrow=False, font=dict(color="black", size=12))


            fig_segment.update_layout(
                width=100,
                height=500
            )


            st.plotly_chart(fig_segment, use_container_width=True)

        with col2:
            st.markdown("")
            st.markdown("")
            st.markdown("")
            st.markdown("")
            st.markdown("")

            # P-Q
            if 'ECG_P_Peaks' in info and 'ECG_Q_Peaks' in info:
                pq_intervals = []
                for i in range(min(len(info['ECG_P_Peaks']), len(info['ECG_Q_Peaks']))):
                    if info['ECG_Q_Peaks'][i] > info['ECG_P_Peaks'][i]:
                        pq_intervals.append(info['ECG_Q_Peaks'][i] - info['ECG_P_Peaks'][i])
                if pq_intervals:
                    average_pq_interval = np.mean(pq_intervals)
                    st.metric(label="Average PQ Interval (ms)", value=f"{average_pq_interval:.2f}")
                else:
                    st.warning("Not enough data to calculate PQ interval.")

            # Q-R-S
            if num_beats > 1:
                    qrs_intervals = np.diff(info['ECG_R_Peaks'])
                    average_qrs_interval = np.mean(qrs_intervals)
                    st.metric(label="Average QRS Interval (ms)", value=f"{average_qrs_interval:.2f}")
            else:
                    st.warning("Not enough data to calculate QRS interval.")


            #S-T
            if 'ECG_S_Peaks' in info and 'ECG_T_Peaks' in info:
                st_intervals = []
                for i in range(min(len(info['ECG_S_Peaks']), len(info['ECG_T_Peaks']))):
                    if info['ECG_T_Peaks'][i] > info['ECG_S_Peaks'][i]:
                        st_intervals.append(info['ECG_T_Peaks'][i] - info['ECG_S_Peaks'][i])
                if st_intervals:
                    average_st_interval = np.mean(st_intervals)
                    st.metric(label="Average ST Interval (ms)", value=f"{average_st_interval:.2f}")
                else:
                    st.warning("Not enough data to calculate ST interval.")




