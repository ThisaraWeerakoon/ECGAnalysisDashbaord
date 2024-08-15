import streamlit as st
import numpy as np
import neurokit2 as nk
import plotly.graph_objects as go
from utils.page import Page
from streamlit_extras.metric_cards import style_metric_cards
import random


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

        colors = ['green', 'yellow', 'red', 'purple']  # Red, Green, Blue with transparency

        # Define the numbers and their corresponding probabilities
        numbers = [0, 1, 2, 3]
        weights = [0.7, 0.1, 0.1, 0.1]  # 70% for 0, 10% for 1, 10% for 2, 10% for 3

        random_numbers = random.choices(numbers, weights, k=len(epochs.keys()))

        # Add vertical dotted lines
        for count, key in enumerate(epochs.keys()):
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

            fig.add_shape(
                type='rect',
                x0=epochs[key]["Index"].to_list()[0],
                x1=epochs[key]["Index"].to_list()[-1],
                y0=-0.2,
                y1=0.7,
                fillcolor=colors[random_numbers[count]],
                opacity=0.5,
                layer='below',
                line_width=0,
            )

        '''
        # Add legend annotations
        legend_annotations = [
            dict(
                x=1.05,
                y=1.1 - i * 0.1,
                xref='paper',
                yref='paper',
                showarrow=False,
                text=f'Category {i + 1}',
                font=dict(color=colors[i]),
                bgcolor='white',
                bordercolor=colors[i]
            ) for i in range(len(colors))
        ]
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
            height=700,
            # annotations=legend_annotations,
            # showlegend=True  # Add annotations to layout
        )

        st.plotly_chart(fig, use_container_width=True)

        total1, total2, total3, total4, total5 = st.columns(5, gap='medium')

        with total1:
            st.metric(label="Total number of beats", value=f"{len(random_numbers)}")

        with total2:
            # st.metric(label="Normal beat count", value=f"{random_numbers.count(0)}")

            st.markdown(
                f"""
                <div style="background-color:green; padding:10px; border-radius:5px;">
                <h3 style="color:white;">N beat count</h3>
                <p style="color:white; font-size:24px;">{random_numbers.count(0)}</p>
                </div>
                """, unsafe_allow_html=True
            )

        with total3:
            # st.metric(label="S beat count", value=f"{random_numbers.count(1)}")
            st.markdown(
                f"""
        <div style="background-color:yellow; padding:10px; border-radius:5px;">
            <h3 style="color:black;">S beat count</h3>
            <p style="color:black; font-size:24px;">{random_numbers.count(1)}</p>
        </div>
        """, unsafe_allow_html=True
            )

        with total4:
            # st.metric(label="V beat count", value=f"{random_numbers.count(2)}")
            st.markdown(
                f"""
        <div style="background-color:lightcoral; padding:10px; border-radius:5px;">
            <h3 style="color:white;">V beat count</h3>
            <p style="color:white; font-size:24px;">{random_numbers.count(2)}</p>
        </div>
        """, unsafe_allow_html=True
            )

        with total5:
            # st.metric(label="F beat count", value=f"{random_numbers.count(3)}")
            st.markdown(
                f"""
        <div style="background-color:plum; padding:10px; border-radius:5px;">
            <h3 style="color:white;">F beat count</h3>
            <p style="color:white; font-size:24px;">{random_numbers.count(3)}</p>
        </div>
        """, unsafe_allow_html=True
            )

        style_metric_cards(background_color="white", border_left_color="#89CFF0", border_color="#89CFF0",
                           box_shadow="#F71938")