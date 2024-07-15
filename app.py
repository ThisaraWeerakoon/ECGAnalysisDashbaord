import streamlit as st
from page.PreProccessing import Preprocess
import torch
import wfdb
import pandas as pd
from electroCardioGuard.PersonIdentification import PersonIdentification
from page.Descriptive_Analysis import DescriptiveAnalysis
from page.Predictive_Analysis import PredictiveAnalysis
from page.Myocardial_Infarction import CNN_LSTM, load_model, predict, encoder,MyocardialInfarction
from streamlit_extras.metric_cards import style_metric_cards

st.set_page_config(layout="wide")
tab1, tab2, tab3 ,tab4,tab5= st.tabs(["📈 Preprocessing", "🗃 Descriptive Analysis", "📊Arrhythmia Detection","📊Mycardioc Infarction Detection", "📊Person Identification"])

# Initialize saved_signals in session state if not already present
if 'saved_signals' not in st.session_state:
    st.session_state.saved_signals = []

def save_signal(option, channel, signal):
    update = False
    for s in st.session_state.saved_signals:
        if s["patient_id"] == option and s["channel"] == channel:
            update = True
            s["signal"] = signal
            break
    if not update:
        t = {"patient_id": option, "channel": channel, "signal": signal}
        st.session_state.saved_signals.append(t)
    return st.session_state.saved_signals

with tab1:
    select_col1, select_col2 = st.columns([0.5, 0.5])

    with select_col1:
        option = st.selectbox("Select A patient Record", ("100", "101", "102", "103", "104", "105", "106"))

    #
    data = f"mit-bih-arrhythmia-database-1.0.0/{option}"
    record = wfdb.rdrecord(data, smooth_frames=True)
    DATA = {"record": record, "channel": 0, "saved_signals": st.session_state.saved_signals}

    page = Preprocess(DATA)
    signal = page.content()

    if st.button('Save'):
        st.session_state.saved_signals = save_signal(option, channel, signal)

    rows = [[s["patient_id"], s["channel"]] for s in st.session_state.saved_signals]
    df1 = pd.DataFrame(rows, columns=["patient_id", "channel"])
    st.table(df1)

with tab2:
    DATA = {"record": record, "signal": signal,'saved_signals': st.session_state.saved_signals}
    st.title("ECG Signal Analysis")
    page = DescriptiveAnalysis(DATA)
    page.content()


with tab3:
    DATA = {"record": record, "signal": signal,'saved_signals': st.session_state.saved_signals}
    # page = PredictiveAnalysis(DATA)
    # page.content()

with tab4:
    DATA = {"record": record, "signal": signal,'saved_signals': st.session_state.saved_signals}
    page = MyocardialInfarction(DATA)
    page.content()

with tab5:
    DATA = {"record": record, "signal": signal,'saved_signals': st.session_state.saved_signals}
    page = PersonIdentification(DATA)
    page.content()
