import streamlit as st
from page.PreProccessing import Preprocess
import torch
import os
import wfdb
import pandas as pd
from electroCardioGuard.PersonIdentification import PersonIdentification
from page.Descriptive_Analysis import DescriptiveAnalysis
from page.Predictive_Analysis import PredictiveAnalysis
from page.Myocardial_Infarction import CNN_LSTM, load_model, predict, encoder,MyocardialInfarction
from streamlit_extras.metric_cards import style_metric_cards

st.set_page_config(layout="wide")
tab1, tab2, tab3 ,tab4,tab5= st.tabs(["📈 Preprocessing", "🗃 Descriptive Analysis", "📊Arrhythmia Detection","📊Mycardioc Infarction Detection", "📊Person Identification"])
record = None
# Initialize saved_signals in session state if not already present
if 'saved_signals' not in st.session_state:
    st.session_state.saved_signals = []

def save_signal(option, signal):
    update = False
    for s in st.session_state.saved_signals:
        if s["patient_id"] == option :
            update = True
            s["signal"] = signal
            break
    if not update:
        t = {"patient_id": option, "signal": signal}
        st.session_state.saved_signals.append(t)
    return st.session_state.saved_signals

with tab1:
    # select_col1, select_col2 = st.columns([0.5, 0.5])
    #
    # with select_col1:
    #     option = st.selectbox("Select A patient Record", ("100", "101", "102", "103", "104", "105", "106"))
    #
    # with select_col2:
    #     channel = st.selectbox("Select A patient Channel", (0,1))
    #

    # Instructions
    st.write("""
        Upload your ECG data files (WFDB format) for analysis.
    """)

    # File uploader
    uploaded_files = st.file_uploader("Choose ECG data files", type=["dat", "hea"], accept_multiple_files=True)

    if uploaded_files:
        for uploaded_file in uploaded_files:
            if uploaded_file.type == "application/octet-stream":  # Check if it's a 'dat' file
                dat_file = uploaded_file
                hea_file = next((f for f in uploaded_files if f.name == uploaded_file.name.replace('.dat', '.hea')),
                                None)
                if hea_file:
                    st.write(f"Uploaded files: {hea_file.name}, {dat_file.name}")
                    if hea_file.name == dat_file.name:
                        break
                    print(f"Uploaded files: {hea_file.name}, {dat_file.name}")

                    # Save uploaded files temporarily
                    with open(dat_file.name, 'wb') as f:
                        f.write(dat_file.read())
                    with open(hea_file.name, 'wb') as f:
                        f.write(hea_file.read())

                    # Read the record using wfdb
                    record = wfdb.rdrecord(dat_file.name.replace('.dat', ''))

                    # # Display basic information
                    # st.write("Record Information:")
                    # st.write(f"Sampling Frequency: {record.fs} Hz")
                    # st.write(f"Number of Signals: {record.n_sig}")
                    # st.write(f"Signal Length: {record.sig_len} samples")
                    #
                    # # Plot the first signal
                    # st.write("ECG Signal:")
                    # plt.figure(figsize=(10, 4))
                    # plt.plot(record.p_signal[:, 0])
                    # plt.xlabel('Sample')
                    # plt.ylabel('Amplitude')
                    # plt.title('ECG Signal')
                    # st.pyplot(plt)
                    #
                    # # Optionally, process and display additional information
                    # # For example, display the first 1000 points of the first signal
                    # st.write("First 1000 samples of the ECG signal:")
                    # st.line_chart(record.p_signal[:1000, 0])

                    # Clean up temporary files
                    os.remove(dat_file.name)
                    os.remove(hea_file.name)

    #
                    # data = f"mit-bih-arrhythmia-database-1.0.0/{option}"
                    # record = wfdb.rdrecord(data, smooth_frames=True)
                    DATA = {"record": record, "channel": 0, "saved_signals": st.session_state.saved_signals}

                    page = Preprocess(DATA)
                    signal = page.content()

                    if st.button('Save'):
                        st.session_state.saved_signals = save_signal(dat_file.name.split('.')[0], signal)

    rows = [[s["patient_id"]] for s in st.session_state.saved_signals]
    df1 = pd.DataFrame(rows, columns=["patient_id"])
    st.table(df1)

with tab2:
    if record is None:
        st.error("Please upload a record first.")
    else:

        DATA = {"record": record, "signal": signal,'saved_signals': st.session_state.saved_signals}
        st.title("ECG Signal Analysis")
        page = DescriptiveAnalysis(DATA)
        page.content()


with tab3:
    if record is None:
        st.error("Please upload a record first.")
    else:
        DATA = {"record": record, "signal": signal,'saved_signals': st.session_state.saved_signals}
    # page = PredictiveAnalysis(DATA)
    # page.content()

with tab4:
    if record is None:
        st.error("Please upload a record first.")
    else:
        DATA = {"record": record, "signal": signal,'saved_signals': st.session_state.saved_signals}
        page = MyocardialInfarction(DATA)
        page.content()

with tab5:
    if record is None:
        st.error("Please upload a record first.")
    else:
        DATA = {"record": record, "signal": signal,'saved_signals': st.session_state.saved_signals}
        page = PersonIdentification(DATA)
        page.content()
