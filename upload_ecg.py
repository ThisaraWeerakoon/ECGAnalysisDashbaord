import streamlit as st
import wfdb
import numpy as np
import matplotlib.pyplot as plt
import os
# Title of the web app
st.title('ECG Data Upload and Visualization')

# Instructions
st.write("""
    Upload your ECG data files (WFDB format) for analysis.
""")

# File uploader
uploaded_files = st.file_uploader("Choose ECG data files", type=["dat", "hea"],   accept_multiple_files=True)

if uploaded_files:
    for uploaded_file in uploaded_files:
        if uploaded_file.type == "application/octet-stream":  # Check if it's a 'dat' file
            dat_file = uploaded_file
            hea_file = next((f for f in uploaded_files if f.name == uploaded_file.name.replace('.dat', '.hea')), None)
            if hea_file:
                st.write(f"Uploaded files: {hea_file.name}, {dat_file.name}")
                if hea_file.name==dat_file.name:
                    break
                print(f"Uploaded files: {hea_file.name}, {dat_file.name}")

                # Save uploaded files temporarily
                with open(dat_file.name, 'wb') as f:
                    f.write(dat_file.read())
                with open(hea_file.name, 'wb') as f:
                    f.write(hea_file.read())

                # Read the record using wfdb
                record = wfdb.rdrecord(dat_file.name.replace('.dat', ''))

                # Display basic information
                st.write("Record Information:")
                st.write(f"Sampling Frequency: {record.fs} Hz")
                st.write(f"Number of Signals: {record.n_sig}")
                st.write(f"Signal Length: {record.sig_len} samples")

                # Plot the first signal
                st.write("ECG Signal:")
                plt.figure(figsize=(10, 4))
                plt.plot(record.p_signal[:, 0])
                plt.xlabel('Sample')
                plt.ylabel('Amplitude')
                plt.title('ECG Signal')
                st.pyplot(plt)

                # Optionally, process and display additional information
                # For example, display the first 1000 points of the first signal
                st.write("First 1000 samples of the ECG signal:")
                st.line_chart(record.p_signal[:1000, 0])

                # Clean up temporary files
                os.remove(dat_file.name)
                os.remove(hea_file.name)

