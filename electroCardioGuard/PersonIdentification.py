# from models.pt_classifier import Classifier
from utils.page import Page
import matplotlib.pyplot as plt
import torch

# Load your ML model (replace with your actual model loading code)
# model = load_model('path_to_your_model')


def plot_ecg(signal, title):
    plt.figure(figsize=(10, 4))
    plt.plot(signal)
    plt.title(title)
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    st.pyplot(plt)


class PersonIdentification(Page):
    def __init__(self, data, **kwargs):
        name = "About"
        super().__init__(name, data, **kwargs)

    def content(self):
        patient_ids= [s["patient_id"] for s in st.session_state.saved_signals]
        # patient_channels = [s["channel"] for s in st.session_state.saved_signals]

        st.title('ECG Signal Comparison')
        predict_col1,predict_col2 = st.columns([0.5,0.5])
        with predict_col1:

            st.write("Choose the first ECG signal")

            p1_select_col1, p1_select_col2 = st.columns([0.5, 0.5])

            # with p1_select_col1:
            ecg1_patient_id = st.selectbox("Select A patient Record For ECG A", set(patient_ids))
            #
            # with p1_select_col2:
            #     ecg1_channel = st.selectbox("Select A patient Channel For ECG A", set(patient_channels))
            st.subheader('ECG Signal 1')
            ecg_signal_1 = [s["signal"] for s in st.session_state.saved_signals if
                            s["patient_id"] == ecg1_patient_id ]

            ## convert to numpy array
            # ecg_signal_1 = ecg_signal_1[0]
            if len(ecg_signal_1) == 0:
                st.write("No signal found for the selected patient and channel.")
                return
            st.line_chart(ecg_signal_1[0])
        with predict_col2:


            st.write("Choose the first ECG signal")

            # p2_select_col1, p2_select_col2 = st.columns([0.5, 0.5])

            # with p2_select_col1:
            ecg2_patient_id = st.selectbox("Select A patient Record  for ECG B", set(patient_ids))

            # with p2_select_col2:
            #     ecg2_channel = st.selectbox("Select A patient Channel For ECG B", set(patient_channels))

            st.subheader('ECG Signal 2')
            ecg_signal_2 = [s["signal"] for s in st.session_state.saved_signals if
                            s["patient_id"] == ecg2_patient_id ]

            if len(ecg_signal_2) == 0:
                st.write("No signal found for the selected patient and channel.")
                return
            st.line_chart(ecg_signal_2[0])

        # File uploader for ECG signals
        # uploaded_file1 = st.file_uploader("Choose the first ECG signal", type=["csv", "txt"])
        # uploaded_file2 = st.file_uploader("Choose the second ECG signal", type=["csv", "txt"])





        # plot_ecg(ecg_signal1, 'ECG Signal 1')


        # Predict if the signals are from the same person
        if st.button('Predict'):
            # prediction = predict_same_person(model, ecg_signal1,
            #                                  ecg_signal2)  # Replace with your actual prediction function

            print(ecg_signal_1[0].shape)
            print(type(ecg_signal_2[0]))
            ecg_signal1_expanded = np.tile(ecg_signal_1[0], (1, 12))
            ecg_signal2_expanded = np.tile(ecg_signal_2[0], (1, 12))

            # Convert the expanded NumPy arrays to PyTorch tensors
            tensor1 = torch.tensor(ecg_signal1_expanded, dtype=torch.float32)
            tensor2 = torch.tensor(ecg_signal2_expanded, dtype=torch.float32)

            # Stack the tensors along a new axis to get shape (2, 4096, 12)
            combined_tensor = torch.stack((tensor1, tensor2))

            embedder = torch.load("electroCardioGuard/best_model/best_embedding")
            discriminator = torch.load("electroCardioGuard/best_model/best_discriminator")
            # c = Classifier(embedder, discriminator, "a3")
            torch.manual_seed(7)
            embeedings=embedder(combined_tensor.transpose(-1, -2))
            print(embeedings[0].shape)

            ecg_1_embedding = embeedings[0]
            ecg_2_embedding = embeedings[1]
            a = torch.stack([ecg_1_embedding, ecg_2_embedding])

            print(discriminator(ecg_1_embedding, a)[1])





            prediction = discriminator(ecg_1_embedding, a)[1].detach().numpy()

            st.subheader('Prediction Result')
            if prediction is not None:
                same_person_prob = prediction
                different_person_prob = 1 - same_person_prob

                st.write(f"Probability they are from the same person: {same_person_prob:.2f}")
                if same_person_prob > 0.90:
                    st.write("The signals are from the same person.")
                else:
                    st.write("The signals are from different persons.")
                # st.write(f"Probability they are from different persons: {different_person_prob:.2f}")
                #
                # # Bar graph of the prediction
                # labels = ['Same Person', 'Different Persons']
                # probabilities = [same_person_prob, different_person_prob]
                #
                # fig, ax = plt.subplots()
                # ax.barh(labels, probabilities)
                # ax.set_ylim([0, 1])
                # ax.set_ylabel('Probability')
                # st.pyplot(fig)
            else:
                st.write("Prediction could not be made. Check your model and input signals.")


import streamlit as st
import numpy as np

# You can run this script using the command: streamlit run ecg_compare.py
