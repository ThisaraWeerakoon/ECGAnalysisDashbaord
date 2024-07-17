# myocardial_infarction.py
import torch
import numpy as np
from torch import nn
from sklearn.preprocessing import LabelEncoder
import wfdb
from utils.page import Page
import streamlit as st
from streamlit_extras.metric_cards import style_metric_cards

# Define the CNN class
class CNN(nn.Module):
    def __init__(self, hidden):
        super(CNN, self).__init__()
        self.hiddenLast = hidden
        self.hidden1 = 187
        self.hidden2 = 64
        self.hidden3 = 64
        self.hidden4 = 64
        self.hidden5 = 64
        self.hidden6 = 64
        self.kernel_size = 2
        self.activation = nn.ReLU()
        self.first = True

        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=self.hidden1, kernel_size=self.kernel_size),
            nn.BatchNorm1d(num_features=self.hidden1),
            self.activation,
        )

        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels=self.hidden1, out_channels=self.hidden2, kernel_size=self.kernel_size),
            nn.BatchNorm1d(num_features=self.hidden2),
            self.activation,
            nn.MaxPool1d(kernel_size=2)
        )

        self.conv3 = nn.Sequential(
            nn.Conv1d(in_channels=self.hidden2, out_channels=self.hidden3, kernel_size=self.kernel_size),
            nn.BatchNorm1d(num_features=self.hidden3),
            self.activation,
            nn.MaxPool1d(kernel_size=2)
        )

        self.conv4 = nn.Sequential(
            nn.Conv1d(in_channels=self.hidden3, out_channels=self.hidden4, kernel_size=self.kernel_size),
            nn.BatchNorm1d(num_features=self.hidden4),
            self.activation,
            nn.MaxPool1d(kernel_size=2)
        )
        
        self.conv5 = nn.Sequential(
            nn.Conv1d(in_channels=self.hidden4, out_channels=self.hidden5, kernel_size=self.kernel_size),
            nn.BatchNorm1d(num_features=self.hidden5),
            self.activation,
            nn.MaxPool1d(kernel_size=2)
        )
        
        self.conv6 = nn.Sequential(
            nn.Conv1d(in_channels=self.hidden5, out_channels=self.hidden6, kernel_size=self.kernel_size),
            nn.BatchNorm1d(num_features=self.hidden6),
            self.activation,
            nn.MaxPool1d(kernel_size=2)
        )
        
        self.convLast = nn.Sequential(
            nn.Conv1d(in_channels=self.hidden4, out_channels=self.hiddenLast, kernel_size=self.kernel_size),
            nn.BatchNorm1d(num_features=self.hiddenLast),
            self.activation,
            nn.Dropout(0.2),
            nn.MaxPool1d(kernel_size=2)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.convLast(x)
        return x

# Define the CNN_LSTM class
class CNN_LSTM(nn.Module):
    def __init__(self):
        super(CNN_LSTM, self).__init__()
        self.cnn_output = 128
        self.seq_length_after_cnn = calculate_output_length([
            CNN(hidden=128).conv1[0],
            CNN(hidden=128).conv2[0],
            CNN(hidden=128).conv2[3],
            CNN(hidden=128).conv3[0],
            CNN(hidden=128).conv3[3],
            CNN(hidden=128).conv4[0],
            CNN(hidden=128).conv4[3],
            CNN(hidden=128).conv5[0],
            CNN(hidden=128).conv5[3],
            CNN(hidden=128).conv6[0],
            CNN(hidden=128).conv6[3],
            CNN(hidden=128).convLast[0],
            CNN(hidden=128).convLast[4]
        ], 187)
        
        self.hidden_lstm = 128
        self.cnn = CNN(hidden=self.cnn_output)
        self.lstm = nn.LSTM(input_size=self.cnn_output, hidden_size=self.hidden_lstm, batch_first=True, num_layers=3)
        self.fc = nn.Linear(in_features=self.hidden_lstm * self.seq_length_after_cnn, out_features=2)

    def forward(self, x):
        x = self.cnn(x)  
        x = x.permute(0, 2, 1)  
        x, _ = self.lstm(x)  
        x = x.reshape(-1, self.hidden_lstm * self.seq_length_after_cnn)
        x = self.fc(x)
        return x

def calculate_output_length(layers, initial_length):
    length = initial_length
    for layer in layers:
        if isinstance(layer, nn.Conv1d):
            kernel_size = layer.kernel_size[0]
            stride = layer.stride[0]
            padding = layer.padding[0]
            dilation = layer.dilation[0]
            length = (length + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1
        elif isinstance(layer, nn.MaxPool1d):
            kernel_size = layer.kernel_size
            stride = layer.stride
            length = (length - kernel_size) // stride + 1
    return length

def preprocess_signal(signal):
    signal = np.expand_dims(signal, axis=0) 
    signal = torch.tensor(signal, dtype=torch.float32)
    return signal

def predict(model, signal, encoder):
    signal = preprocess_signal(signal)
    output = model(signal)
    pred = output.argmax(dim=1).item()
    return encoder.inverse_transform([pred])[0]

def load_model(model_path):
    model = CNN_LSTM()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

# Label encoder
encoder = LabelEncoder()
encoder.classes_ = np.array(['Normal', 'Abnormal'])


class MyocardialInfarction(Page):
    def __init__(self, data, **kwargs):
        name = "About"
        super().__init__(name, data, **kwargs)

    def content(self):
        model_path = "models/cnn_lstm_model.pth"

        def load_model(model_path):
            model = CNN_LSTM()
            model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
            return model

        model = load_model(model_path)
        model.eval()  


        st.title("Myocardial Infarction Detection")
        st.write("Myocardial Infarction (MI), commonly known as a heart attack, occurs when blood flow to a part of the heart muscle is blocked. This blockage usually results from a buildup of cholesterol, fat, and other substances, forming a plaque in the coronary arteries.")

        select_col1, select_col2 = st.columns([0.5, 0.5])

        with select_col1:
            option2 = st.selectbox("Patient Record", ("100", "101", "102", "103", "104", "105", "106"))

        data = f"mit-bih-arrhythmia-database-1.0.0/{option2}"
        record = wfdb.rdrecord(data, smooth_frames=True)
        ecg_signal = record.p_signal[:8000, 0]

        device = 'cpu'
        input_data = torch.tensor(ecg_signal, dtype=torch.float32).unsqueeze(0).to(device)

        st.line_chart(input_data[0].cpu().numpy(), use_container_width=True)
        with torch.no_grad():
            output = model(input_data.unsqueeze(1))  

        predictions = torch.argmax(output, dim=1).cpu().numpy()[0]
        print(torch.argmax(output, dim=1).cpu().numpy())

        total1, total2, total3 = st.columns(3, gap='small')
        with total2: 
            if predictions == 0:

                st.metric(label="Result", value="MI is undetected.")

                style_metric_cards(background_color="white", border_left_color="#89CFF0", border_color="#89CFF0",
                                box_shadow="#F71938",border_size_px =6)
            elif predictions == 1:
                st.metric(label="Result", value="MI is detected.")

                style_metric_cards(background_color="white", border_left_color="red", border_color="red",
                                box_shadow="#F71938",border_size_px =3)
