import streamlit as st


class Page:
    def __init__(self, name, data, **kwargs):
        self.name = name
        self.data = data
        self.signal= data["record"].p_signal[:, 0]

        self.kwargs = kwargs

    def content(self):
        """Returns the content of the page"""

        raise NotImplementedError("Please implement this method.")

    def title(self):
        """Returns the title of the page"""
        st.header(f"{self.name}")

    def __call__(self):
        self.title()
        self.content()