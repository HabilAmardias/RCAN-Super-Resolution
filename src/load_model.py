import torch
import streamlit as st
from model import RCAN

@st.cache_resource
def load_model(path):
    model = RCAN()
    model.load_state_dict(
        torch.load(
            path,
            map_location='cpu',
            weights_only=True
        )
    )
    return model