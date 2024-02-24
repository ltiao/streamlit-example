import torch

import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
import plotly.graph_objects as go

from botorch.test_functions import Branin

"""
# Welcome to my Streamlit App!

Edit `/streamlit_app.py` to customize this app to your heart's desire :heart:.
If you have any questions, check out our [documentation](https://docs.streamlit.io) and [community
forums](https://discuss.streamlit.io).

In the meantime, below is an example of what you can do with just a few lines of code:
"""

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.double

STEPS = 128

func = Branin()
xs = [torch.linspace(*bound, steps=STEPS) for bound in func._bounds]
x_grids = torch.meshgrid(*xs, indexing='xy')
X_grid = torch.dstack(x_grids)
Z_grid = func(X_grid)

fig = go.Figure(data=[go.Surface(x=x_grids[0], y=x_grids[1], z=Z_grid)])

st.plotly_chart(fig, use_container_width=True)
