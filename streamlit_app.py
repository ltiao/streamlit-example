import torch

import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
import plotly.graph_objects as go

from botorch.test_functions import (
    Branin, 
    Bukin,
    Beale,
    DropWave,
    DixonPrice,
    HolderTable,
    Levy,
    Griewank,
    Michalewicz,
    Rastrigin,
    Rosenbrock,
    StyblinskiTang,
    SixHumpCamel,
    ThreeHumpCamel, 
    EggHolder
)

from src.config import default_device, default_float
from src.constants import STEPS

FUNC_CLASSES = dict(
    branin=Branin, 
    bukin=Bukin,
    beale=Beale,
    dropwave=DropWave,
    dixon_price=DixonPrice,
    six_hump_camel=SixHumpCamel,
    three_hump_camel=ThreeHumpCamel, 
    eggholder=EggHolder,
    griewank=Griewank,
    levy=Levy,
    holder_table=HolderTable,
    michalewicz=Michalewicz,
    rastrigin=Rastrigin,
    rosenbrock=Rosenbrock,
    styblinski_tang=StyblinskiTang
)

FUNC_NAMES = dict(
    branin="Branin", 
    bukin="Bukin",
    beale="Beale",
    dropwave="Drop-Wave",
    dixon_price="Dixon-Price",
    six_hump_camel="Six-Hump Camel",
    three_hump_camel="Three-Hump Camel", 
    eggholder="Eggholder",
    griewank="Griewank",
    levy="Levy",
    holder_table="Holder Table",
    michalewicz="Michalewicz",
    rastrigin="Rastrigin",
    rosenbrock="Rosenbrock",
    styblinski_tang="Styblinski-Tang"
)


def surface_plot(func):

    xs = [torch.linspace(*bound, steps=STEPS) for bound in func._bounds]
    x_grids = torch.meshgrid(*xs, indexing='xy')
    X_grid = torch.dstack(x_grids)
    Z_grid = func(X_grid)

    return go.Figure(data=[go.Surface(x=x_grids[0], y=x_grids[1], z=Z_grid)])


def optimizers_dataframe(func):

    return pd.DataFrame(
        func._optimizers, 
        columns=[f'x{d+1}' for d in range(func.dim)]
    )


"""
# Test Functions for Optimization

Edit `/streamlit_app.py` to customize this app to your heart's desire :heart:.
If you have any questions, check out our [documentation](https://docs.streamlit.io) and [community
forums](https://discuss.streamlit.io).

In the meantime, below is an example of what you can do with just a few lines of code:
"""


for func_key, func_cls in FUNC_CLASSES.items():

    func = func_cls()
    st.header(FUNC_NAMES[func_key])

    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Dimensions", func.dim, help=None)

    with col2:
        st.metric("Global Optimum", func.optimal_value, help=None)

    with st.expander("Optimizer(s)", expanded=True):
        st.dataframe(optimizers_dataframe(func), use_container_width=True)

    st.plotly_chart(surface_plot(func), use_container_width=True)
    st.divider()
