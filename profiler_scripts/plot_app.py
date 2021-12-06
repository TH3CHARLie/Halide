import pickle
import argparse
import plotly.graph_objects as go
from plotly.offline import plot
import plotly.express as px
import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output
import pandas as pd
import math
from scipy import stats
import numpy as np
import json
from json import JSONEncoder


styles = {
    'pre': {
        'border': 'thin lightgrey solid',
        'overflowX': 'scroll'
    }
}


class SampleEncoder(JSONEncoder):
        def default(self, o):
            return o.__dict__


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, action="store", help="path to pickled file")
    args = parser.parse_args()
    return args


def deseralize(input_file):
    with open(input_file, "rb") as f:
        res = pickle.load(f)
    return res


def rsquared(predicted_runtimes, actual_runtimes):
    slope, intercept, r_value, p_value, std_err = stats.linregress(predicted_runtimes, actual_runtimes)
    return r_value ** 2


def relative_loss(predicted_runtimes, actual_runtimes):
    actual = np.array(actual_runtimes)
    predicted = np.array(predicted_runtimes)
    reference = np.min(actual)
    scale = 1.0 / reference

    actual = actual * scale
    predicted = predicted * scale
    predicted[predicted < 1e-10] = 1e-10
    return np.sum((1.0 / predicted - 1.0 / actual) ** 2)


def run(samples):
    sample_names = [x.sample_name for x in samples]
    sample_jsonfied = [json.dumps(x, cls=SampleEncoder, indent=2) for x in samples]
    actual_runtimes = [x.actual_runtime for x in samples]
    predicted_runtimes = [x.predicted_runtime for x in samples]
    df = pd.DataFrame(dict(
        predicted_runtimes = predicted_runtimes,
        actual_runtimes = actual_runtimes
    ))
    max_val = math.ceil(max(max(actual_runtimes), max(predicted_runtimes)))
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=predicted_runtimes,
        y=actual_runtimes,
        customdata=sample_jsonfied,
        mode='markers',
        hovertemplate =
        'predicted_runtime: %{x}'+
        '<br>actual_runtime: %{y}'))
    fig.add_trace(go.Scatter(x=[0, max_val], y=[0, max_val], mode='lines'))
    fig.update_layout(
        autosize=False,
        width=1280,
        height=960,
    )
    fig.update_xaxes(title_text="predicted_runtimes", type="log")
    fig.update_yaxes(title_text="actual_runtimes", type="log")
    plot_div = plot(fig, output_type='div', include_plotlyjs=True)
    r2 = rsquared(predicted_runtimes, actual_runtimes)
    title = "{}: Run Time Predictions (R^2 = {:.2f}; Loss = {:.2f})".format("bilateral grid", r2, relative_loss(predicted_runtimes, actual_runtimes))
    fig.update_layout(
    title={
        'text': title})


    app = dash.Dash()
    app.layout = html.Div([
        dcc.Graph(
            id='basic-interactions',
            figure=fig
        ),
        html.Div(className='row', children=[
            html.Div([
                    dcc.Markdown("""
                        **Hover Data**

                        Mouse over values in the graph.
                    """),
                    html.Pre(id='hover-data', style=styles['pre'])
            ], className='three columns')
        ])
    ])

    @app.callback(
        Output('hover-data', 'children'),
        Input('basic-interactions', 'hoverData'))
    def display_hover_data(hoverData):
        if hoverData is not None and "points" in hoverData:
            return hoverData["points"][0]["customdata"]
        return ""

    app.run_server(host=('0.0.0.0'), debug=True, use_reloader=False)


if __name__ == "__main__":
    args = parse_args()
    samples = deseralize(args.input)
    run(samples)
