import pickle
import argparse
import plotly.graph_objects as go
import plotly.express as px
import dash
from dash import dcc
from dash import html
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, action="store", help="path to pickled file")
    args = parser.parse_args()
    return args


def deseralize(input_file):
    with open(input_file, "rb") as f:
        res = pickle.load(f)
    return res

def run(samples):
    sample_names = [x.sample_name for x in samples]
    actual_runtimes = [x.actual_runtime for x in samples]
    predicted_runtimes = [x.predicted_runtime for x in samples]
    df = pd.DataFrame(dict(
        predicted_runtimes = predicted_runtimes,
        actual_runtimes = actual_runtimes
    ))
    fig = px.scatter(df, x="predicted_runtimes", y="actual_runtimes")
    app = dash.Dash()
    app.layout = html.Div([
        dcc.Graph(figure=fig)
    ])

    app.run_server(host=('0.0.0.0'), debug=True, use_reloader=False)


if __name__ == "__main__":
    args = parse_args()
    samples = deseralize(args.input)
    run(samples)