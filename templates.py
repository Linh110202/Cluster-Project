import dash
from dash import dcc
from dash import html, Output, Input, State
import dash_bootstrap_components as dbc

import plotly.express as px
import pandas as pd
import base64
import io
from typing import List, Union, Tuple
import plotly.graph_objects as go

def blank_fig():
    fig = go.Figure(go.Scatter(x=[], y = []))
    fig.update_layout(template = None)
    fig.update_xaxes(showgrid = False, showticklabels = False, zeroline=False)
    fig.update_yaxes(showgrid = False, showticklabels = False, zeroline=False)
    
    return fig


def get_main_html():
    return html.Div([
            dbc.Row([
                    dbc.Col([
                            dcc.Upload(id= "upload-csv",
                                       children= ['Drag/Drop or ',
                                                  html.A('Select a File')], 
                                       multiple = False,
                                       style={
                                            'left': '200px',
                                            'top': '50px',
                                            'position': 'relative',
                                            'width': '200px',
                                            'height': '100px',
                                            'lineHeight': '60px',
                                            'borderWidth': '1px',
                                            'borderStyle': 'dashed',
                                            'borderRadius': '5px',
                                            'textAlign': 'center'
                                            }),
                            dcc.Checklist(options = ['K-Means'],
                                        id='algo-dropdown',
                                        style= {
                                            'left': '200px',
                                            'top': '90px',
                                            'position': 'relative',
                                            'width': '200px',
                                            'height': '60px',                 
                                        }),
                            html.Button('Submit', 
                                        id='submit-button', 
                                        n_clicks=0,
                                        style={
                                            'left': '200px',
                                            'top': '110px',
                                            'position': 'relative',
                                            'width': '200px',
                                            'height': '60px',
                                        })
                        ], style= {
                                'left': '0px',
                                'top': '50px',
                                'width': '400px',
                                'height': '900px',
                                'position': 'absolute',
                            }),
                    dbc.Col(
                        dcc.Graph(id='visualization', 
                                  figure= blank_fig(), 
                                  style= {'display': 'none',
                                        'left': '0px',
                                        'top': '0px',
                                        'position': 'relative',
                                        'width': '100%',
                                        'height': '100%',
                                        'lineHeight': '60px',
                                        'borderWidth': '1px',
                                        'borderStyle': 'dashed',
                                        'borderRadius': '5px',
                                        'textAlign': 'center'
                                    }
                        ), style= {
                                'left': '800px',
                                'top': '50px',
                                'width': '1200px',
                                'height': '900px',
                                'position': 'absolute',
                        })
                    ])
            
        ], id= "main-html")


def get_alert_html():
    return html.Div([
            dbc.Alert(
                    id="alert-fade",
                    dismissable=True,
                    color="warning",
                    style= {
                        'top': '0px',
                        'lineHeight': '60px',
                        'borderWidth': '1px',
                        'borderStyle': 'dashed',
                        'borderRadius': '5px',
                        'width': '100%',
                        'height': '80px',
                        'textAlign': 'center',
                    }),
            html.A(
            html.Button('Refresh Page', 
                        id='refresh-button', 
                        n_clicks=0,
                        style={
                            'top': '100px',
                            'position': 'relative',
                            'width': '100%',
                            'height': '60px',
                            'textAlign': 'center',
                            'margin': 'auto',
                        })
            ,href='/')
    ], id = "alert-html", style= {'display': 'none'})