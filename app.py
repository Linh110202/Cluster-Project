
# Run this app with `python app.py` and
# visit http://127.0.0.1:8050/ in your web browser.

import dash
from dash import dcc
from dash import html, Output, Input, State
import dash_bootstrap_components as dbc
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN, KMeans, Birch, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

import plotly.express as px
import pandas as pd
import base64
import io
from typing import List, Union, Tuple
import numpy as np
from templates import get_main_html, get_alert_html, blank_fig


# assume you have a "long-form" data frame
# see https://plotly.com/python/px-arguments/ for more options

# fig = px.bar(df, x="Fruit", y="Amount", color="City", barmode="group")
# dcc.Graph(
#         id='MyGraph',
#         figure=None,
#         style={
            
#         }
#     )


app = dash.Dash(__name__)
app.layout = html.Div([
    get_alert_html(),
    get_main_html()
])

#Tóm lại, hàm get_RFM này giúp tự động tính toán và chuẩn hóa dữ liệu RFM cho từng khách hàng từ một bảng dữ liệu đầu vào.
def get_RFM(input_df: pd.DataFrame):
    Monetary = input_df.groupby('Customer_id')['Price'].sum()
    Monetary = Monetary.reset_index()
    
    Frequency = input_df.groupby('Customer_id')['Invoice'].count()
    Frequency = Frequency.reset_index()
    
    input_df["Purchase_day"] = pd.to_datetime(input_df['Purchase_day'],format='%Y-%m-%d %H:%M:%S')
    latest_time = max(input_df["Purchase_day"])
    input_df["day_diff"] = latest_time - input_df["Purchase_day"]
    input_df["day_diff"] = input_df["day_diff"].dt.days

    Recency = input_df.groupby("Customer_id")['day_diff'].min()
    Recency = Recency.reset_index()
    
    RFM = Monetary.merge(Frequency, left_on = "Customer_id", right_on = "Customer_id")
    RFM = RFM.merge(Recency, left_on = "Customer_id", right_on = "Customer_id")

    RFM.rename({"Invoice": "Frequency", "day_diff":"Recency"}, axis = 1, inplace = True)
    
    scaler = StandardScaler()
    # fit_transform
    rfm_df_scaled = scaler.fit_transform(RFM)
    
    return rfm_df_scaled
  
def get_clustering(input_data, 
                   color_sets= {0:"green", 1: "blue", 2: "magenta",3:"goldenrod", 4: "portland", 5:"twilight", -1: "red"}):
    cluster_score = {}
    for k_trial in range(3,6):
        model_instance = KMeans(n_clusters = k_trial, n_init = "auto")
        
        score = silhouette_score(input_data, model_instance.fit_predict(input_data))
        cluster_score[k_trial] = score
    
    # find optimal num clusters
    sort_score = {k: v for k, v in sorted(cluster_score.items(), key=lambda item: item[1])}
    
    optimal_cluster = list(sort_score.keys())[-1]
    optimal_score = round(sort_score[optimal_cluster], 4)
    print("optimal: ",optimal_cluster)
    # finalize
    model_instance = KMeans(n_clusters = optimal_cluster, n_init = "auto")
    Z = model_instance.fit_predict(input_data)
    centroids = model_instance.cluster_centers_

    rfm_df_scaled = np.concatenate([input_data,centroids], axis = 0)
    Z = np.concatenate([Z, np.full(shape = (optimal_cluster), fill_value = -1)], axis = 0)
    print("max Z: ", Z.max())

    cluster_df = pd.DataFrame({"M": rfm_df_scaled[:,0],"F": rfm_df_scaled[:,1], "R":rfm_df_scaled[:,2], "lable": Z})
    cluster_df["cluster_color"] = cluster_df["lable"].apply(lambda x: color_sets[x])
    cluster_df["marker_size"] = cluster_df["lable"].apply(lambda x: 2 if x >= 0 else 10)
    cluster_df["symbol"] = cluster_df["lable"].apply(lambda x: "circle" if x >= 0 else "triangle")

    fig = px.scatter_3d(cluster_df, 
                        x = cluster_df["M"],
                        y = cluster_df["F"], 
                        z = cluster_df["R"],
                        size = cluster_df["marker_size"], 
                        symbol = cluster_df["symbol"], 
                        color = cluster_df["cluster_color"],
                        width=1000, height=800,
                        title = f"Optimal number of clusters: {optimal_cluster} with silhouette score: {optimal_score}"
                    )    
    return fig


def check_df(input_df: pd.DataFrame, 
             target_cols=["Invoice", "Purchase_day", "Customer_id", "Price"]
             )->Tuple[bool, Union[pd.DataFrame, str]]:
    if len([col for col in list(input_df.columns) if col in target_cols]) == len(target_cols):
        
        try:
            pd.to_datetime(input_df['Purchase_day'],format='%Y-%m-%d %H:%M:%S')
            return True, input_df[target_cols]
        except:
            return False, "Column 'Purchase_day' not in correct format, expect: '%Y-%m-%d %H:%M:%S' "
    else:
        return False, f"the table's columns do not have these columns {target_cols}"



@app.callback([Output('alert-fade', 'is_open'),
               Output('alert-fade', 'children'),
               Output('main-html', 'style'),
               Output('alert-html', 'style'),
               Output('visualization', 'figure'),
               Output('visualization', 'style')],
              [Input('upload-csv', 'contents'),
               Input('algo-dropdown', 'value'),
               Input('submit-button', 'n_clicks')],
              [State('upload-csv', 'filename')],
              prevent_initial_call=True)
def update_output(contents, values: List[str], n_clicks:int, file_name:str):
    if contents is not None and values is not None and n_clicks>0:
        print("values:", values)
        print("file name:", file_name)
        
        # string to pd.Dataframe
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        main_df = pd.read_csv(io.StringIO(decoded.decode("utf-8")))

        status, result = check_df(input_df= main_df)
        
        if status == True:
            RFM_df = get_RFM(input_df= result)
            
            figure = get_clustering(input_data= RFM_df)
            
            # visualization            
            return False, " ", dict(), dict(display='none'), figure, dict()
        else:
            return True, result, dict(display='none'), dict(), blank_fig(), dict(display='none')

    else:
        return False, "", dict(), dict(display='none'), blank_fig(), dict(display='none')

if __name__ == '__main__':
    app.run(debug=True)
