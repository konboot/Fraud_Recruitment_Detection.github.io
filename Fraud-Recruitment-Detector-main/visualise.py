import plotly.graph_objects as go 
import dash
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
import dash_core_components as dcc
import dash_html_components as html
import dash_split_pane
import plotly.express as px
import pandas as pd
from datetime import datetime
from dash.dependencies import Input, Output
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn import tree, neighbors
from sklearn.ensemble import RandomForestClassifier 


df = pd.read_csv("fake_job_postings.csv")

X = df.labeled[:, None]
X_train, X_test, y_train, y_test = train_test_split(
    X, df.labeled, random_state=42)

models = {'Random Forest': RandomForestClassifier,
          'Decision Tree': tree.DecisionTreeRegressor,
          'k-NN': neighbors.KNeighborsRegressor}
app = dash.Dash(__name__)

app.layout = html.Div(children = [html.Div("Welcome to Our Dashboard", style = {
                                                                        "color": "red",
                                                                        "text-align": "center","background-color": "dark-orange",
                                                                        "border-style": "dotted" , "display":"inline-block", "width":"80%"
                                                                      }), 
                      html.Div("Fraud Recruitment Detector", style = {
                                                                        "color": "red",
                                                                        "text-align": "center","background-color": "dark-orange",
                                                                        "border-style": "dotted" , "display":"inline-block", "width":"40%"
                                                                      }),
                      html.Div("Fraud Recruitment Detector", style = {
                                                                        "color": "red",
                                                                        "text-align": "center","background-color": "yellow",
                                                                        "border-style": "dotted" , "display":"inline-block", "width":"40%"
                                                                     })])





app.layout = html.Div([
    html.P("Select Model:"),
    dcc.Dropdown(
        id='model-name',
        options=[{'label': x, 'value': x} 
                 for x in models],
        value='Random Forest',
        clearable=False
    ),
    dcc.Graph(id="graph"),
])

@app.callback(
    Output("graph", "figure"), 
    [Input('model-name', "value")])
def train_and_display(name):
    model = models[name]()
    model.fit(X_train, y_train)

    x_range = np.linspace(X.min(), X.max(), 100)
    y_range = model.predict(x_range.reshape(-1, 1))

    fig = go.Figure([
        go.Scatter(x=X_train.squeeze(), y=y_train, 
                   name='train', mode='markers'),
        go.Scatter(x=X_test.squeeze(), y=y_test, 
                   name='test', mode='markers'),
        go.Scatter(x=x_range, y=y_range, 
                   name='prediction')
    ])

    return fig



if __name__ =='__main__':
    app.run_server()
