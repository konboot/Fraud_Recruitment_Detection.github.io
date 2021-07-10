import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn import metrics, datasets
from sklearn import tree, neighbors
from sklearn.ensemble import RandomForestClassifier 
import pandas as pd

df = pd.read_csv("fake_job_postings.csv")

X = df.labeled[:, None]
X_train, X_test, y_train, y_test = train_test_split(
    X, df.labeled, random_state=42)

models = {'Random Forest': RandomForestClassifier,
          'Decision Tree': tree.DecisionTreeRegressor,
          'k-NN': neighbors.KNeighborsRegressor}
app = dash.Dash(__name__)

app.layout = html.Div([
    html.P("Train Model:"),
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

    y_score = model.predict_proba(X_test)[:, 1]

    fpr, tpr, thresholds = metrics.roc_curve(y_test, y_score)
    score = metrics.auc(fpr, tpr)

    fig = px.area(
        x=fpr, y=tpr,
        title=f'ROC Curve (AUC={score:.4f})',
        labels=dict(
            x='False Positive Rate', 
            y='True Positive Rate'))
    fig.add_shape(
        type='line', line=dict(dash='dash'),
        x0=0, x1=1, y0=0, y1=1)

    return fig

app.run_server(debug=True)