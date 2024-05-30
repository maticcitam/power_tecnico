'''
PROJECT 2
Author: Matic Robnik
Date: March 2024
The objective is to make a dashboard of the first project. This is a python script creating the frontend part of application. For backend calculations and loading of data see file 'backend.py'. Custom stylesheet is saved in file 'assets/style.css'.
'''
from dash import dcc, html
import dash
import plotly.express as px
from dash.dependencies import Input, Output
from backend import *
from dash_bootstrap_templates import load_figure_template

load_figure_template(["darkly"])
model_labels = ['Decision Tree Regressor', 
                'Gradient Boosting Regressor', 
                'Linear Regression', 
                'Neural Network MLPRegressor', 
                'Random Forest Regressor']
error_labels = ['MAE', 
            'MBE', 
            'MSE', 
            'RMSE', 
            'cvRMSE', 
            'NMBE',
            'R2']

# backend stuff
data = load_data()
result_df = run_model(data)
errors_df = pd.DataFrame()
for model_label in model_labels:
    errors = calculate_errors(result_df['Power consumption [kWh]'], result_df[model_label])
    errors.rename(model_label, inplace=True)
    errors_df = errors_df.append(errors)

data.rename(mapper={
    'North Tower (kWh)': 'Power consumption [kWh]',
    'temp_C': 'Temperature [C]',
    'HR': 'Relative humidity [%]',
    'windSpeed_m/s': 'Wind speed [m/s]',
    'windGust_m/s': 'Wind gusts [m/s]',
    'pres_mbar': 'Pressure [mbar]',
    'solarRad_W/m2': 'Solar radiation [W/m2]',
    'rain_mm/h': 'Rain [mm/h]'
}, axis=1, inplace=True)


external_stylesheets = ['style.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets, suppress_callback_exceptions=True)

app.layout = html.Div([
    html.H1("Energy Modeling - Predicting Power Consumption at IST"),
    
    dcc.Tabs(id='tabs', value='tab-1', parent_className='custom-tabs', className='custom-tabs-container',children=[
        dcc.Tab(label='DATA OVERVIEW', value='tab-1', className='custom-tab', selected_className='custom-tab--selected'),
        dcc.Tab(label='MODELING RESULTS', value='tab-2', className='custom-tab', selected_className='custom-tab--selected'),
    ]),
    html.Div(id='tabs-content')
])

# Create the callback function for choosing tabs
@app.callback(Output('tabs-content', 'children'),
              Input('tabs', 'value'))
def render_content(tab):
    if tab == 'tab-1':
        return eda_tab
    elif tab == 'tab-2':
        return model_tab

# Create separate div elements for tabs
eda_tab = html.Div([
    html.H2('Data overview'),
    html.Div('Select the data you want to visualize in the plots below.'),
    html.Br(),
    html.Div([
        dcc.Dropdown(
            id='yaxis-column',
            options=[{'label': i, 'value': i} for i in data.columns[:-5]],
            value=['Power consumption [kWh]'],
            multi=True,
            className='custom-dropdown',
        ),
    ]),
    html.Br(),
    html.Div([
    dcc.Graph(id='weather-energy'),
    dcc.Graph(id='energy-consumption')
    ], style={'width': '70%'}),
    ])

model_tab = html.Div([
    html.H2('Modeling Power Consumption of Tecnico'),
    html.Div([
        "We utilized several models, including: ",
        html.Ul([
            html.Li("Decision Tree Regressor"),
            html.Li("Gradient Boosting Regressor"),
            html.Li("Linear Regression"),
            html.Li("Neural Network MLPRegressor"),
            html.Li("Random Forest Regressor")
        ]),
        " to predict power consumption of the North tower of Tecnico Lisboa. ",
        "These models were trained on consumption and weather data from 2017 and 2018. ",
        "Below are the features considered for this prediction:",
        html.Ul([
            html.Li("Solar radiation"),
            html.Li("Hour of the day"),
            html.Li("Power consumption in some previous moments in time")
        ])
    ]),
    html.Br(),
    html.Div([
        dcc.Dropdown(
            id='model-selection',
            options=[{'label': i, 'value': i} for i in model_labels],
            value=['Power consumption [kWh]'],
            multi=True,
            className='custom-dropdown',
        ),
    ]),
    html.Br(),
    html.Div([
        dcc.Graph(id='energy-consumption-plot'),
        html.Br(),
    ], style={'width': '60%', 'float':'left', 'display': 'inline-block'}),
    html.Div([
        html.H2("Error Metrics"),
        html.Table(id='error-table')
    ], style={'width': '30%', 'float':'right', 'display': 'inline-block'})
])


# Create the callback function for updating the scatter plot
@app.callback(
    Output('weather-energy', 'figure'),
    Input('yaxis-column', 'value'))
def update_scatter_plot(yaxis_column_name):
    fig = px.line(data, y=yaxis_column_name, template='darkly')
    return fig

# Create the callback function for updating the histogram
@app.callback(
    Output('energy-consumption', 'figure'),
    Input('yaxis-column', 'value'))
def update_histogram(yaxis_column_name):
    fig = px.histogram(data, x=yaxis_column_name, template='darkly')
    return fig

@app.callback(
    Output('energy-consumption-plot', 'figure'),
    Input('model-selection', 'value'))
def update_model_plot(selected_models):
    y = ['Power consumption [kWh]'] + selected_models
    fig = px.line(result_df, x=result_df.index, y=y, template='darkly')
    return fig

@app.callback(
    Output('error-table', 'children'),
    Input('model-selection', 'value'))
def update_table(selected_models):
    if 'Power consumption [kWh]' in selected_models:
        return []
    
    header = [html.Th('Error Metric')]
    for model in selected_models:
        header.append(html.Th(model))

    # Create table rows with error metrics and corresponding values
    rows = []
    for metric in error_labels:
        row = [html.Td(metric, style={'border': '1px solid grey'})]
        for model in selected_models:
            value = round(errors_df.at[model, metric], 3)
            row.append(html.Td(value, style={'border': '1px solid grey'}))
        rows.append(html.Tr(row))

    return [html.Thead(html.Tr(header)), html.Tbody(rows)]


if __name__ == '__main__':
    app.run_server(debug=True)