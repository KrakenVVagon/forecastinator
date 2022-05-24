import dash
from dash import html
from dash import dcc
import plotly.graph_objects as go
import plotly.express as px
import dash.dependencies as dd
import base64
import pandas as pd
import numpy as np
import io
import forecastinator.models as fm

def make_traces(df,x=None):
    '''
    Plots all the columns of a dataframe as individual traces.
    If x is provided use that as the xaxis value, otherwise default to first
    '''
    x = x or df.columns[0]
    data = [go.Scatter(x=df[x],y=df[c],name=c) for c in df.columns if c != x]
    return data

def predict_total_players(df,x,y):
    '''
    Returns a plot that has the total player prediction as a newly plotted trace.
    No start or end is needed for this prediction since it assumes a number of days from the existing data
    df : pandas df of data
    x : xaxis column
    y : column that should be predicted
    '''
    predict_df = df.copy()
    return make_traces(predict_df,x=x)

def predict_dau(df,x,y,start_date,end_date):
    '''
    Returns a plot with the DAU prediction as a trace
    df : pandas df of data
    x : xaxis column
    y : column to be predicted
    start_date : where the prediction should start
    end_date : where the prediction should end
    '''
    predict_df = df.copy()
    predict_df.index = predict_df[x]
    predict_df.sort_index(inplace=True)

    predict_df.index.map(str)
    predict_df.index = pd.to_datetime(predict_df.index)
    predict_df.index = predict_df.index.strftime('%Y-%m-%d')

    prediction_base = fm.TimeSeries(predict_df[y].loc[predict_df.index[0]:start_date])
    prediction_time = fm.TimeSeries(predict_df[y].loc[start_date:end_date])

    arima_model = prediction_base.create_arima_model(transform='meanshift',seasonal=True)
    prediction = arima_model.predict(start_date,end_date,transform='lineartrend')

    predict_df['predicted_dau'] = [np.nan]*len(predict_df)
    predict_df.loc[start_date:end_date,'predicted_dau'] = prediction.values

    return make_traces(predict_df,x=x)

app = dash.Dash()

title_style = {
        'textAlign':'center',
        'marginTop':40,
        'marginBottom':40
        }

upload_style = {
        'width':'100%',
        'height':'60px',
        'lineHeight':'60px',
        'borderWidth':'1px',
        'borderStyle':'dashed',
        'borderRadius':'5px',
        'textAlign':'center',
        'margin':'10px'
        }

lower_style = {
        'width':'49%',
        'display': 'inline-block'
        }

# we need to break the page down into 3 different sections
# an upper section with title, file selector and graph
# a lower section with options for DAU forecasting
# an adjacent lower section with options for total player forecasting

page_title = html.H1(id='page_title',children='Forecastinator!',style=title_style)
upload= dcc.Upload(
        id='upload_data',
        children=html.Div(['Drag and Drop or ',html.A('Select Files')]),
        style=upload_style,multiple=True
        )
graph = dcc.Graph(id='graph')
upper_section = html.Div(id='upper_section',children=[page_title,upload,graph])

dau_button = html.Button('Predict DAU',id='dau_button')
dau_datepicker = dcc.DatePickerRange(id='dau_datepicker',
        display_format='YYYY-MM-DD',
        start_date_placeholder_text='yyyy-mm-dd',
        end_date_placeholder_text='yyyy-mm-dd'
        )
dau_xdropdown = dcc.Dropdown(id='dau_xdropdown')
dau_ydropdown = dcc.Dropdown(id='dau_ydropdown')
dau_title_text = html.H2(id='dau_title',children='DAU Forecasting',style=title_style)
dau_date_text = dcc.Markdown('Select start/end dates')
dau_x_text = dcc.Markdown('Select x axis variable')
dau_y_text = dcc.Markdown('Select y axis variable')
lower_section1 = html.Div(id='lower_section1',children=[
    dau_title_text,
    dau_date_text,
    dau_datepicker,
    dau_x_text,
    dau_xdropdown,
    dau_y_text,
    dau_ydropdown,
    dau_button],
    style=lower_style)

total_button = html.Button('Predict Total Players',id='total_button')
total_xdropdown = dcc.Dropdown(id='total_xdropdown')
total_ydropdown = dcc.Dropdown(id='total_ydropdown')
total_title = html.H2(id='total_title',children='Total Player Forecasting',style=title_style)
total_x_text = dcc.Markdown('Select x axis variable')
total_y_text = dcc.Markdown('Select y axis variable')
lower_section2 = html.Div(id='lower_section2',children=[
    total_title,
    total_x_text,
    total_xdropdown,
    total_y_text,
    total_ydropdown,
    total_button],
    style=lower_style)

app.layout = html.Div(id='parent',children=[upper_section,lower_section1,lower_section2])

def parse_upload(contents,filename):
    content_type, content_string = contents.split(',')
    decoded_string = base64.b64decode(content_string)

    file_end = filename.split('.')[-1]
    if file_end == 'csv':
        df = pd.read_csv(io.StringIO(decoded_string.decode('utf-8')))
        return df
    elif 'xls' in file_end:
        df = pd.read_excel(io.BytesIO(decoded_string))
        return df

    return html.Div(['There was an error processing this file.'])

# update x and y axis dropdowns from file upload
@app.callback(
    [dd.Output('dau_xdropdown','options'),
    dd.Output('dau_xdropdown','value'),
    dd.Output('dau_ydropdown','options'),
    dd.Output('dau_ydropdown','value'),
    dd.Output('total_xdropdown','options'),
    dd.Output('total_xdropdown','value'),
    dd.Output('total_ydropdown','options'),
    dd.Output('total_ydropdown','value')],
    [dd.Input('upload_data','contents'),
    dd.Input('upload_data','filename')]
    )
def update_dropdowns(contents,filename):
    if contents:
        contents = contents[0]
        filename = filename[0]
        df = parse_upload(contents,filename)
        return df.columns,df.columns[0],df.columns,df.columns[1],df.columns,df.columns[0],df.columns,df.columns[1]
    return [],'',[],'',[],'',[],''

# update graph based on input click & upload
@app.callback(
        dd.Output('graph','figure'),
        [dd.Input('upload_data','contents'),
        dd.Input('upload_data','filename'),
        dd.Input('dau_xdropdown','value'),
        dd.Input('dau_ydropdown','value'),
        dd.Input('dau_datepicker','start_date'),
        dd.Input('dau_datepicker','end_date'),
        dd.Input('total_xdropdown','value'),
        dd.Input('total_ydropdown','value'),
        dd.Input('dau_button','n_clicks'),
        dd.Input('total_button','n_clicks')]
        )
def update_graph(contents,filename,dau_x,dau_y,dau_start,dau_end,total_x,total_y,dau_n,total_n):
    ctx = dash.callback_context
    if contents:
        contents = contents[0]
        filename = filename[0]
        df = parse_upload(contents,filename)
        trigger = ctx.triggered[0]['prop_id'].split('.')[0]
        
        if trigger == 'total_button':
            data = predict_total_players(df,total_x,total_y)
        elif trigger == 'dau_button':
            data = predict_dau(df,dau_x,dau_y,dau_start,dau_end)
        elif trigger in ['dau_xdropdown','dau_ydropdown']:
            data = make_traces(df,dau_x)
        elif trigger in ['total_xdropdown','total_ydropdown']:
            data = make_traces(df,total_x)
        else:
            data = make_traces(df)

        fig = go.Figure(data)
        fig.update_layout(
                title='',
                xaxis_title='',
                yaxis_title='',
                )

        return fig
    fig = go.Figure([go.Scatter(x=[],y=[])])
    return fig

if __name__ == '__main__':
    app.run_server(debug=True,port=8080,host='0.0.0.0')
