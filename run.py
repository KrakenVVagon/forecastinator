import dash
from dash import html
from dash import dcc
import plotly.graph_objects as go
import plotly.express as px
import dash.dependencies as dd
import base64
import pandas as pd
import io
import forecastinator.models as m

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
        display_format='DD/MM/YYYY',
        start_date_placeholder_text='dd/mm/yyyy',
        end_date_placeholder_text='dd/mm/yyyy'
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
'''
@app.callback([dd.Output('dropdown','options'),
         dd.Output('dropdown','value')],
        [dd.Input('upload_data','contents'),
         dd.Input('upload_data','filename')]
        )
def update_dropdown(contents,filename):
    if contents:
        contents = contents[0]
        filename = filename[0]
        df = parse_upload(contents,filename)
        return [{'label':i,'value':i} for i in df.columns], df.columns[0]
    return [],''

@app.callback(dd.Output(component_id='plot2',component_property='figure'),
        [dd.Input('upload_data','contents'),
         dd.Input('upload_data','filename'),
         dd.Input('dropdown','value'),
         dd.Input('reverse_button','n_clicks')]
        )
def upload_graph_update(contents,filename,dropdown_value,n):
    ctx = dash.callback_context
    if contents:
        contents = contents[0]
        filename = filename[0]
        df = parse_upload(contents,filename)

        scatter = go.Scatter(x=df['day'],y=df[dropdown_value],mode='lines+markers')
        data = [scatter]

        if ctx.triggered[0]['prop_id'].split('.')[0] == 'reverse_button':
            flipped = go.Scatter(x=df['day'],y=m.reverse(df[dropdown_value]),mode='lines+markers')
            data.append(flipped)

        fig = go.Figure(data)

        fig.update_layout(
            title='Some Random Thing',
            xaxis_title='Day',
            yaxis_title=dropdown_value
            )
        return fig
    fig = go.Figure(go.Scatter(x=[],y=[]))
    return fig
'''
if __name__ == '__main__':
    app.run_server(debug=True,port=8080,host='0.0.0.0')
