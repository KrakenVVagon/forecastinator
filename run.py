import dash
from dash import html
from dash import dcc
import plotly.graph_objects as go
import plotly.express as px
import dash.dependencies as dd
import base64
import pandas as pd
import io

app = dash.Dash()

df = px.data.stocks()

h1_style = {
        'textAlign':'center',
        'marginTop':40,
        'marginBottom':40
        }

heading = html.H1(id='H1',children='Testing Dash Application',style=h1_style)
dropdown = dcc.Dropdown(id='dropdown')
upload= dcc.Upload(id='upload_data',children=html.Div(['Drag and Drop or ',html.A('Select Files')]),multiple=True)
upload_graph = dcc.Graph(id='plot2')

app.layout = html.Div(id='parent',children=[heading,upload,dropdown,upload_graph])

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
         dd.Input('dropdown','value')]
        )
def upload_graph_update(contents,filename,dropdown_value):
    if contents:
        contents = contents[0]
        filename = filename[0]
        df = parse_upload(contents,filename)

        scatter = go.Scatter(x=df['day'],y=df[dropdown_value],mode='lines+markers')
        fig = go.Figure(scatter)

        fig.update_layout(
            title='Some Random Thing',
            xaxis_title='Day',
            yaxis_title=dropdown_value
            )
        return fig
    fig = go.Figure(go.Scatter(x=[],y=[]))
    return fig

if __name__ == '__main__':
    app.run_server(debug=True,port=8080,host='0.0.0.0')
