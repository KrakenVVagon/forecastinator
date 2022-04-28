import dash
from dash import html
from dash import dcc
import plotly.graph_objects as go
import plotly.express as px
import dash.dependencies as dd

app = dash.Dash()

df = px.data.stocks()

h1_style = {
        'textAlign':'center',
        'marginTop':40,
        'marginBottom':40
        }

dropdown_options = [
        {'label':'Google','value':'GOOG'},
        {'label':'Apple','value':'AAPL'},
        {'label':'Amazon','value':'AMZN'}
        ]

heading = html.H1(id='H1',children='Testing Dash Application',style=h1_style)
dropdown= dcc.Dropdown(id='dropdown',options=dropdown_options,value='GOOG')
graph = dcc.Graph(id='plot')

app.layout = html.Div(id='parent',children=[heading,dropdown,graph])

@app.callback(dd.Output(component_id='plot',component_property='figure'),[dd.Input(component_id='dropdown',component_property='value')])
def graph_update(dropdown_value):
    print(dropdown_value)
    scatter = go.Scatter(x=df['date'],y=df[dropdown_value],line=dict(color='firebrick',width=4))
    fig = go.Figure([scatter])

    fig.update_layout(
            title='Stock Prices Over Time',
            xaxis_title='Date',
            yaxis_title='Price'
            )

    return fig

if __name__ == '__main__':
    app.run_server(debug=True,port=8080,host='0.0.0.0')
