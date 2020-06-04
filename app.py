import plotly.graph_objects as go
import pandas as pd
from sklearn import preprocessing
import dash
import dash_core_components as dcc
import dash_html_components as html

df = pd.read_pickle(r'./data/normalized.p')


######## LAYOUT ########

app = dash.Dash()
app.layout = html.Div([
    dcc.Input(id='player_name_input',placeholder="Enter player's name", type='text', debounce=True),
    dcc.Graph(id='polar_bar'),
])


#polar plot
@app.callback(
    dash.dependencies.Output('polar_bar', 'figure'),
    [dash.dependencies.Input('player_name_input', 'value')]
)

def make_polar_chart(input):

    mf_attributes = [
        ('Unnamed: 1_level_0', 'Player'),
        ('Standard',            'Gls'),
        ('Expected_y',          'npxG'),
        ('Expected_y',          'npxG/Sh'),
        ('Unnamed: 24_level_0', 'xA'),
        ('Unnamed: 27_level_0', '1/3'),
        ('Total',               'PrgDist'),
        ('Total',               'Cmp%'),
        ('Dribbles',            'Succ%'),
        ('Unnamed: 28_level_0', 'Dispos'),
        ('Vs Dribbles',         'Past')
       ]

    player_df = df[mf_attributes].loc[df[( 'Unnamed: 1_level_0',  'Player')] == input]
    print(player_df)
    player_attr_values = player_df.values[0].tolist()
    fig = go.Figure()
    fig.add_trace(go.Barpolar(
        r=player_attr_values,  # wartosci atrybutow w kolejnosci przeciwnej do zegara
        name='Shots',
        marker_color='rgb(106,81,163)',
        theta=['Gls','npxG''npxG/Sh', 'xA','1/3','PrgDis','Cmp%', 'Succ%', 'Dispos', 'Past']
    ))

    fig.update_traces(
        text=['Gls','npxG''npxG/Sh', 'xA','1/3','PrgDis','Cmp%', 'Succ%', 'Dispos', 'Past'])

    fig.update_layout(
        title=input,
        font_size=16,
        legend_font_size=16,
        polar_angularaxis_rotation=90,
    )

    return fig


if __name__ == '__main__':
    app.run_server(debug=True)