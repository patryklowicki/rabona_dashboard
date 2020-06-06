import plotly.graph_objects as go
import pandas as pd
from sklearn import preprocessing
import dash
import dash_core_components as dcc
import dash_html_components as html
from sklearn import preprocessing
import dash_bootstrap_components as dbc

external_stylesheets = ['https://stackpath.bootstrapcdn.com/bootstrap/4.3.1/css/bootstrap.min.css" integrity="sha384-ggOyR0iXCbMQv3Xipma34MD+dH/1fQ784/j6cY/iJTQUOhcWr7x9JvoRxT2MZw1T']

df = pd.read_pickle(r'./data/merged_fieldplayers.p')


#### attributes

info_columns = [('Unnamed: 2_level_0', 'Nation'),
                  ('Unnamed: 3_level_0', 'Pos'),
                  ('Unnamed: 4_level_0', 'Squad'),
                  ('Unnamed: 5_level_0', 'Comp'),
                  ('Position', 'Pos'),
                  ('Position', 'Alt')]

gk_attributes = [

]

df_attributes = [
        ('Unnamed: 27_level_0', '1/3'),
        ('Total',               'PrgDist'),
        ('Total',               'Cmp%'),
        ('Long', 'Cmp%'),
        ('Vs Dribbles', 'Tkl%'),
        ('Vs Dribbles', 'Past'),
        ('Tackles', 'TklW'),
]

am_attributes = [
        ('Standard',            'Gls'),
        ('Expected_y',          'npxG'),
        ('Expected_y',          'npxG/Sh'),
        ('Unnamed: 24_level_0', 'xA'),
        ('Unnamed: 27_level_0', '1/3'),
        ('Total',               'PrgDist'),
        ('Total',               'Cmp%'),
        ('Dribbles', 'Succ'),
        ('Dribbles',            'Succ%'),
        ('Unnamed: 28_level_0', 'Dispos'),
        ('Vs Dribbles',         'Past')
       ]

fw_attributes = [
        ('Standard',            'Gls'),
        ('Per 90 Minutes', 'npxG'),
        ('Per 90 Minutes', 'xA'),
        ('Standard', 'SoT%'),
        ('Touches', 'Att Pen'),
        ('Carries', 'PrgDist'),
        ('Unnamed: 28_level_0', 'PPA'),
        ('Unnamed: 30_level_0', 'Prog'),
        ('Expected_y',          'npxG/Sh'),
        ('Unnamed: 28_level_0', 'Dispos')
]






######## LAYOUT ########

app = dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = dbc.Container(
    children=[
    html.H1('RABONA DASHBOARD'),

    dbc.Row(children=[
        dbc.Col(dbc.Input(id='player_name_input', placeholder="Enter player's name", type='text', debounce=True)),
        dbc.Col(dbc.RadioItems(id='position_template_picker',
                                        options=[
                                            {'label': 'Goalkeeper', 'value': 'GK'},
                                            {'label': 'Defender', 'value': 'DF'},
                                            {'label': 'Midfielder', 'value': 'AM'},
                                            {'label': 'Forward', 'value': 'FW'}
                                        ],
                                )),
        dbc.Col(dbc.RadioItems(id='compare_with_picker',
                                        options=[
                                             {'label': 'Top 5 Leagues', 'value': 'top'},
                                             {'label': 'Serie A',       'value': 'ita'},
                                             {'label': 'Premier League','value': 'eng'},
                                             {'label': 'Bundesliga',    'value': 'ger'},
                                             {'label': 'League 1',      'value': 'fra'},
                        ]
                               ))]),




    dcc.Graph(id='polar_bar')]
)


def normalize(df, template):
    df_info = df[info_columns]
    df_info.reset_index(inplace=True)

    x = df[template].values  # returns a numpy array
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    normalized = pd.DataFrame(x_scaled, columns=template)
    result = pd.merge(df_info, normalized, left_index=True, right_index=True)
    result.set_index(('Unnamed: 1_level_0', 'Player'), inplace=True)

    return result


#polar plot
@app.callback(
    dash.dependencies.Output('polar_bar', 'figure'),
    [dash.dependencies.Input('player_name_input', 'value'),
     dash.dependencies.Input('position_template_picker', 'value')]
)
def make_polar_chart(playername, template):

    if template == 'AM':
        print('AM')

        player_raw_values   = df.loc[df.index == playername][am_attributes]
        template_df = df.loc[ (df.index == playername) | ( df[('Position', 'Pos')] == 'MF' ) | (df[('Position', 'Alt')] == 'MF') ]
        normalized_to_position = normalize(template_df, am_attributes)
        player_normalized = normalized_to_position.loc[normalized_to_position.index == playername][am_attributes]
        columns = [str(x) for x in am_attributes]

    elif template == 'DF':
        print('DF')

        player_raw_values = df.loc[df.index == playername][df_attributes]
        template_df = df.loc[ (df.index == playername) | (df[('Position', 'Pos')] == 'DF') | (df[('Position', 'Alt')] == 'DF')]    #jesli jakis zawodnik jest MF to nie da sie mu sprawdzic porownania do DF bo nie ma go w tym locu - do poprawienia!!
        normalized_to_position = normalize(template_df, df_attributes)
        player_normalized = normalized_to_position.loc[normalized_to_position.index == playername][df_attributes]
        columns = [str(x) for x in df_attributes]

    elif template == 'FW':
        print('FW')

        player_raw_values = df.loc[df.index == playername][fw_attributes]
        template_df = df.loc[ (df.index == playername) | (df[('Position', 'Pos')] == 'FW') | (df[('Position', 'Alt')] == 'FW')]    #jesli jakis zawodnik jest MF to nie da sie mu sprawdzic porownania do DF bo nie ma go w tym locu - do poprawienia!!
        normalized_to_position = normalize(template_df, fw_attributes)
        player_normalized = normalized_to_position.loc[normalized_to_position.index == playername][fw_attributes]
        columns = [str(x) for x in fw_attributes]

    else:
        pass


    print(player_normalized)
    player_attr_values = player_normalized.values[0].tolist()

    fig = go.Figure()
    fig.add_trace(go.Barpolar(
        r=player_attr_values,  # wartosci atrybutow w kolejnosci przeciwnej do zegara
        name='Shots',
        marker_color='rgb(106,81,163)',
        theta=columns
    ))

    fig.update_traces(
        text= columns)

    fig.update_layout(
        title=playername,
        font_size=16,
        legend_font_size=16,
        polar_angularaxis_rotation=90,
    )

    return fig


if __name__ == '__main__':
    app.run_server(debug=True)