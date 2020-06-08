import plotly.graph_objects as go
import pandas as pd
from sklearn import preprocessing
import dash
import dash_table
import dash_core_components as dcc
import dash_html_components as html
from sklearn import preprocessing
import dash_bootstrap_components as dbc
import os

external_stylesheets = ['https://stackpath.bootstrapcdn.com/bootstrap/4.3.1/css/bootstrap.min.css" integrity="sha384-ggOyR0iXCbMQv3Xipma34MD+dH/1fQ784/j6cY/iJTQUOhcWr7x9JvoRxT2MZw1T']

df = pd.read_pickle(r'./data/merged_fieldplayers.p')


#### attributes

info_columns = [('Unnamed: 2_level_0', 'Nation'),
                  ('Unnamed: 3_level_0', 'Pos'),
                  ('Unnamed: 4_level_0', 'Squad'),
                  ('Unnamed: 5_level_0', 'Comp'),
                  # ('Position', 'Pos'),
                  # ('Position', 'Alt')
                ]

gk_attributes = [

]

df_attributes = {
      "Passes to final third" : ('Unnamed: 27_level_0', '1/3'),
      "Progressive passes distance" : ('Total',               'PrgDist'),
      "Pass completion %" : ('Total',               'Cmp%'),
      "Long pass completion %" : ('Long', 'Cmp%'),
      "Successful tackles %" : ('Vs Dribbles', 'Tkl%'),
      "Dribbled past" : ('Vs Dribbles', 'Past'),
      "Tackles won" : ('Tackles', 'TklW'),
}

am_attributes = {
        "Goals" : ('Standard',            'Gls'),
        "Non-penalty xG" : ('Expected_y',          'npxG'),
        "Non-penalty xG per shot" : ('Expected_y',          'npxG/Sh'),
        "Xa" : ('Unnamed: 24_level_0', 'xA'),
        "Passes to final third" : ('Unnamed: 27_level_0', '1/3'),
        "Progressive passes distance" : ('Total',               'PrgDist'),
        "Pass completion %" : ('Total',               'Cmp%'),
        "Successful dribbles" : ('Dribbles', 'Succ'),
        "Successful dribbes %" : ('Dribbles',            'Succ%'),
        "Disposesed" : ('Unnamed: 28_level_0', 'Dispos')
       }

fw_attributes = {
       'Goals' : ('Standard',            'Gls'),
       "Non-penalty xG per 90'" : ('Per 90 Minutes', 'npxG'),
       "xA per 90'" : ('Per 90 Minutes', 'xA'),
       'Shots on target %' : ('Standard', 'SoT%'),
       'Attempted passes to penalty area' : ('Touches', 'Att Pen'),
       'Progressive carries distance' : ('Carries', 'PrgDist'),
       'Passes to penalty area' : ('Unnamed: 28_level_0', 'PPA'),
       'Progressive passes' : ('Unnamed: 30_level_0', 'Prog'),
       'Non-penalty xG per shot' : ('Expected_y',          'npxG/Sh'),
       'Dispossesed' : ('Unnamed: 28_level_0', 'Dispos')
}






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

    dbc.Row(children = [
        dbc.Col(
            dbc.Card([
                dbc.CardImg(id='player_image'),
                dbc.CardBody(dash_table.DataTable(id='summary_table'))
            ])
        , width=4),

        dbc.Col(
            dcc.Graph(id='polar_bar')
               )
    ])





    ]
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

        player_raw_values   = df.loc[df.index == playername][list(am_attributes.values())]
        template_df = df.loc[ (df.index == playername) | ( df[('Position', 'Pos')] == 'MF' ) | (df[('Position', 'Alt')] == 'MF') ]
        normalized_to_position = normalize(template_df, list(am_attributes.values()))
        player_normalized = normalized_to_position.loc[normalized_to_position.index == playername][list(am_attributes.values())]
        columns = [str(x) for x in list(am_attributes.keys())]

    elif template == 'DF':
        print('DF')

        player_raw_values = df.loc[df.index == playername][list(df_attributes.values())]
        template_df = df.loc[ (df.index == playername) | (df[('Position', 'Pos')] == 'DF') | (df[('Position', 'Alt')] == 'DF')]    #jesli jakis zawodnik jest MF to nie da sie mu sprawdzic porownania do DF bo nie ma go w tym locu - do poprawienia!!
        normalized_to_position = normalize(template_df, list(df_attributes.values()))
        player_normalized = normalized_to_position.loc[normalized_to_position.index == playername][list(df_attributes.values())]
        columns = [str(x) for x in list(df_attributes.keys())]

    elif template == 'FW':
        print('FW')

        player_raw_values = df.loc[df.index == playername][list(fw_attributes.values())]
        template_df = df.loc[ (df.index == playername) | (df[('Position', 'Pos')] == 'FW') | (df[('Position', 'Alt')] == 'FW')]    #jesli jakis zawodnik jest MF to nie da sie mu sprawdzic porownania do DF bo nie ma go w tym locu - do poprawienia!!
        normalized_to_position = normalize(template_df, list(fw_attributes.values()))
        player_normalized = normalized_to_position.loc[normalized_to_position.index == playername][list(fw_attributes.values())]
        columns = [str(x) for x in list(fw_attributes.keys())]

    else:
        pass


    # print(player_normalized)
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
        font_size=16,
        legend_font_size=16,
        polar_angularaxis_rotation=90,
    )

    return fig


@app.callback(
    [dash.dependencies.Output('summary_table', 'data'),
     dash.dependencies.Output('summary_table', 'columns')],
     [dash.dependencies.Input('player_name_input', 'value')]
)
def build_summary_table(input):
    player_info = df.loc[df.index == input][info_columns]
    player_info.reset_index(inplace=True)
    player_info.columns = ['Name', 'Nationality', 'Position', 'Club' , 'League']

    player_info = player_info.transpose()

    data = player_info.to_dict("records")
    columns = [{"name": i, "id": i} for i in [str(x) for x in player_info.columns]]

    return data, columns


@app.callback(
    dash.dependencies.Output('player_image', 'src'),
    [dash.dependencies.Input('player_name_input', 'value')]
)
def get_image(input):
    DIRPATH = os.getcwd() + '\data\images\\'
    imgpath = ''.join([DIRPATH, input, '.png'])

    namepng = '.'.join([input, 'png'])
    output = app.get_asset_url(namepng)

    return output


if __name__ == '__main__':
    app.run_server(debug=True)