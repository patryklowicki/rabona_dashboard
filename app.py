import plotly.graph_objects as go
import pandas as pd
import numpy as np
from sklearn import preprocessing
import dash
import dash_table
import dash_core_components as dcc
import dash_html_components as html
from sklearn import preprocessing
import dash_bootstrap_components as dbc
import os
import plotly.figure_factory as ff



df = pd.read_pickle(r'./data/merged_fieldplayers.p')
gk = pd.read_pickle(r'./data/merged_gk.p')


#### attributes

info_columns = [('Unnamed: 2_level_0', 'Nation'),
                  ('Unnamed: 3_level_0', 'Pos'),
                  ('Unnamed: 4_level_0', 'Squad'),
                  ('Unnamed: 5_level_0', 'Comp'),
                  ('Playing Time_x', 'MP'),
                  ('Playing Time_x', 'Starts'),
                  ('Playing Time_x', 'Min'),
                  ('Playing Time_y', 'Min%'),
                  ('Performance', 'Gls'),
                  ('Performance', 'Ast')
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
        'Passes into Box' : ('Unnamed: 28_level_0', 'PPA'),
        "Progressive passes distance" : ('Total',               'PrgDist'),
        "Pass completion %" : ('Total',               'Cmp%'),
        "Successful dribbles" : ('Dribbles', 'Succ'),
        "Successful dribbes %" : ('Dribbles',            'Succ%'),
        "Disposesed" : ('Unnamed: 28_level_0', 'Dispos'),
        'Successful Pressures' : ('Pressures', 'Succ'),
       }

fw_attributes = {
    'Goals': ('Per 90 Minutes', 'Gls'),
    'Non-Penalty xG': ('Per 90 Minutes', 'npxG'),
    'Non-penalty xG/Shot' : ('Expected_y', 'npxG/Sh'),
    'Conversion Rate' : ('Standard', 'G/Sh'),
    'Touches in Box'  : ('Touches', 'Att Pen'),
    'xA' : ('Per 90 Minutes', 'xA'),
    'Passes into Box' : ('Unnamed: 28_level_0', 'PPA'),
    'Succesfull Dribbles' : ('Dribbles', 'Succ'),
    'Dribbles Success %' : ('Dribbles', 'Succ%'),
    'Successful Pressures' : ('Pressures', 'Succ'),
}



#CYBORG


######## LAYOUT ########

external_stylesheets = ['https://stackpath.bootstrapcdn.com/bootstrap/4.3.1/css/bootstrap.min.css" integrity="sha384-ggOyR0iXCbMQv3Xipma34MD+dH/1fQ784/j6cY/iJTQUOhcWr7x9JvoRxT2MZw1T']
app = dash.Dash(external_stylesheets=[dbc.themes.FLATLY])

app.layout = dbc.Container(
    children=[
    html.H1('RABONA DASHBOARD'),

    dbc.Collapse(
        dbc.Jumbotron(
            [
            html.H2('Welcome to Rabona Dashboard', className='display-3'),
            html.P('Rabona is a stats visualization tool powered that allows you to see players statistics TOP 5 european football leagues '),
            html.Hr(className='my-2'),
            html.P("Type player's name to see stats vizualization. You can change positional template and filter by league here. "
                   "You can see more details on a distplot when you click on single bar/attribute"
                   "Hide this message by clicking button below or just start typing player's name! "),
            dbc.Button(
                   "Got it",
                   id="collapse-button",
                   className="mb-3",
                   color="primary",
                   block=True),
            ],
        id='jumbotron'
        ),  #jumbotron
    id='collapse',
    is_open = True) #collapse
    ,

    dbc.Row(children=[
        dbc.Col(dbc.Input(id='player_name_input', placeholder="Enter player's name", type='text', debounce=True, bs_size='lg', className='mb-3')),

        dbc.Col(dbc.RadioItems(id='position_template_picker',
                                        options=[
                                            {'label': 'Goalkeeper', 'value': 'GK'},
                                            {'label': 'Defender', 'value': 'DF'},
                                            {'label': 'Midfielder', 'value': 'MF'},
                                            {'label': 'Forward', 'value': 'FW'}
                                        ],
                                )),
        dbc.Col(dbc.RadioItems(id='compare_with_picker',
                                        options=[
                                             {'label': 'Top 5 Leagues', 'value': 'top'},
                                             {'label': 'Serie A',       'value': 'Serie A'},
                                             {'label': 'Premier League','value': 'Premier League'},
                                             {'label': 'Bundesliga',    'value': 'Bundesliga'},
                                             {'label': 'Ligue 1',      'value': 'Ligue 1'},
                                                ],
                               ))]
    ),

    dbc.Row(id='info_row',
            children = [
        dbc.Col(children=[
            dbc.Card([
                dbc.CardImg(id='player_image'),
                dbc.CardBody(dash_table.DataTable(id='summary_table',
                                                  style_as_list_view = True,
                                                  style_header = {'textAlign'  : 'left',
                                                                  'fontFamily' : 'Helvetica',
                                                                  'fontWeight' : 'bold',
                                                                  'fontSize' : 12,
                                                                  },

                                                  style_cell = { 'textAlign' : 'left',
                                                                 'fontFamily': 'Helvetica' },

                                                  ))
            ]),
           ]
        , width=3),

        dbc.Col(
            dcc.Graph(id='polar_bar'),
               ),
    ], style = {'display' : 'none'}),

    dbc.Row(id='second_row',
            children = [
                dbc.Col(
                    dcc.Graph(id='distplot')
                )
    ], style= {'display': 'none'}
    )

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


@app.callback(
    dash.dependencies.Output("collapse", "is_open"),
    [dash.dependencies.Input("collapse-button", "n_clicks")],
    [dash.dependencies.State("collapse", "is_open")],
)
def toggle_collapse(n, is_open):
    if n:
        return not is_open
    return is_open


@app.callback(
    [dash.dependencies.Output('position_template_picker', 'value'),
    dash.dependencies.Output('compare_with_picker', 'value')],
    [dash.dependencies.Input('player_name_input', 'value')]
)
def initial_radio_selection(playername):
    player_data =  df.loc[ (df.index == playername) ]
    position = player_data['Position', 'Pos'][0]

    initial_top5 = 'top'

    return position, initial_top5



#polar plot
@app.callback(
    [dash.dependencies.Output('polar_bar', 'figure'),
    dash.dependencies.Output('info_row', 'style'),
     dash.dependencies.Output('jumbotron', 'style')],
    [dash.dependencies.Input('player_name_input', 'value'),
     dash.dependencies.Input('position_template_picker', 'value'),
     dash.dependencies.Input('compare_with_picker', 'value')]
)
def make_polar_chart(playername, template, league):

    if league != 'top':
        print(league)
        df_league = df.loc[ (df.index == playername) | (df[('info', 'league')] == league)]
    else:
        print(league)
        df_league = df


    if template == 'MF':
        print('MF')

        player_raw_values   = df.loc[df.index == playername][list(am_attributes.values())]
        template_df_pre = df_league.loc[ (df_league.index == playername) | ( df_league[('Position', 'Pos')] == 'MF' ) | (df_league[('Position', 'Alt')] == 'MF') ]
        template_df = template_df_pre.loc[template_df_pre[('Playing Time_y', 'Min%')] > 50]
        normalized_to_position = normalize(template_df, list(am_attributes.values()))
        player_normalized = normalized_to_position.loc[normalized_to_position.index == playername][list(am_attributes.values())]
        columns = [str(x) for x in list(am_attributes.keys())]
        customdata = [x for x in list(am_attributes.values())]

    elif template == 'DF':
        print('DF')

        player_raw_values = df.loc[df.index == playername][list(df_attributes.values())]
        template_df_pre = df_league.loc[ (df_league.index == playername) | (df_league[('Position', 'Pos')] == 'DF') | (df_league[('Position', 'Alt')] == 'DF')]    #jesli jakis zawodnik jest MF to nie da sie mu sprawdzic porownania do DF bo nie ma go w tym locu - do poprawienia!!
        template_df = template_df_pre.loc[template_df_pre[('Playing Time_y', 'Min%')] > 50]
        normalized_to_position = normalize(template_df, list(df_attributes.values()))
        player_normalized = normalized_to_position.loc[normalized_to_position.index == playername][list(df_attributes.values())]
        columns = [str(x) for x in list(df_attributes.keys())]
        customdata = [x for x in list(df_attributes.values())]

    elif template == 'FW':
        print('FW')

        player_raw_values = df.loc[df.index == playername][list(fw_attributes.values())]
        template_df_pre = df_league.loc[ (df_league.index == playername) | (df_league[('Position', 'Pos')] == 'FW') | (df_league[('Position', 'Alt')] == 'FW')]    #jesli jakis zawodnik jest MF to nie da sie mu sprawdzic porownania do DF bo nie ma go w tym locu - do poprawienia!!
        template_df = template_df_pre.loc[template_df_pre[('Playing Time_y', 'Min%')] > 50]
        normalized_to_position = normalize(template_df, list(fw_attributes.values()))
        player_normalized = normalized_to_position.loc[normalized_to_position.index == playername][list(fw_attributes.values())]
        columns = [str(x) for x in list(fw_attributes.keys())]
        customdata = [x for x in list(fw_attributes.values())]

    else:
        pass # automaticaly set for the actual player's position


    # print(player_normalized)
    player_attr_values = player_normalized.values[0].tolist()

    fig = go.Figure()
    fig.add_trace(go.Barpolar(
        r=player_attr_values,  # wartosci atrybutow w kolejnosci przeciwnej do zegara
        name='Shots',
        marker_color='#119DFF',
        theta=columns,
        customdata=customdata
    ))

    fig.update_traces(
        text= player_raw_values.values[0].tolist()
    )

    fig.update_layout(
        polar_angularaxis={
            'direction':'clockwise'
        },
        polar_radialaxis={
          'range' : [0,1],
           'tickmode' : 'array',
          'tickvals' : [0.25, 0.5, 0.75],
        },
        # template = 'plotly_dark',
        width=600,
        height=600,
        font_size=14,
        legend_font_size=14,
        polar_angularaxis_rotation=90,
    )

    style_content   = {'display': 'flex'}
    style_jumbotron = {'display': 'none'}
    return fig, style_content, style_jumbotron


# SUMMARY TABLE
@app.callback(
    [dash.dependencies.Output('summary_table', 'data'),
     dash.dependencies.Output('summary_table', 'columns')],
     [dash.dependencies.Input('player_name_input', 'value')]
)
def build_summary_table(input):
    player_info = df.loc[df.index == input][info_columns]
    player_info.reset_index(inplace=True)
    player_info.columns = ['Name', 'Nationality', 'Position', 'Club' , 'League', 'Matches', 'Starts', 'Minutes', 'Minutes %', 'Goals', 'Assists']

    player_info = player_info.transpose()
    player_info.reset_index(inplace=True)
    player_info.columns = ['', input]


    data = player_info[1:].to_dict("records")
    columns = [{"name": i, "id": i} for i in [str(x) for x in player_info.columns]]

    return data, columns


#GET PLAYER IMAGE
@app.callback(
    dash.dependencies.Output('player_image', 'src'),
    [dash.dependencies.Input('player_name_input', 'value')]
)
def get_image(input):
    IMAGEDIR = os.getcwd() + r'\assets\\'
    filepath = ''.join([IMAGEDIR, input, '.png'])
    if os.path.isfile(filepath):
        img = ''.join([input, '.png'])
    else:
        img = ''.join(['blank_avatar', '.png'])

    output = app.get_asset_url(img)

    return output


#distplot
@app.callback(
    [dash.dependencies.Output('distplot', 'figure'),
     dash.dependencies.Output('second_row', 'style')],
    [dash.dependencies.Input('player_name_input', 'value'),
     dash.dependencies.Input('position_template_picker', 'value'),
     dash.dependencies.Input('polar_bar', 'clickData')]
)
def make_distplot(playername, position, clickData):
    print(tuple(clickData['points'][0]['customdata']))
    position = position
    minimum_minutes_treshold = 50
    attribute = tuple(clickData['points'][0]['customdata'])
    player_attribute_value = df.loc[df.index == playername][attribute][0]

    xg = df.loc[(df[('Playing Time_y', 'Min%')] > minimum_minutes_treshold) &
                (df[('Position', 'Pos')] == position)].sort_values(by=attribute, ascending=False)[[
        ('Playing Time_x', 'MP'),
        ('Playing Time_x', 'Starts'),
        ('Playing Time_x', 'Min'), ('Playing Time_y', 'Min%'), attribute]]

    xg.drop_duplicates(inplace=True)
    lxg = np.array(xg[attribute])

    fig = ff.create_distplot([lxg], [str(attribute)], bin_size=0.01, show_rug=True, show_hist=False)
    fig.update_layout(showlegend=False,
                      height=600,
                      # template = 'plotly_dark',
                      annotations=[
                          dict(
                              x=player_attribute_value,
                              y=0,
                              xref="x",
                              yref="y",
                              text=playername,
                              showarrow=True,
                              arrowhead=4,
                              arrowcolor="red",
                              ax=0,
                              ay=-300
                          )
                      ])

    style_property = {'display' : 'block'}

    return fig, style_property




if __name__ == '__main__':
    app.run_server(debug=True)