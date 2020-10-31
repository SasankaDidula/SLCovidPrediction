import csv
import os
import pickle

import dash_table
import pandas as pd
import dash_html_components as html
import dash_core_components as dcc
from dash import dash
import plotly.express as px
from geopy.geocoders import Nominatim
from opencage.geocoder import OpenCageGeocode

from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

geokey = '92244e2e0ea946d3ac6a1024ef5a74dc'

bgcolors = {
    'background': '#13263a',
    'text': '#FFFFFF'
}
external_scripts = [
    'https://www.google-analytics.com/analytics.js',
    {'src': 'https://cdn.polyfill.io/v2/polyfill.min.js'},
    {
        'src': 'https://cdnjs.cloudflare.com/ajax/libs/lodash.js/4.17.10/lodash.core.js',
        'integrity': 'sha256-Qqd/EfdABZUcAxjOkMi8eGEivtdTkh3b65xCZL4qAQA=',
        'crossorigin': 'anonymous'
    }
]
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
# app stuff
app = dash.Dash(__name__,
                external_scripts=external_scripts,
                external_stylesheets=external_stylesheets)
server = app.server


def createDataset():
    SCOPES = ['https://www.googleapis.com/auth/spreadsheets.readonly']
    SAMPLE_SPREADSHEET_ID = '1zIgPU0ZlYkiKaavYAUcHKgEP95jdaMaf9ljJgRqtog4'
    SAMPLE_RANGE_NAME = 'Patient Data!A2:K'

    creds = None
    if os.path.exists('token.pickle'):
        with open('token.pickle', 'rb') as token:
            creds = pickle.load(token)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                'credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
        with open('token.pickle', 'wb') as token:
            pickle.dump(creds, token)

    service = build('sheets', 'v4', credentials=creds)

    sheet = service.spreadsheets()
    result = sheet.values().get(spreadsheetId=SAMPLE_SPREADSHEET_ID,
                                range=SAMPLE_RANGE_NAME).execute()
    values = result.get('values', [])

    if not values:
        print('No data found.')
    else:
        with open('SLCovid.csv', 'w', newline='') as csvfile:
            fieldnames = ['Number', 'Date_Announced', 'Date_Added', 'Age', 'Gender', 'Residence_City, Prefecture',
                          'Ditected_City',
                          'Detected_Prefecture', 'Status', 'Notes']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for row in values:
                if len(row) == 11:
                    writer.writerow(
                        {fieldnames[0]: row[0], fieldnames[1]: row[1], fieldnames[2]: row[2], fieldnames[3]: row[3],
                         fieldnames[4]: row[4], fieldnames[5]: row[5], fieldnames[6]: row[6], fieldnames[7]: row[7],
                         fieldnames[8]: row[8], fieldnames[9]: row[10]})


#if ~os.path.exists('SLCovid.csv'):
#    createDataset()
df = pd.read_csv('SLCovid.csv')
df1 = df.copy()
df1.drop(['Number', 'Age', 'Gender'], axis=1, inplace=True)
df['Date_Announced'] = df['Date_Announced'].astype('datetime64')
df['Date_Added'] = df['Date_Added'].astype('datetime64')
df2 = df.groupby('Date_Added')['Number'].nunique()

fig0 = px.bar(df2
              , x=df2.index
              , y='Number'
              , title="<b>Daily Cases</b>")

fig0.update_layout(
    template='plotly_dark'
)

df3 = df.groupby(['Detected_Prefecture'])['Number'].count().to_frame().rename(
    columns={'Detected_Prefecture': 'Cases'}).reset_index()
fig1 = px.bar(df3
              , x='Detected_Prefecture'
              , y='Number'
              , title="<b>Cases By City</b>")
# Add figure title
fig1.update_layout(
    template='plotly_dark'
)
df3.head()
data = []  # create empty lists
geocoder = OpenCageGeocode(geokey)


def geometry():
    for index, row in df3.iterrows():  # iterate over rows in dataframe
        lat = 0
        long = 0
        location = ''
        try:
            City = str(row[0])
            query = str(City) + ', Sri Lanka'
            geolocator = Nominatim(user_agent="slcovidprediction")
            location = geolocator.geocode(query)

            if len(location.raw) > 0:
                lat = location.latitude
                long = location.longitude

                data.append({'City': row[0], 'Cases': row[1], 'lat': lat, 'long': long})
        except AttributeError:
            continue
    return data


#if ~os.path.exists('geometry.csv'):
 #   df4 = pd.DataFrame(geometry())
 #   print('geomatry csv created!!')
  #  df4.to_csv('geometry.csv')

df5 = pd.read_csv('geometry.csv')
df5["scaled"] = df5["Cases"] ** 0.5
fig5 = px.scatter_mapbox(
    df5,
    lat="lat",
    lon="long",
    color="Cases",
    size="scaled",
    size_max=50,
    hover_name="City",
    hover_data=["Cases", "City"],
    color_continuous_scale=px.colors.sequential.Cividis_r
)
fig5.update_layout(
    mapbox_style="carto-darkmatter",
    template='plotly_dark'
)

app.layout = html.Div(children=[
    html.Div([
        html.H2("DataSet"),
        dash_table.DataTable(
            id='table',
            columns=[{"name": i, "id": i, "deletable": True, "selectable": True} for i in df1.columns],
            data=df1.to_dict('records'),
            style_cell={
                'font-family': 'sans-serif',
                'font-size': '14px',
                'text-align': 'center'
            },
            editable=True,
            sort_action="native",
            sort_mode="multi",
            row_deletable=True,
            page_action="native",
            page_current=0,
            page_size=10,
            style_table={'overflowX': 'scroll'}
        )
    ], style={'padding-left': '2%', 'padding-right': '2%'}),
    html.Br(),
    html.Br(),
    html.Br(),
    html.Div([
        html.Div([
            html.H4("Daily Cases"),
            dcc.Graph(figure=fig0)
            # ])
        ], className="six columns"
            , style={'padding-left': '2%', 'padding-right': '2%',
                     'vertical-align': 'middle'})

        , html.Br(),

        html.Div([
            html.H4("Cases by City"),
            dcc.Graph(figure=fig1)
            # ])
        ], className="six columns"
            , style={'padding-left': '2%', 'padding-right': '2%',
                     'margin-top': -24})

    ], className="row"),
    html.Br(),
    html.Br(),
    html.Div([
        html.H4("Map"),
        dcc.Graph(figure=fig5)
    ]),
    html.Br(),
], style={'padding-left': '10%'
    , 'padding-right': '10%'})

if __name__ == '__main__':
    app.run_server(debug=True)
