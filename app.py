import pickle

from fbprophet import Prophet
from pandas import to_datetime
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.svm import SVR
from itertools import chain

import Datasets

import pandas as pd
import numpy as np
import dash_html_components as html
import dash_core_components as dcc
from dash import dash
import plotly.express as px
import plotly.graph_objects as go
from dash.dependencies import Input, Output

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

app = dash.Dash(__name__,
                external_scripts=external_scripts,
                external_stylesheets=external_stylesheets)
server = app.server

Datasets.createDatasetForSL()
Datasets.SLdatasetbyCity()
Datasets.SLdatasetbyDate()
Datasets.geometry()

# Daily Cases graph
dfsumbyday = pd.read_csv('sumbyday.csv')
dfsumbyday['Date'] = dfsumbyday['Date'].astype('datetime64')
dfsumbyday['Date'] = dfsumbyday['Date'].astype('datetime64')

fig0 = px.line(dfsumbyday
               , x='Date'
               , y='Confirmed'
               , title="<b>Total Cases</b>")

fig0.update_layout(
    template='plotly_dark'
)

dfsumbyday = pd.read_csv('sumbyday.csv')
dfsumbyday['Date'] = dfsumbyday['Date'].astype('datetime64')
dfsumbyday['Date'] = dfsumbyday['Date'].astype('datetime64')

fig1 = px.line(dfsumbyday
               , x='Date'
               , y='Recovered'
               , title="<b>Total Recovered</b>")

fig1.update_layout(
    template='plotly_dark'
)

df = Datasets.SLDataPreprocess()
df['Date_Announced'] = df['Date_Announced'].astype('datetime64')
df['Date_Added'] = df['Date_Added'].astype('datetime64')
newDataSet = df.groupby(['Detected_Prefecture'])['Number'].count().to_frame().rename(
    columns={'Detected_Prefecture': 'Cases'}).reset_index()
fig2 = px.bar(newDataSet
              , x='Detected_Prefecture'
              , y='Number'
              , title="<b>Cases By City</b>"
              , height=600)
# Add figure title
fig2.update_layout(
    template='plotly_dark'
)

df5 = pd.read_csv('geometry.csv')
df5["scaled"] = df5["Cases"] ** 0.8
fig5 = px.scatter_mapbox(
    df5,
    lat="lat",
    lon="long",
    color="Cases",
    size="scaled",
    size_max=75,
    hover_name="City",
    hover_data=["Cases"],
    color_continuous_scale=px.colors.sequential.Cividis_r
    , height=600
)
fig5.update_layout(
    mapbox_style="carto-darkmatter",
    template='plotly_dark'
)

SLDailyCasesPrediction = pd.read_csv('sumbyday.csv')
SLDailyCasesPrediction.drop(['Recovered', 'Deceased', 'Critical', 'Tested'], axis=1, inplace=True)
newDataSet2 = SLDailyCasesPrediction.copy()
newDataSet2.columns = ['ds', 'y']
newDataSet2['ds'] = to_datetime(newDataSet2['ds'])
model = Prophet(interval_width=0.95)
model.fit(newDataSet2)
future = model.make_future_dataframe(periods=30)
forecast1 = model.predict(newDataSet2)
forecast2 = model.predict(future.tail(30))
y_pred1 = forecast1['yhat'].values
y_pred2 = forecast2['yhat'].values
y_true = newDataSet2['y'].values
figfbprophet = go.Figure()
figfbprophet.add_trace(go.Scatter(x=future['ds']
                                  , y=y_pred1
                                  , mode='lines',
                                  name='predict'))
figfbprophet.add_trace(go.Scatter(x=future['ds']
                                  , y=y_true
                                  , mode='lines',
                                  name='Actual'))
figfbprophet.add_trace(go.Scatter(x=forecast2['ds']
                                  , y=y_pred2
                                  , mode='lines+markers',
                                  name='predict for next month'))

figfbprophet.update_layout(
    mapbox_style="carto-darkmatter",
    template='plotly_dark'
)


def preprocessing(url):
    Df_dataset = pd.read_csv(url, error_bad_lines=False)

    Df_dataset.drop(['Country/Region', 'Lat', 'Long'], axis=1, inplace=True)

    col_num = 0
    TotalObjects = Df_dataset.shape[0]
    print("Column\t\t\t\t\t Null Values%")

    for x in Df_dataset:
        nullCount = Df_dataset[x].isnull().sum()
        nullPercent = nullCount * 100 / TotalObjects
        if nullCount > 0 and nullPercent > 30:
            col_num = col_num + 1
            Df_dataset.drop(x, axis=1, inplace=True)
            print(str(x) + "\t\t\t\t\t " + str(nullPercent))
    print("A total of " + str(col_num) + " deleted !")

    df1_transposed = Df_dataset.T  # or df1.transpose()
    df1_new = df1_transposed[2:266] - df1_transposed[2:266].shift()
    df1_new = df1_new.replace(np.nan, 0)
    return df1_new


def convert_date(df1_new, countryIndex, date):
    float_date = pd.to_datetime(date).toordinal()
    val = df1_new[countryIndex]
    date = []
    strDate = []
    Total_cases = []

    for x in val.index.values:
        strDate.append(x)
        y = pd.to_datetime(x).toordinal()
        date.append(y)

    for x in val.values:
        if x < 0:
            Total_cases.append(0)
        else:
            Total_cases.append(x)

    return date, Total_cases, [float_date], strDate


def predict_cases_global(dates, cases, predictDate, strDate):
    dates1 = np.reshape(dates, (len(dates), 1))  # convert to 1xn dimension
    cases1 = np.reshape(cases, (len(cases), 1))
    predictDate = np.reshape(predictDate, (len(predictDate), 1))

    svr_rbf = Pipeline([('scaler', StandardScaler()), ('svr', SVR(kernel='rbf', C=1e3, cache_size=7000, gamma=0.1))])

    # Fit regression model
    svr_rbf.fit(dates1, cases1)
    y_pred1 = svr_rbf.predict(dates1)
    y_pred2 = svr_rbf.predict(predictDate)
    fig = go.Figure(layout=go.Layout(
        title=go.layout.Title(text="cases predict"),
        template='plotly_dark'
    ))

    fig.add_trace(go.Scatter(x=strDate,
                             y=cases
                             , mode='lines',
                             name='Actual'))
    fig.add_trace(go.Scatter(x=strDate
                             , y=y_pred1
                             , mode='lines',
                             name='predict'))
    fig.add_trace(go.Scatter(x=['03/11/20']
                             , y=y_pred2
                             , mode='lines+markers',
                             name='predict for the selected date'))

    return fig, y_pred2


def predict_cases_country(countryIndex, date):
    url = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data' \
          '/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv '
    df1_new = preprocessing(url)
    data = convert_date(df1_new, countryIndex, date)
    predicted_cases = predict_cases_global(data[0], data[1], data[2], data[3])
    print(predicted_cases)
    return predicted_cases


def predict_deaths(dates, deaths, predictDate, strDate):
    dates1 = np.reshape(dates, (len(dates), 1))  # convert to 1xn dimension
    deaths1 = np.reshape(deaths, (len(deaths), 1))
    predictDate = np.reshape(predictDate, (len(predictDate), 1))

    regression_model = LinearRegression()
    # Fit linear_regression model
    regression_model.fit(dates1, deaths1)

    y1 = regression_model.predict(dates1)
    y2 = regression_model.predict(predictDate)
    y1 = list(chain(*y1))
    y2 = list(chain(*y2))
    print(y1)
    print("aaaa:", y2)
    fig = go.Figure(layout=go.Layout(
        title=go.layout.Title(text="death predict"),
        template='plotly_dark'
    ))

    fig.add_trace(go.Scatter(x=strDate,
                             y=deaths
                             , mode='lines',
                             name='Actual'))
    fig.add_trace(go.Scatter(x=strDate
                             , y=y1
                             , mode='lines',
                             name='predict'))
    fig.add_trace(go.Scatter(x=['03/11/20']
                             , y=y2
                             , mode='lines+markers',
                             name='predict for the selected date'))

    return fig, y2


def predict_deaths_country(countryIndex, date):
    url = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data' \
          '/csse_covid_19_time_series/time_series_covid19_deaths_global.csv '
    df1_new = preprocessing(url)
    data = convert_date(df1_new, countryIndex, date)
    predicted_deaths = predict_deaths(data[0], data[1], data[2], data[3])
    return predicted_deaths


def predict_recovered(dates, recovered, x, strDate):
    dates = np.reshape(dates, (len(dates), 1))  # convert to 1xn dimension
    x = np.reshape(x, (len(x), 1))

    polynominal_regg = PolynomialFeatures(degree=3)
    x_Polynom = polynominal_regg.fit_transform(dates)

    leniar_regg = LinearRegression()
    leniar_regg.fit(x_Polynom, recovered)

    y1 = leniar_regg.predict(polynominal_regg.fit_transform(dates))
    y2 = leniar_regg.predict(polynominal_regg.fit_transform(x))
    fig = go.Figure(layout=go.Layout(
        title=go.layout.Title(text="recovered predict"),
        template='plotly_dark'
    ))

    fig.add_trace(go.Scatter(x=strDate,
                             y=recovered
                             , mode='lines',
                             name='Actual'))
    fig.add_trace(go.Scatter(x=strDate
                             , y=y1
                             , mode='lines',
                             name='predict'))
    fig.add_trace(go.Scatter(x=['03/11/20']
                             , y=y2
                             , mode='lines+markers',
                             name='predict for the selected date'))

    return fig, y2


def predict_recovered_country(countryIndex, date):
    url = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data' \
          '/csse_covid_19_time_series/time_series_covid19_recovered_global.csv '
    df1_new = preprocessing(url)
    data = convert_date(df1_new, countryIndex, date)
    print(type(data[0]))
    predicted_recovered = predict_recovered(data[0], data[1], data[2], data[3])
    return predicted_recovered


def predict_cases_sldata():
    Df_dataset = Datasets.SLDataPreprocess()
    new1 = Df_dataset[["Date_Added", "Detected_Prefecture"]]
    a = new1.groupby("Date_Added").size().values
    df1 = new1.drop_duplicates(subset="Date_Added").assign(Count=a)
    dfnew = df1.pivot_table('Count', ['Date_Added'], 'Detected_Prefecture')
    dfnew.fillna(0, inplace=True)
    return dfnew


def predict_cases(dates, cases, predictDate):
    dates = dates
    cases = cases
    dates1 = np.reshape(dates, (len(dates), 1))  # convert to 1xn dimension
    cases1 = np.reshape(cases, (len(cases), 1))
    predictDate = np.reshape(predictDate, (len(predictDate), 1))

    polynominal_regg = PolynomialFeatures(degree=3)
    x_Polynom = polynominal_regg.fit_transform(dates1)

    leniar_regg = LinearRegression()
    leniar_regg.fit(x_Polynom, cases1)

    y1 = leniar_regg.predict(x_Polynom)
    y2 = leniar_regg.predict(polynominal_regg.fit_transform(predictDate))

    fig = go.Figure(layout=go.Layout(
        title=go.layout.Title(text="cases predict for city"),
        template='plotly_dark'
    ))
    y1 = list(chain(*y1))
    y2 = list(chain(*y2))

    fig.add_trace(go.Scatter(x=dates,
                             y=cases
                             , mode='lines',
                             name='Actual'))
    fig.add_trace(go.Scatter(x=dates
                             , y=y1
                             , mode='lines',
                             name='predict'))
    fig.add_trace(go.Scatter(x=['11/11/20']
                             , y=y2
                             , mode='lines+markers',
                             name='predict for the selected date'))

    return fig, y2


def predict_cases_sl(index, date):
    float_date = pd.to_datetime(date).toordinal()
    dfnew = predict_cases_sldata()
    val = dfnew[index]
    date = []

    for x in val.index.values:
        y = pd.to_datetime(x).toordinal()
        date.append(y)

    Total_cases = val.values
    predicted_cases = predict_cases(date, Total_cases, [float_date])
    return predicted_cases


predictType = ["New Cases", "New Recoverd", "New Deaths"]
countries = ['Afghanistan',
             'Albania',
             'Algeria',
             'Andorra',
             'Angola',
             'Antigua and Barbuda',
             'Argentina',
             'Armenia',
             'Australia/Australian Capital Territory',
             'Australia/New South Wales',
             'Australia/Northern Territory',
             'Australia/Queensland',
             'Australia/South Australia',
             'Australia/Tasmania',
             'Australia/Victoria',
             'Australia/Western Australia',
             'Austria',
             'Azerbaijan',
             'Bahamas',
             'Bahrain',
             'Bangladesh',
             'Barbados',
             'Belarus',
             'Belgium',
             'Belize',
             'Benin',
             'Bhutan',
             'Bolivia',
             'Bosnia and Herzegovina',
             'Botswana',
             'Brazil',
             'Brunei',
             'Bulgaria',
             'Burkina Faso',
             'Burma',
             'Burundi',
             'Cabo Verde',
             'Cambodia',
             'Cameroon',
             'Canada/Alberta',
             'Canada/British Columbia',
             'Canada/Diamond Princess',
             'Canada/Grand Princess',
             'Canada/Manitoba',
             'Canada/New Brunswick',
             'Canada/Newfoundland and Labrador',
             'Canada/Northwest Territories',
             'Canada/Nova Scotia',
             'Canada/Ontario',
             'Canada/Prince Edward Island',
             'Canada/Quebec',
             'Canada/Saskatchewan',
             'Canada/Yukon',
             'Central African Republic',
             'Chad',
             'Chile',
             'China/Anhui',
             'China/Beijing',
             'China/Chongqing',
             'China/Fujian',
             'China/Gansu',
             'China/Guangdong',
             'China/Guangxi',
             'China/Guizhou',
             'China/Hainan',
             'China/Hebei',
             'China/Heilongjiang',
             'China/Henan',
             'China/Hong Kong',
             'China/Hubei',
             'China/Hunan',
             'China/Inner Mongolia',
             'China/Jiangsu',
             'China/Jiangxi',
             'China/Jilin',
             'China/Liaoning',
             'China/Macau',
             'China/Ningxia',
             'China/Qinghai',
             'China/Shaanxi',
             'China/Shandong',
             'China/Shanghai',
             'China/Shanxi',
             'China/Sichuan',
             'China/Tianjin',
             'China/Tibet',
             'China/Xinjiang',
             'China/Yunnan',
             'China/Zhejiang',
             'Colombia',
             'Comoros',
             'Congo (Brazzaville)',
             'Congo (Kinshasa)',
             'Costa Rica',
             "Cote d'Ivoire",
             'Croatia',
             'Cuba',
             'Cyprus',
             'Czechia',
             'Denmark/Faroe Islands',
             'Denmark/Greenland',
             'Denmark',
             'Diamond Princess',
             'Djibouti',
             'Dominica',
             'Dominican Republic',
             'Ecuador',
             'Egypt',
             'El Salvador',
             'Equatorial Guinea',
             'Eritrea',
             'Estonia',
             'Eswatini',
             'Ethiopia',
             'Fiji',
             'Finland',
             'France/French Guiana',
             'France/French Polynesia',
             'France/Guadeloupe',
             'France/Martinique',
             'France/Mayotte',
             'France/New Caledonia',
             'France/Reunion',
             'France/Saint Barthelemy',
             'France/Saint Pierre and Miquelon',
             'France/St Martin',
             'France',
             'Gabon',
             'Gambia',
             'Georgia',
             'Germany',
             'Ghana',
             'Greece',
             'Grenada',
             'Guatemala',
             'Guinea',
             'Guinea-Bissau',
             'Guyana',
             'Haiti',
             'Holy See',
             'Honduras',
             'Hungary',
             'Iceland',
             'India',
             'Indonesia',
             'Iran',
             'Iraq',
             'Ireland',
             'Israel',
             'Italy',
             'Jamaica',
             'Japan',
             'Jordan',
             'Kazakhstan',
             'Kenya',
             'Korea, South',
             'Kosovo',
             'Kuwait',
             'Kyrgyzstan',
             'Laos',
             'Latvia',
             'Lebanon',
             'Lesotho',
             'Liberia',
             'Libya',
             'Liechtenstein',
             'Lithuania',
             'Luxembourg',
             'MS Zaandam',
             'Madagascar',
             'Malawi',
             'Malaysia',
             'Maldives',
             'Mali',
             'Malta',
             'Mauritania',
             'Mauritius',
             'Mexico',
             'Moldova',
             'Monaco',
             'Mongolia',
             'Montenegro',
             'Morocco',
             'Mozambique',
             'Namibia',
             'Nepal',
             'Netherlands/Aruba',
             'Netherlands/Bonaire, Sint Eustatius and Saba',
             'Netherlands/Curacao',
             'Netherlands/Sint Maarten',
             'Netherlands',
             'New Zealand',
             'Nicaragua',
             'Niger',
             'Nigeria',
             'North Macedonia',
             'Norway',
             'Oman',
             'Pakistan',
             'Panama',
             'Papua New Guinea',
             'Paraguay',
             'Peru',
             'Philippines',
             'Poland',
             'Portugal',
             'Qatar',
             'Romania',
             'Russia',
             'Rwanda',
             'Saint Kitts and Nevis',
             'Saint Lucia',
             'Saint Vincent and the Grenadines',
             'San Marino',
             'Sao Tome and Principe',
             'Saudi Arabia',
             'Senegal',
             'Serbia',
             'Seychelles',
             'Sierra Leone',
             'Singapore',
             'Slovakia',
             'Slovenia',
             'Somalia',
             'South Africa',
             'South Sudan',
             'Spain',
             'Sri Lanka',
             'Sudan',
             'Suriname',
             'Sweden',
             'Switzerland',
             'Syria',
             'Taiwan*',
             'Tajikistan',
             'Tanzania',
             'Thailand',
             'Timor-Leste',
             'Togo',
             'Trinidad and Tobago',
             'Tunisia',
             'Turkey',
             'US',
             'Uganda',
             'Ukraine',
             'United Arab Emirates',
             'United Kingdom/Anguilla',
             'United Kingdom/Bermuda',
             'United Kingdom/British Virgin Islands',
             'United Kingdom/Cayman Islands',
             'United Kingdom/Channel Islands',
             'United Kingdom/Falkland Islands (Malvinas)',
             'United Kingdom/Gibraltar',
             'United Kingdom/Isle of Man',
             'United Kingdom/Montserrat',
             'United Kingdom/Turks and Caicos Islands',
             'United Kingdom',
             'Uruguay',
             'Uzbekistan',
             'Venezuela',
             'Vietnam',
             'West Bank and Gaza',
             'Western Sahara',
             'Yemen',
             'Zambia',
             'Zimbabwe']
city = ['Gampaha', 'Colombo', 'Kalutara', 'Kurunegala', 'Jaffna', 'Ratnapura', 'Kegalle', 'Puttalam', 'Polonnaruwa',
        'Galle', 'Kandy', 'Anuradhapura', 'Nuwara Eliya', 'Vavuniya', 'Badulla,Hambantota', 'Matale', 'Batticaloa',
        'Mannar', 'rincomalee', 'Moneragala', 'Matara', 'Ampara', 'Godagama']

app.layout = html.Div([
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
            html.H4("Total Recovered"),
            dcc.Graph(figure=fig1)
            # ])
        ], className="six columns"
            , style={'padding-left': '2%', 'padding-right': '2%',
                     'margin-top': -24})

    ], className="row"),
    html.Br(),
    html.Br(),
    html.Div([
        html.H4("Total Cases by City"),
        dcc.Graph(figure=fig2)
    ]),
    html.Br(),
    html.Div([
        html.H4("Cases by city"),
        dcc.Graph(figure=fig5)
    ]),
    html.Br(),
    html.Div([
        html.H4("Sri Lanka Cases prediction"),
        dcc.Graph(figure=figfbprophet)
    ]),
    html.Br(),
    html.Br(),
    html.Br(),
    html.Br(),
    html.Div([
        html.H4("Local prediction"),
        html.Div([

            dcc.Dropdown(
                id='city',
                options=[{'label': i, 'value': i} for i in city],
                value='Colombo'
            ),
        ],
            style={'width': '48%', 'display': 'inline-block'}),

    ]),
    dcc.Graph(id='SlPrectionCity'),
    html.Br(),
    html.Br(),
    html.Div([
        html.H4("Global prediction"),
        html.Div([
            dcc.Dropdown(
                id='typepred',
                options=[{'label': i, 'value': i} for i in predictType],
                value='New Recoverd'
            ),
        ],
            style={'width': '48%', 'display': 'inline-block'}),

        html.Div([
            dcc.Dropdown(
                id='country',
                options=[{'label': i, 'value': i} for i in countries],
                value='Sri Lanka'
            ),
        ], style={'width': '48%', 'float': 'right', 'display': 'inline-block'}),
        html.Br(),
        dcc.Graph(id='SlPrection'),

    ]),
], style={'padding-left': '10%'
    , 'padding-right': '10%'})

if __name__ == '__main__':
    app.run_server(debug=True)


@app.callback(
    Output('SlPrection', 'figure'),
    [Input('typepred', 'value'),
     Input('country', 'value')]
)
def update_graph_global(typepred, country):
    index = countries.index(country)
    if typepred == 'New Cases':
        output = predict_cases_country(index, '2020-11-03')
    elif typepred == 'New Deaths':
        output = predict_deaths_country(index, '2020-11-03')
    else:
        output = predict_recovered_country(index, '2020-11-03')
    fig = output[0]
    return fig


@app.callback(
    Output('SlPrectionCity', 'figure'),
    [Input('city', 'value')]
)
def update_table_local(cityname):
    fig = predict_cases_sl(cityname, '2020-11-03')[0]
    return fig
