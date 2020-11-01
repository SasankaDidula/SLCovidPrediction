import csv
import os
import pickle

import dash_table
import pandas as pd
import dash_html_components as html
import dash_core_components as dcc
from dash import dash
import plotly.express as px
from opencage.geocoder import OpenCageGeocode

from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR
from datetime import datetime as dt
from sklearn.linear_model import LinearRegression
from datetime import datetime as dt
#get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
get_ipython().run_line_magic('matplotlib', 'inline')


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


if ~os.path.exists('SLCovid.csv'):
    createDataset()
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
# Add figure title
fig0.update_layout(
    template='plotly_dark'
)

df3 = df.groupby('Ditected_City')['Number'].nunique()

fig1 = px.bar(df3
              , x=df3.index
              , y='Number'
              , title="<b>Cases By City</b>")
# Add figure title
fig1.update_layout(
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
    html.Br()
], style={'padding-left': '10%'
    , 'padding-right': '10%'})

if __name__ == '__main__':
    app.run_server(debug=True)


##preprocessing

def preprocessing(url):

    Df_dataset = pd.read_csv(url, error_bad_lines=False)

    col_num = 0
    TotalObjects = Df_dataset.shape[0]
    print("Column\t\t\t\t\t Null Values%")
    for x in Df_dataset:
        nullCount = Df_dataset[x].isnull().sum();
        nullPercent = nullCount * 100 / (TotalObjects)
        if nullCount > 0 and nullPercent > 30:
            col_num = col_num + 1
            Df_dataset.drop(x, axis=1, inplace=True)
            print(str(x) + "\t\t\t\t\t " + str(nullPercent))
    print("A total of " + str(col_num) + " deleted !")

    Df_dataset = Df_dataset.replace(np.nan, 0)
    new1 = Df_dataset[["Date_Added", "Detected_Prefecture"]]
    a = new1.groupby("Date_Added").size().values
    df1 = new1.drop_duplicates(subset="Date_Added").assign(Count=a)

    dfnew = df1.pivot_table('Count', ['Date_Added'], 'Detected_Prefecture')
    dfnew.fillna(0, inplace=True)

    return Df_dataset, new1, df1, dfnew



def convert_date(Df_dataset, countryIndex, date):
    float_date = pd.to_datetime(date).toordinal() #date that we select
    val = Df_dataset[countryIndex]
    date = []
    Total_cases = []

    for x in val.index.values:
        y = pd.to_datetime(x).toordinal()
        date.append(y)

    for x in val.values:
        if x < 0:
            Total_cases.append(0) #check whether the date is valid or not
        else:
            Total_cases.append(x)

    return date, Total_cases, [float_date]

def predict_cases_country(countryIndex, date):
    url = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data' \
          '/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv '
    df1_new = preprocessing(url)
    data = convert_date(df1_new, countryIndex, date)
    predicted_cases = predict_cases(data[0], data[1], data[2])
    return predicted_cases

def predict_cases_svm(dates, cases, predictDate):

    dates = np.reshape(dates, (len(dates), 1))  # convert to 1xn dimension
    predictDate = np.reshape(predictDate, (len(predictDate), 1))

    svr_lin = Pipeline([('scaler', StandardScaler()), ('svr', SVR(kernel='linear', C=1e3, cache_size=7000))])
    svr_poly = Pipeline([('scaler', StandardScaler()), ('svr', SVR(kernel='poly', C=1e3, cache_size=7000, degree=2))])
    svr_rbf = Pipeline([('scaler', StandardScaler()), ('svr', SVR(kernel='rbf', C=1e3, cache_size=7000, gamma=0.1))])

    # Fit regression model
    svr_lin.fit(dates, cases)
    svr_poly.fit(dates, cases)
    svr_rbf.fit(dates, cases)

    plt.scatter(dates, cases, c='k', label='Data')
    plt.plot(dates, svr_lin.predict(dates), c='g', label='Linear model')
    plt.plot(dates, svr_rbf.predict(dates), c='r', label='RBF model')
    plt.plot(dates, svr_poly.predict(dates), c='b', label='Polynomial model')

    plt.xlabel('Date')
    plt.ylabel('cases')
    plt.title('Support Vector Regression')
    plt.legend()
    plt.show()

    MeanErrorLin = 'Mean Squared Error Linear model: ' + str(mean_squared_error(cases, svr_lin.predict(dates)))
    MeanErrorRbf = 'Mean Squared Error RBF model: ' + str(mean_squared_error(cases, svr_rbf.predict(dates)))
    MeanErrorPoly = 'Mean Squared Error Polynomial model: ' + str(mean_squared_error(cases, svr_poly.predict(dates)))

    return svr_rbf.predict(predictDate)[0], svr_lin.predict(predictDate)[0], svr_poly.predict(predictDate)[
        0], MeanErrorLin, MeanErrorRbf, MeanErrorPoly


def predict_cases_city_svm(index, date):
    float_date = pd.to_datetime(date).toordinal()
    val = dfnew[index]
    date = []

    for x in val.index.values:
        y = pd.to_datetime(x).toordinal()
        date.append(y)

    Total_cases = val.values
    predicted_cases = predict_cases_svm(date, Total_cases, [float_date])
    return predicted_cases


def predict_cases_lin(dates, cases, predictDate):
    dates = np.reshape(dates, (len(dates), 1))  # convert to 1xn dimension
    predictDate = np.reshape(predictDate, (len(predictDate), 1))

    regression_model = LinearRegression()
    regression_model.fit(dates, cases)

    plt.scatter(dates, cases, c='k', label='Data')
    plt.plot(dates, regression_model.predict(dates), c='g', label='Linear model')

    plt.xlabel('Date')
    plt.ylabel('Cases')
    plt.title('Linear Regression')
    plt.legend()
    plt.show()

    MeanErrorLin = 'Mean Error Linear regression model: ' + str(
        mean_squared_error(cases, regression_model.predict(dates)))

    return regression_model.predict(predictDate)[0], MeanErrorLin


def predict_cases_city(index, date):
    float_date = pd.to_datetime(date).toordinal()
    val = dfnew[index]
    date = []

    for x in val.index.values:
        y = pd.to_datetime(x).toordinal()
        date.append(y)

    Total_cases = val.values
    predicted_cases = predict_cases_lin(date, Total_cases, [float_date])
    return predicted_cases


def predict_cases_poly(dates, cases, predictDate):
    dates = np.reshape(dates, (len(dates), 1))  # convert to 1xn dimension
    predictDate = np.reshape(predictDate, (len(predictDate), 1))

    # Fitting Polynomial Regression to the dataset
    polynominal_regg = PolynomialFeatures(degree=3)
    x_Polynom = polynominal_regg.fit_transform(dates)

    leniar_regg = LinearRegression()
    leniar_regg.fit(x_Polynom, cases)
    # Visualizing the Polymonial Regression results
    plt.scatter(dates, cases, c='k', label='Data')
    plt.plot(dates, leniar_regg.predict(x_Polynom), c='b', label='RBF model')
    plt.xlabel("Dates")
    plt.ylabel("Cases")
    plt.title('Polynomial Regression')
    return leniar_regg.predict(polynominal_regg.fit_transform(predictDate))


def predict_cases_city_poly(index, date):
    float_date = pd.to_datetime(date).toordinal()
    val = dfnew[index]
    date = []

    for x in val.index.values:
        y = pd.to_datetime(x).toordinal()
        date.append(y)

    Total_cases = val.values
    predicted_cases = predict_cases_poly(date, Total_cases, [float_date])
    return predicted_cases
