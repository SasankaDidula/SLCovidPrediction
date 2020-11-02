import pickle
import csv
import os
import numpy as np
import pandas as pd
from geopy.geocoders import Nominatim
from opencage.geocoder import OpenCageGeocode
from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build


def createDatasetForSL():
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
                if len(row) == 10:
                    writer.writerow(
                        {fieldnames[0]: row[0], fieldnames[1]: row[1], fieldnames[2]: row[2], fieldnames[3]: row[3],
                         fieldnames[4]: row[4], fieldnames[5]: row[5], fieldnames[6]: row[6], fieldnames[7]: row[7]})
                if len(row) == 9:
                    writer.writerow(
                        {fieldnames[0]: row[0], fieldnames[1]: row[1], fieldnames[2]: row[2], fieldnames[3]: row[3],
                         fieldnames[4]: row[4], fieldnames[5]: row[5], fieldnames[6]: row[6], fieldnames[7]: row[7]})

def SLdatasetbyDate():
    SCOPES = ['https://www.googleapis.com/auth/spreadsheets.readonly']
    SAMPLE_SPREADSHEET_ID = '1zIgPU0ZlYkiKaavYAUcHKgEP95jdaMaf9ljJgRqtog4'
    SAMPLE_RANGE_NAME = 'Sum By Day!A2:F'

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
        with open('sumbyday.csv', 'w', newline='') as csvfile:
            fieldnames = ['Date', 'Confirmed', 'Recovered', 'Deceased', 'Critical', 'Tested']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for row in values:
                if len(row) >= 5:
                    writer.writerow(
                        {fieldnames[0]: row[0], fieldnames[1]: row[1], fieldnames[2]: row[2], fieldnames[3]: row[3],
                         fieldnames[4]: row[4], fieldnames[5]: row[5]})

def SLdatasetbyCity():
    SCOPES = ['https://www.googleapis.com/auth/spreadsheets.readonly']
    SAMPLE_SPREADSHEET_ID = '1zIgPU0ZlYkiKaavYAUcHKgEP95jdaMaf9ljJgRqtog4'
    SAMPLE_RANGE_NAME = 'Prefecture Data!A3:F'

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
        with open('sumbycity.csv', 'w', newline='') as csvfile:
            fieldnames = ['City', 'Confirmed', 'Recovered', 'Deaths']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for row in values:
                if len(row) >= 6:
                    writer.writerow(
                        {fieldnames[0]: row[0], fieldnames[1]: row[3], fieldnames[2]: row[4], fieldnames[3]: row[5]})

def SLDataPreprocess():
    col_num = 0
    Df_dataset = pd.read_csv('SLCovid.csv')
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

    Df_dataset = Df_dataset.replace(np.nan, 0)
    return Df_dataset

def geometry():
    geokey = '92244e2e0ea946d3ac6a1024ef5a74dc'
    data = []  # create empty lists

    geocoder = OpenCageGeocode(geokey)
    df = SLDataPreprocess()
    df.drop(['Number','Date_Added','Date_Announced','Status', 'Notes'], axis=1, inplace=True)

    newDataSet = df['Detected_Prefecture'].value_counts()
    print(newDataSet)
    for index, row in newDataSet.items():  # iterate over rows in dataframe
        lat = 0
        long = 0
        location = ''
        try:
            City = str(index)
            query = str(City) + ', Sri Lanka'
            geolocator = Nominatim(user_agent="slcovidprediction")
            location = geolocator.geocode(query)

            if len(location.raw) > 0:
                lat = location.latitude
                long = location.longitude

                data.append({'City': index, 'Cases': row, 'lat': lat, 'long': long})
        except AttributeError:
            continue
    df4 = pd.DataFrame(data)
    print('geomatry csv created!!')
    df4.to_csv('geometry.csv')
