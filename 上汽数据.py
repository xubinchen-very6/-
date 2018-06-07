import pandas as pd
import numpy as np
import tensorflow as tf
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt

path = '新厂油漆车间数据1.csv'
worktime = pd.read_csv('16-17worktime.csv')
ori_data = pd.read_csv(path)
area = {'CT6': 125.1, 'XT5': 123.8, 'U358': 156.7}
keyword = ['day']


class DataProcess(object):

    def __init__(self, path, ori_data, area, keyword ,worktime):
        self.path = path
        self.ori_data = ori_data
        self.workTime = worktime
        self.area = area
        self.keyword = keyword

    def dataProcess(self):
        weekday = self.ori_data['date'].tolist()
        days = []
        for day in weekday:
            days.append(datetime.strptime(day, '%Y/%m/%d').weekday() + 1)
        days = pd.DataFrame(days)
        days.columns = ['day']
        data = self.ori_data.join(days)
        work = np.array(data['U358'].tolist())
        workday = np.where(work > 0, 1, 0)
        workday = pd.DataFrame(workday)
        workday.columns = ['workday']
        data = data.join(workday)
        data['CT6'] = data['CT6'] * self.area['CT6'] * 0.001
        data['XT5'] = data['XT5'] * self.area['XT5'] * 0.001
        data['U358'] = data['U358'] * self.area['U358'] * 0.001
        return data


    def processNan(self,data):
        new_date = []
        weekday = self.ori_data['date'].tolist()
        for month in weekday:
            new_date.append(int(month.split('/')[1]))
        month = pd.DataFrame(new_date)
        month.columns = ['month']
        data = data.join(month)
        nan = data[data.isnull().values == True].index.values
        nan = np.unique(nan)
        nanData = data.ix[nan]
        nanindex0 = nanData[nanData['workday'] == 0].index.values
        data.ix[nanindex0] = data.ix[nanindex0].fillna(0)
        return data

    def dummy(self, data):
        for word in self.keyword:
            dummies = pd.get_dummies(data[word], prefix=word)
            data = data.join(dummies)
            data = data.drop([word], axis=1)
        data['dvalue'] = data['Max_TEMP'] - data['Min_TEMP']
        return data

    def concatWorkTime(self,data):
        self.workTime.fillna(0, inplace=True)
        data['workTime'] = self.workTime['workTime']
        return data

    def modifyWorkDay(self,data):
        temp = data['workTime'].values
        temp = np.where(temp == 0, 0, 1)
        data['workday'] = temp
        return data

    def logFeature(self,data):
        logFeature = data[['CT6', 'XT5', 'U358']] + 0.0001
        poly = PolynomialFeatures(interaction_only=True)
        logFeature = poly.fit_transform(logFeature)
        logFeature = pd.DataFrame(logFeature)
        logFeature.drop([0, 1, 2, 3], axis=1, inplace=True)
        data = pd.concat([data, logFeature], axis=1)
        return data

def makeData(path,ori_data,area,keyword,worktime):
    dpModel = DataProcess(path,ori_data,area,keyword,worktime)
    data = dpModel.dataProcess()
    data = dpModel.processNan(data)
    data = dpModel.dummy(data)
    data = dpModel.concatWorkTime(data)
    data = dpModel.modifyWorkDay(data)
    data = dpModel.logFeature(data)
    date= data['date']
    return data,date
data,date = makeData(path,ori_data,area,keyword,worktime)
