import numpy as np
import requests

def getData(url):
    response = requests.get(url)
    rData = response.text.strip() #get the row data
    n = 0
    point1 = []
    point2 = []
    flag = 0
    for i in range(len(rData)):   #get the comments
        if(rData[i] == '#' and flag == 0):
            point1.append(i)
            flag = 1-flag
        if(rData[i] == '\n' and flag == 1):
            point2.append(i)
            flag = 1-flag
    dataSp = rData.split('\n')[len(point1):] #remove comments
    dataSpRow = dataSp[0].strip().split(' ') #set a observer as a rowlist
    metricLen = len(dataSp)
    metricWid = len(dataSpRow)-1
    x = np.zeros((metricLen,metricWid))
    y = np.zeros((metricLen,1))
    for i in range(metricLen):
        dataSpRow = dataSp[i].strip().split(' ')
        for j in range(metricWid):
            x[i][j] = float(dataSpRow[j])
            y[i][0] = float(dataSpRow[-1])
    return x,y