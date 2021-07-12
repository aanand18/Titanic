from django.shortcuts import render
from django.http import HttpResponse

import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score

def home(request):
    return render(request,'app1/home.html')

def predict(request):
    if(request.method == 'POST'):
        temp = {}
        temp['Pclass'] = int(request.POST.get('Pclass'))
        temp['Age'] = int(request.POST.get('Age'))
        temp['Sex'] = int(request.POST.get('Sex'))
        temp['SibSp'] = int(request.POST.get('SibSp'))
        if(temp['SibSp'] > 1):
            temp['SibSp'] = 1
        temp['Parch'] = int(request.POST.get('Parch'))
        if(int(request.POST.get('emb')) == 1):
            temp['S'] = int(1)
            temp['Q'] = int(0)
        elif(int(request.POST.get('emb')) == 0):
            temp['S'] = int(0)
            temp['Q'] = int(1)

    testData = pd.DataFrame({'x':temp}).transpose()
    data = pd.read_csv('./dataset/titanic.csv')
    data.drop(["PassengerId","Fare","Ticket","Name"],axis = 1,inplace = True)
    data.drop("Cabin", axis = 1, inplace = True)

    age_one = 0
    one_count = 0 
    age_zero = 0
    zero_count = 0
    heading = data.columns
    data = np.array(data).tolist()
    for i in range(len(data)):
        if str(data[i][3]) != "nan": 
            if str(data[i][0]) == "1":
                age_one += float(data[i][3])
                one_count +=1
            else: 
                age_zero += float(data[i][3])
                zero_count +=1
    avg_one = age_one/one_count
    avg_zero = age_zero/zero_count

    for i in range(len(data)):
        if str(data[i][3]) == "nan": 
            if str(data[i][0]) == "1":
                data[i][3] = avg_one
            else: 
                data[i][3] = avg_zero
    data = pd.DataFrame(data,columns = heading)

    data.dropna(inplace = True)

    heading = data.columns
    data = np.array(data).tolist()
    for i in range(len(data)):
        if str(data[i][2]) == "male":
            data[i][2] = 1
        else:
            data[i][2] = 0
    data = pd.DataFrame(data, columns = heading)

    emb = pd.get_dummies(data["Embarked"],drop_first = True)

    data = pd.concat([data,emb], axis =1)
    data.drop(["Embarked"], axis = 1, inplace = True)
        
    heading = data.columns
    data = np.array(data).tolist()
    for i in range(len(data)):
        if data[i][4]>0:
            data[i][4] = 1
    data = pd.DataFrame(data, columns = heading)

    heading = data.columns
    data = np.array(data).tolist()
    for i in range(len(data)):
        data[i][3] = int(data[i][3])
    data = pd.DataFrame(data, columns = heading)

    data = data.sort_index(axis = 1)
    Y = data["Survived"]
    X = data.drop("Survived", axis = 1)

    clf = LogisticRegression()
    clf.fit(X,Y)
    prediction = clf.predict(testData)[0]
    context = {'prediction' : prediction}

    return render(request, 'app1/home.html', context)
    
