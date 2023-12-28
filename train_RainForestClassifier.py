import pickle

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from  sklearn.model_selection import  train_test_split
from sklearn.metrics import accuracy_score

def convert_list_string_to_list_num(l: list):
    data_float = []
    for item in l:
        item = item.strip('[]')

        float_list = [float(x.strip()) for x in item.split(',')]

        data_float.append(float_list)

    data_float = np.array(data_float)
    return  data_float


df = pd.read_csv('data.csv')
data_ = np.asarray(df['data'])
labels = np.asarray(df['labels'])

data = convert_list_string_to_list_num(data_)

x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size= 0.2, shuffle= True, stratify= labels)

model = RandomForestClassifier()

model.fit(x_train,y_train)

y_predict = model.predict(x_test)

score = accuracy_score(y_predict,y_test)

print('Do chinh xac: {}% !'.format(score * 100))

f = open('model.pickle','wb')
pickle.dump({'model' : model},f)
f.close()

# modelcsv = pd.DataFrame()
# modelcsv['model'] = model
# modelcsv.to_csv('model.csv')