import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

def utils():
    all_dataset = pd.read_csv('../fm/data/adult_data.csv')
    print('features num initially : ',len(all_dataset.columns))


    '''
    ['age', 'workclass', 'fnlwgt', 'education', 'education_num',
           'marital_status', 'occupation', 'relationship', 'race', 'gender',
           'capital_gain', 'capital_loss', 'hours_per_week', 'native_country',
           'income_bracket', 'income_label']
    '''

    label_dataset = all_dataset['income_label']

    all_dataset.drop(['fnlwgt','education_num','capital_gain','capital_loss','income_bracket','income_label'],axis = 1,inplace=True)
    print('features num after drop : ',len(all_dataset.columns))

    print(all_dataset.shape)


    age_list = (np.arange(11) * 10).tolist()
    temp_list = ((np.arange(10)+1) * 10).tolist()
    label_list = []

    for i,j in zip(age_list,temp_list):
        label_list.append(str(i) + '-' + str(j))

    all_dataset['age'] = pd.cut(all_dataset['age'],age_list,labels = label_list)
    all_dataset['hours_per_week'] = pd.cut(all_dataset['hours_per_week'],age_list,labels = label_list)

    normalize_col_name = ['age','hours_per_week']

    '''
    for col in normalize_col_name:
        all_dataset[col] = preprocessing.scale(all_dataset[col])
    '''

    cols_name = all_dataset.columns.values.tolist()
    dummy_col_name = list(set(cols_name) - set(normalize_col_name))

    all_data_arr = pd.get_dummies(all_dataset)

    print(all_data_arr.shape[1])

    seed = 2019
    X_train , X_test = train_test_split(all_data_arr.values,test_size=0.3,random_state=seed)
    Y_train , Y_test = train_test_split(np.array(label_dataset),test_size=0.3 , random_state=seed)

    features_num = []
    for col in cols_name:
        features_num.append(len(all_dataset[col].unique()))

    print(sum(features_num))

    return X_train,Y_train,X_test,Y_test,features_num