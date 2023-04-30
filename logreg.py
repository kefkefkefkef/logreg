import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import streamlit as st

# train0 = pd.read_csv('aux/LogRegtrain.csv').drop('Unnamed: 0', axis=1)

# ss = StandardScaler()
# train0[['x1', 'x2', 'x3']] = ss.fit_transform(train0[['x1', 'x2', 'x3']])

class LogReg:
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate
        
        
        
    def fit(self, X, y):
        X = np.array(X)
        self.w = np.random.normal(size=X.shape[1])
        self.bias = np.random.normal(1)
        
        N=10000
        for i in range(N):
            yhat = 1/(1 + np.exp(-(self.bias + X@self.w)))
            error = (y - yhat)
            w0_grad = - error 
            w_grad = - X * error.reshape(-1, 1)
            self.w -= self.learning_rate * w_grad.mean(axis=0) 
            self.bias -= self.learning_rate * w0_grad.mean()
        
            

    def predict(self, X):
            
            #X['y_pred'] = round(1/(1 + np.exp(-(self.bias + X@self.w))))
            return np.array(list(map(int,round(1/(1 + np.exp(-(self.bias + X@self.w)))))))




#train = st.file_uploader('Загрузите свои данные:', 'csv')
input_file = st.file_uploader("Загрузите свои данные для обучения модели",type=['csv'])
if (input_file is not None) and input_file.name.endswith(".csv"):
    train = pd.read_csv(input_file).drop('Unnamed: 0', axis=1)
   
    y = st.selectbox('Выберите таргет:',(train.columns))
    xs = st.multiselect('Выберите показатели для вычисления весов:', (train.columns))
   
    if st.button("Поехали"):
     ss = StandardScaler()
     train[xs] = ss.fit_transform(train[xs])
     logreg = LogReg(0.01)
     logreg.fit(train[xs], train[y].to_numpy())     
     #st.write('You selected:', option)
     
     st.write('Веса модели:', logreg.w, 'Свободный член:', f'{logreg.bias}')   
     prediction = logreg.predict(train[xs])
    
     
     st.write('Сверим предсказание модели с входными данными:')
     compare_train = pd.DataFrame(data={'y': train[y], 'y^': prediction})

     def compare(s):
        return ['background-color: #90EE90']*len(s) if s['y'] == s['y^'] else ['background-color: #FFCCCB']*len(s)

     st.dataframe(compare_train.style.apply(compare, axis=1))
     precision = compare_train.loc[(compare_train['y'] == compare_train['y^'])].shape[0]/ compare_train.shape[0]*100
     st.write(f'Точность предсказания: {precision}%')

     input_file2 = st.file_uploader("Загрузите данные для предсказания",type=['csv'])
     if (input_file2 is not None) and input_file2.name.endswith(".csv"):
        test = pd.read_csv(input_file2).drop('Unnamed: 0', axis=1)
        test[xs] = ss.fit_transform(test[xs])
        test['y^'] = logreg.predict(test[xs])

        st.dataframe(test.style.apply(compare, axis=1))
        precision_test = test.loc[(test['y'] == test['y^'])].shape[0]/ test.shape[0]*100
        st.write(f'Точность предсказания: {precision}%')

        





