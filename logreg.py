import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import streamlit as st

train = pd.read_csv('aux/LogRegtrain.csv').drop('Unnamed: 0', axis=1)

ss = StandardScaler()
train[['x1', 'x2', 'x3']] = ss.fit_transform(train[['x1', 'x2', 'x3']])

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




#df = st.file_uploader('Загрузите свои данные:', 'csv')
input_file = st.file_uploader("Загрузите свои данные для обучения модели",type=['csv'])
if (input_file is not None) and input_file.name.endswith(".csv"):
    df = pd.read_csv(input_file).drop('Unnamed: 0', axis=1)
   
    y = st.selectbox('Выберите таргет:',(df.columns))
    xs = st.multiselect('Выберите показатели для вычисления весов:', (df.columns))
   
    if st.button("Поехали"):
     df[xs] = ss.fit_transform(df[xs])
     logreg = LogReg(0.01)
     logreg.fit(df[xs], df[y].to_numpy())     
     #st.write('You selected:', option)
     
     st.write('Веса модели:', logreg.w, 'Свободный член:', logreg.bias)   
     prediction = logreg.predict(df[xs])
    
     
     st.write('Сверим предсказание модели с входными данными:')
     compare_df = pd.DataFrame(data={'y': df[y], 'y^': prediction})

     def compare(s):
        return ['background-color: celadon']*len(s) if s['y'] == s['y^'] else ['background-color: #FFCCCB']*len(s)

     st.dataframe(compare_df.style.apply(compare, axis=1))
     
     #st.write(compare_df)
     #st.success(f'Your prediction is: {prediction}')


