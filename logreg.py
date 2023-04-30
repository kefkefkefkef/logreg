import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import streamlit as st
import seaborn as sns
from matplotlib import pyplot as plt
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
        
        N=st.number_input('Количество эпох', min_value=1000, max_value=8001, value=2000, step=200)
        for i in range(N):
            yhat = 1/(1 + np.exp(-(self.bias + X@self.w)))
            error = (y - yhat)
            w0_grad = - error 
            w_grad = - X * error.reshape(-1, 1)
            self.w -= self.learning_rate * w_grad.mean(axis=0) 
            self.bias -= self.learning_rate * w0_grad.mean()
        
            

    def predict(self, X):
            
            #X['y_pred'] = round(1/(1 + np.exp(-(self.bias + X@self.w))))
            sigmoid = 1/(1 + np.exp(-(self.bias + X@self.w)))
            return np.array(sigmoid), np.array(list(map(int,round(sigmoid))))

st.write('''
        ### Логистическая регрессия
        #### Предскажем ваши данные без регистрации и смс
        ''')


#train = st.file_uploader('Загрузите свои данные:', 'csv')
input_file = st.file_uploader("Загрузите данные для обучения модели",type=['csv'])
input_file2 = st.file_uploader("Загрузите данные для предсказания",type=['csv'])
if (input_file is not None) and input_file.name.endswith(".csv"):
    train = pd.read_csv(input_file).drop('Unnamed: 0', axis=1)
   
    y = st.selectbox('Выберите таргет:',(train.columns))
    xs = st.multiselect('Выберите показатели для вычисления весов:', (train.columns))
   
    #if st.button("Поехали"):
    ss = StandardScaler()
    train_new = train
    train_new[xs] = ss.fit_transform(train_new[xs])
    learning_rate = st.number_input('Точность обучения', min_value=0.001, max_value=0.0155, value=0.005, step=0.001, format="%f")
    logreg = LogReg(learning_rate)
    logreg.fit(train_new[xs], train_new[y].to_numpy())     
    
    
    st.write('Веса модели:', logreg.w, 'Свободный член:', f'{logreg.bias}')   
    train['y_sigm'],train['y^'] = logreg.predict(train[xs])
    
    
    
    
    precision = train.loc[(train['y'] == train['y^'])].shape[0]/ train.shape[0]*100
    st.write(f'Точность предсказания: {precision}%')
    #st.write('Сверим предсказание модели с входными данными:')
    def compare(s):
        return ['background-color: #90EE90']*len(s) if s['y'] == s['y^'] else ['background-color: #FFCCCB']*len(s)

    #st.dataframe(train.style.apply(compare, axis=1))
   

    #input_file2 = st.file_uploader("Загрузите данные для предсказания",type=['csv'])
    if (input_file2 is not None) and input_file2.name.endswith(".csv"):
        test = pd.read_csv(input_file2).drop('Unnamed: 0', axis=1)
        test_new = test        
        test_new[xs] = ss.fit_transform(test_new[xs])
        test['y_sigm'],test['y^'] = logreg.predict(test_new[xs])
        st.write('''
        #### Итоговый результат
        ''')
        st.dataframe(test.style.apply(compare, axis=1))
        precision_test = test.loc[(test['y'] == test['y^'])].shape[0]/ test.shape[0]*100
        #st.write(f'Точность предсказания: {precision}%')
        fig, ax = plt.subplots()
        sns.scatterplot(data = test.sort_values('y_sigm').reset_index()[['y','y^']])
        sns.lineplot(data = test.sort_values('y_sigm').reset_index()['y_sigm'], label='Предсказание модели')
        #plt.axhline(0.5, color='r', label='0.5')
        #plt.axvline(test.loc[(test['y^'] == 0)].shape[0])
        st.pyplot(fig)
        error = test.loc[(test['y'] != test['y^'])].shape[0]#/ test.shape[0]*100
        st.write('Значений предсказано неверно', error)





