import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.preprocessing import StandardScaler, PolynomialFeatures



def main():
    df = pd.read_csv('fortune500.csv')

    #multiple linear regression
    sns.pairplot(df)
    #Then we definne the variable that we are going to predic (rank) *marketcap
    y = df['rank']
    #We define the variables we use to make predictions # encode 'sector', 'industry'
    x = df[['employees', 'revenues', 'revchange', 'profits','prftchange','assets','totshequity']]
    #binary categories transformaiton (todo)

    #train test
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)

    #build model
    LinReg = LinearRegression()
    LinReg.fit(x_train, y_train)

    print(LinReg.coef_)
    coef = pd.DataFrame(LinReg.coef_, x.columns, columns = ['Coeff'])
    print(coef)
    #test - predictions
    predictions = LinReg.predict(x_test)
    plt.scatter(y_test, predictions)
    plt.hist(y_test - predictions)
    # plt.show()
    print(predictions)
    
    #performance
    mae = metrics.mean_absolute_error(y_test, predictions)
    mse = metrics.mean_squared_error(y_test, predictions)
    r2 =  metrics.r2_score(predictions, y_test)
    print(mae,mse,r2)

    ##regression polynomial
    scale = StandardScaler()
    x = scale.fit_transform(x)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)
    poly = PolynomialFeatures(degree = 7)
    x_poly_train = poly.fit_transform(x_train)

    model = LinearRegression()
    model.fit(x_poly_train, y_train)

    y_pred = model.predict(poly.fit_transform(x_test))
    print(model.coef_)
    r2 =  metrics.r2_score(y_pred, y_test)
    #0.7698104896162192
    print(r2)
    #response - y_pred - y_test
    # plt.scatter()
    print(y_pred)
    #response = model.intercept_ + 

if __name__=='__main__': 
    main()
