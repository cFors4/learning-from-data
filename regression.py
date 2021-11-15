import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.preprocessing import OrdinalEncoder


def main():
    #read in data to dataframe
    df = pd.read_csv('fortune500.csv')

    #preprocessing - encode industry and sector
    ord_enc = OrdinalEncoder()
    df["industry_code"] = ord_enc.fit_transform(df[["industry"]])
    df["sector_code"] = ord_enc.fit_transform(df[["sector"]])

    #Then we definne the variable that we are going to predic (rank) *revenues defining where in the the Fortune 500
    y = df['revenues']
    #We define the variables we use to make predictions # encode 'sector', 'industry'
    x = df[['employees', 'revchange', 'profits','prftchange','assets','totshequity','industry_code','sector_code']]

    #small dataset - get average of 10 runs
    NumRuns = 100
    linr2Sum,linMSESum = 0,0
    polyr2Sum,polyMSESum = 0,0
    for i in range(NumRuns):
        linr2,linMSE = LinearRegressionPrediction(x,y)
        linr2Sum += linr2
        linMSESum += linMSE
        polyr2,polyMSE = PolynomialRegressionPrediction(x,y,2)
        polyr2Sum += polyr2
        polyMSESum += polyMSE

    print("\nMultiple linear regression")
    print("r^2: ")
    print(linr2Sum/NumRuns)
    print("Mean squared Error: ")
    print(linMSESum/NumRuns)

    print("\nPolynomial regression")
    print("r^2: ")
    print(polyr2Sum/NumRuns)
    print("Mean squared Error: ")
    print(polyMSESum/NumRuns)


def LinearRegressionPrediction(x,y):
    #train test
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)

    #build model
    LinReg = LinearRegression()
    LinReg.fit(x_train, y_train)

    # Coefficients
    coef = pd.DataFrame(LinReg.coef_, x.columns, columns = ['Coeff'])
    # print(coef)
    #test - predictions
    predictions = abs(LinReg.predict(x_test))
    
    #performance
    #variance
    r2 =  metrics.r2_score(y_test, predictions)
    #bias
    mse = metrics.mean_squared_error(y_test, predictions)
    return r2,mse

def PolynomialRegressionPrediction(x,y,polDegree):
    ##regression polynomial with scalar
    
    scale = StandardScaler()
    x = scale.fit_transform(x)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)
    poly = PolynomialFeatures(degree = polDegree)
    x_poly_train = poly.fit_transform(x_train)

    model = LinearRegression()
    model.fit(x_poly_train, y_train)

    y_pred = model.predict(poly.fit_transform(x_test))

    #variance
    r2 =  metrics.r2_score(y_test, y_pred)
    #bias
    mse = metrics.mean_squared_error(y_test, y_pred)
    return r2,mse

if __name__=='__main__': 
    main()
