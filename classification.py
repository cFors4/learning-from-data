import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split, ShuffleSplit, GridSearchCV
from sklearn.preprocessing import OrdinalEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report,confusion_matrix
import warnings 
warnings.filterwarnings("ignore")

def loadAndClean():
    #read in data to dataframe
    df = pd.read_csv('2014_Financial_Data.csv',index_col=0)

    # Drop rows with no information
    df.dropna(how='all', inplace=True)

    # Drop columns relative to classification, we will use them later
    class_data = df.loc[:, ['Class']]
    df.drop(['Class', '2015 PRICE VAR [%]'], inplace=True, axis=1)

    # Plot initial status of data quality in terms of nan-values and zero-values
    nan_vals = df.isna().sum()
    zero_vals = df.isin([0]).sum()
    ind = np.arange(df.shape[1])

    plt.figure(figsize=(50,10))

    plt.subplot(2,1,1)
    plt.title('INITIAL INFORMATION ABOUT DATASET', fontsize=22)
    plt.bar(ind, nan_vals.values.tolist())
    plt.ylabel('NAN-VALUES COUNT', fontsize=18)

    plt.subplot(2,1,2)
    plt.bar(ind, zero_vals.values.tolist())
    plt.ylabel('ZERO-VALUES COUNT', fontsize=18)
    plt.xticks(ind, nan_vals.index.values, rotation='90')

    # Find count and percent of nan-values, zero-values
    total_nans = df.isnull().sum().sort_values(ascending=False)
    percent_nans = (df.isnull().sum()/df.isnull().count() * 100).sort_values(ascending=False)
    total_zeros = df.isin([0]).sum().sort_values(ascending=False)
    percent_zeros = (df.isin([0]).sum()/df.isin([0]).count() * 100).sort_values(ascending=False)
    df_nans = pd.concat([total_nans, percent_nans], axis=1, keys=['Total NaN', 'Percent NaN'])
    df_zeros = pd.concat([total_zeros, percent_zeros], axis=1, keys=['Total Zeros', 'Percent Zeros'])

    # Find reasonable threshold for nan-values situation
    test_nan_level = 0.5
    # print(df_nans.quantile(test_nan_level))
    _, thresh_nan = df_nans.quantile(test_nan_level)

    # Find reasonable threshold for zero-values situation
    test_zeros_level = 0.6
    # print(df_zeros.quantile(test_zeros_level))
    _, thresh_zeros = df_zeros.quantile(test_zeros_level)
    # Clean dataset applying thresholds for both zero values, nan-values
    # print(f'INITIAL NUMBER OF VARIABLES: {df.shape[1]}')
    # print()

    df_test1 = df.drop((df_nans[df_nans['Percent NaN'] > thresh_nan]).index, 1)
    # print(f'NUMBER OF VARIABLES AFTER NaN THRESHOLD {thresh_nan:.2f}%: {df_test1.shape[1]}')
    # print()

    df_zeros_postnan = df_zeros.drop((df_nans[df_nans['Percent NaN'] > thresh_nan]).index, axis=0)
    df_test2 = df_test1.drop((df_zeros_postnan[df_zeros_postnan['Percent Zeros'] > thresh_zeros]).index, 1)
    # print(f'NUMBER OF VARIABLES AFTER Zeros THRESHOLD {thresh_zeros:.2f}%: {df_test2.shape[1]}')

    # Replace nan-values with mean value of column, considering each sector individually.
    df_test2 = df_test2.groupby(['Sector']).transform(lambda x: x.fillna(x.mean()))
    # Add the sector column
    # encode sector to int
    ord_enc = OrdinalEncoder()
    df["sector_code"] = ord_enc.fit_transform(df[["Sector"]])
    df_out = df_test2.join(df['sector_code'])

    # Add back the classification columns
    # df_out = df_out.join(class_data)    

    # Print information about dataset
    # df_out.info()
    # print(df_out.describe(include = 'all'))
    print(df_out.shape)
    print(class_data.shape)

    return df_out,class_data

def MultiLayerPerceptron(x,y):
    # remember to scale inputs between 0 to 1 ************
    x_train, x_test, y_train, y_test = train_test_split(x, y,test_size=0.2)
    mlp = MLPClassifier(max_iter=5000, activation='relu')
    # print(mlp)
    mlp.fit(x_train,y_train.values.ravel())
    y_predict_mlp = mlp.predict(x_test)
    print(confusion_matrix(y_test,y_predict_mlp))
    print()
    print(classification_report(y_test,y_predict_mlp))
    return 0

def TunedMLP(x,y):
    cv = ShuffleSplit(n_splits=1, test_size=0.2, random_state=1)

    x_train, x_test, y_train, y_test = train_test_split(x, y,test_size=0.2)
    # param_grid = {
    # 'hidden_layer_sizes':[5],
    # 'activation': ['identity','logistic','tanh','relu'],
    # 'solver': ['lbfgs','sgd','adam'],
    # 'learning_rate':['constant','invscaling','adaptive']}

    # gridSearch = GridSearchCV(MLPClassifier(), param_grid, cv=cv,
    #                       scoring=['recall','f1'],refit='f1',verbose=2)

    # gridSearch.fit(x_train, y_train.values.ravel())
    # print('Score: ', gridSearch.best_score_)
    # print('Parameters: ', gridSearch.best_params_)

    # 0.62 {'activation': 'relu', 'hidden_layer_sizes': 5, 'learning_rate': 'invscaling', 'solver': 'lbfgs'}

    # param_grid = {
    # 'hidden_layer_sizes':[5,10,15,(5,5),(5,10)],
    # 'activation': ['relu'],
    # 'solver': ['lbfgs'],
    # 'learning_rate':['invscaling']
    # }

    # gridSearch = GridSearchCV(MLPClassifier(), param_grid, cv=cv,
    #                       scoring=['recall','f1'],refit='f1',verbose=2)

    # gridSearch.fit(x_train, y_train.values.ravel())
    # print('Score: ', gridSearch.best_score_)
    # print('Parameters: ', gridSearch.best_params_)

    # Score:  0.6213364595545136 parameters:  {'activation': 'logistic', 'hidden_layer_sizes': (5,5), 'learning_rate': 'invscaling', 'solver': 'sgd'}

    # param_grid = {
    # 'hidden_layer_sizes':[5,10,15,(5,5),(5,10)],
    # 'activation': ['logistic'],
    # 'solver': ['sgd'],
    # 'alpha': [0,0.0005,0.0001,0.0005,0.001],
    # 'learning_rate':['invscaling'],
    # 'learning_rate_init': [0,0.0001,0.0005,0.001, 0.005,0.01],
    # 'max_iter': [500,1000,5000],
    # 'shuffle': [True,False] 
    # }

    # gridSearch = GridSearchCV(MLPClassifier(), param_grid, cv=cv,
    #                         scoring='recall',verbose=2)
    # gridSearch.fit(x_train, y_train.values.ravel())

    # print('Score: ', gridSearch.best_score_)
    # print('Parameters: ', gridSearch.best_params_)
    # mlp = MLPClassifier(activation= 'logistic')
    #Score:  1.0 Parameters:  {'activation': 'logistic', 'alpha': 0, 'hidden_layer_sizes': 5, 'learning_rate': 'invscaling', 'learning_rate_init': 0.0001, 'max_iter': 200, 'shuffle': False, 'solver': 'sgd'}
#  Score:  1.0
# Parameters:  {'activation': 'logistic', 'alpha': 0, 'hidden_layer_sizes': 5, 'learning_rate': 'invscaling', 'learning_rate_init': 0.0001, 'max_iter': 500, 'shuffle': False, 'solver': 'sgd'}

    X_train_s, X_tune, y_train_s, y_tune = train_test_split(x_train, y_train,test_size=0.2)
    # for i in range(15):
    #     mlp = MLPClassifier(hidden_layer_sizes=5,activation='logistic',learning_rate_init=0.0001,
    #                     learning_rate='invscaling', solver='sgd',alpha = 0, max_iter = 1000, shuffle = False,
    #                     random_state=i)
    
    #     mlp.fit(X_train_s, y_train_s.values.ravel())
    #     y_predict_mlp = mlp.predict(X_tune)
    #     print(classification_report(y_tune,y_predict_mlp))
    #     # i = 10 or 14
        
    mlp = MLPClassifier(hidden_layer_sizes=5,activation='logistic',learning_rate_init=0.0001,
                        learning_rate='invscaling', solver='sgd', max_iter = 5000, shuffle = False,
                        random_state=14)
    
        
    mlp.fit(x_train, y_train.values.ravel())
    y_predict_mlp = mlp.predict(x_test)
    print(classification_report(y_test,y_predict_mlp))
    return 0

def main():
    x,y = loadAndClean()
    out = MultiLayerPerceptron(x,y)
    out = TunedMLP(x,y)



if __name__=='__main__': 
    main()
