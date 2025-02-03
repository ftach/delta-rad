'''Main script to run the feature selection and prediction algorithms comparison on the dataset. '''

import pandas as pd 
import json 
import os 
import time
import numpy as np 
import random 

import feature_selection as fsa 
import dataset 
import predictions as p 

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score



np.random.seed(42)
random.seed(42)

def main(): 
    start_time = time.time()

    ###################### INITIALIZATION ##############################
    folder_path = '/home/tachennf/Documents/delta-rad/extracted_radiomics/'
    delta_rad_tables = ['rd_f1_f5_gtv.csv']
    feat_sel_algo_list = ['RF', 'ADABOOST', 'ANOVA_PERC', 'ANOVA_K_BEST', 'CHI2_PERC', 'CHI2_K_BEST', 'MI_PERC', 'MI_K_BEST', 'NO_SEL', 'RDM_SEL', 'LASSO']
    outcome_csv = 'outcomes.csv'
    results_file = 'results_ncv_test.json'
    dset_selection_method = 'random' 
    pred_algo_list = ['DT', 'RF', 'ADABOOST', 'PSVM', 'KNN', 'LOGREG', 'LOGREGRIDGE', 'BAGG', 'MLP', 'LDA', 'QDA', 'NaiveB'] # 'LSVM', 
    MAX_FEATURES = 3
    outcomes_list = ['Récidive Locale'] # 'Récidive Méta', 

    for table in delta_rad_tables: 
        for outcome in outcomes_list: 
            # Load the dataset 
            X = pd.read_csv(os.path.join(folder_path, table)) # header=None, names=feature_list
            X = X.dropna() # delete nan values 
            # drop first X column 
            X = X.drop(X.columns[0], axis=1)

            correlation_matrix = X.corr(method='pearson') 
            X = dataset.remove_highly_corr_features(dataset.get_highly_corr_features(correlation_matrix), X) #  drop features whom collinearity > 0.9 

            outcome_df = pd.read_csv(os.path.join(folder_path, outcome_csv))
            outcome_df = outcome_df.drop(outcome_df.columns[0], axis=1)
            y = outcome_df.loc[outcome_df.index.isin(X.index)] 

            y = y.loc[:, [outcome]] # get the outcome column

            # Init classifiers 
            clf1 = LogisticRegression(multi_class='multinomial',
                                  solver='newton-cg',
                                  random_state=42)
            clf2 = KNeighborsClassifier(algorithm='ball_tree',
                                        leaf_size=50)
            clf3 = DecisionTreeClassifier(random_state=42)
            clf4 = SVC(random_state=42)

            # Setting up the parameter grids 
            param_grid1 = [{'clf1__penalty': ['l2'],
                        'clf1__C': np.power(10., np.arange(-4, 4))}]

            param_grid2 = [{'clf2__n_neighbors': list(range(1, 10)),
                            'clf2__p': [1, 2]}]

            param_grid3 = [{'max_depth': list(range(1, 10)) + [None],
                            'criterion': ['gini', 'entropy']}]

            param_grid4 = [{'clf4__kernel': ['rbf'],
                            'clf4__C': np.power(10., np.arange(-4, 4)),
                            'clf4__gamma': np.power(10., np.arange(-5, 0))},
                           {'clf4__kernel': ['linear'],
                            'clf4__C': np.power(10., np.arange(-4, 4))}]

            # Setting up the pipelines
            pipe1 = Pipeline([('std', StandardScaler()),
                          ('clf1', clf1)])

            pipe2 = Pipeline([('std', StandardScaler()),
                              ('clf2', clf2)])

            pipe4 = Pipeline([('std', StandardScaler()),
                              ('clf4', clf4)])

            # Setting up multi GridSearchCV for each classifier
            gridcvs = {}

            for pgrid, est, name in zip((param_grid1, param_grid2,
                                         param_grid3, param_grid4),
                                        (pipe1, pipe2, clf3, pipe4),
                                        ('Softmax', 'KNN', 'DTree', 'SVM')):
                gcv = GridSearchCV(estimator=est,
                                   param_grid=pgrid,
                                   scoring='accuracy',
                                   n_jobs=1,
                                   cv=4,
                                   verbose=0,
                                   refit=True)
                gridcvs[name] = gcv

                cv_scores = {name: [] for name, gs_est in gridcvs.items()}

            skfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

            # The outer loop for algorithm selection
            for c, (outer_train_idx, outer_valid_idx) in enumerate(skfold.split(X, y)):
                for name, gs_est in sorted(gridcvs.items()):
                    print('outer fold %d/5 | tuning %-8s' % (c, name), end='')
                    # The inner loop for hyperparameter tuning
                    gs_est.fit(X.iloc[outer_train_idx], y.iloc[outer_train_idx])
                    y_pred = gs_est.predict(X.iloc[outer_valid_idx])
                    acc = accuracy_score(y_true=y.iloc[outer_valid_idx], y_pred=y_pred)
                    print(' | inner ACC %.2f%% | outer ACC %.2f%%' %
                          (gs_est.best_score_ * 100, acc * 100))
                    cv_scores[name].append(acc)

            # Looking at the results
            for name in cv_scores:
                print('%-8s | outer CV acc. %.2f%% +\- %.3f' % (
                      name, 100 * np.mean(cv_scores[name]), 100 * np.std(cv_scores[name])))
            print('\nSVM Best parameters', gridcvs['SVM'].best_params_)

            # Fitting a model to the whole training set
            # using the "best" algorithm
            best_algo = gridcvs['DTree']

            best_algo.fit(X.iloc[outer_train_idx], y.iloc[outer_train_idx])
            train_acc = accuracy_score(y_true=y.iloc[outer_train_idx], y_pred=best_algo.predict(X.iloc[outer_train_idx]))
            test_acc = accuracy_score(y_true=y.iloc[outer_valid_idx], y_pred=best_algo.predict(X.iloc[outer_valid_idx]))

            print('Accuracy %.2f%% (average over CV test folds)' %
                  (100 * best_algo.best_score_))
            print('Best Parameters: %s' % gridcvs['SVM'].best_params_)
            print('Training Accuracy: %.2f%%' % (100 * train_acc))
            print('Test Accuracy: %.2f%%' % (100 * test_acc))

if __name__ == "__main__": 
    main()