import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.model_selection import GridSearchCV


def get_metrics(confmat):
    '''Unravel confusion matrix and calculate performance metrics'''
    tn, fp, fn, tp = confmat.ravel()

    acc = (tp+tn)/(tn + fp + fn + tp)
    sen = tp/(tp+fn)
    esp = tn/(tn+fp)
    ppv = tp/(tp+fp)
    fsc = 2*(sen*ppv)/(sen+ppv)

    return acc, sen, esp, ppv, fsc


def print_performance_metrics(confmat_train, *confmat_test):
    '''Print performance metrics'''

    if not confmat_test:
        acc, sen, esp, ppv, fsc = get_metrics(confmat_train)
        print('ACC: %2.2f' % (100*acc))
        print('SEN: %2.2f' % (100*sen))
        print('SPE: %2.2f' % (100*esp))
        print('PPV: %2.2f' % (100*ppv))
        print('F1: %2.2f' % (100*fsc))
    else:
        acc_train, sen_train, esp_train, ppv_train, fsc_train = get_metrics(
            confmat_train)
        acc_test, sen_test, esp_test, ppv_test, fsc_test = get_metrics(
            confmat_test[0])

        print('PERFORMANCE METRICS')
        print('\tTRAIN\tTEST')
        print('ACC:\t%2.2f\t%2.2f' % (100*acc_train, 100*acc_test))
        print('SEN:\t%2.2f\t%2.2f' % (100*sen_train, 100*sen_test))
        print('SPE:\t%2.2f\t%2.2f' % (100*esp_train, 100*esp_test))
        print('PPV:\t%2.2f\t%2.2f' % (100*ppv_train, 100*ppv_test))
        print('F1:\t%2.2f\t%2.2f' % (100*fsc_train, 100*fsc_test))


def plot_roc_curve(y, y_prob):
    '''Plot ROC-AUC Curve and target probability'''

    ejex, ejey, _ = roc_curve(y, y_prob)
    roc_auc = auc(ejex, ejey)

    plt.figure(figsize=(12, 4))

    # ROC-AUC CURVE
    plt.subplot(1, 2, 1)
    plt.plot(ejex, ejey, color='darkorange',
             lw=2, label='AUC = %0.2f' % roc_auc)
    plt.plot([0, 1], [0, 1], color=(0.6, 0.6, 0.6), linestyle='--')
    plt.plot([0, 0, 1], [0, 1, 1], lw=2, linestyle=':',
             color='black', label='Perfect classifier')
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('FPR (1-ESP)')
    plt.ylabel('SEN')
    plt.legend(loc="lower right")

    # PROB DENSITY
    idx_0 = (y == 0)
    idx_1 = (y == 1)

    plt.subplot(1, 2, 2)
    plt.hist(y_prob[idx_0], density=1, bins=20, label='y=0', alpha=0.5)
    plt.hist(y_prob[idx_1], density=1, bins=20,
             facecolor='red', label='y=1', alpha=0.5)
    plt.legend()
    plt.xlabel('target probability')

    plt.show()


def plot_confusion_matrix(confmat_train, *confmat_test):
    ''' Plot confusion matrix
        - A single confusion matrix
        - Comparing two confusion matrices, if provided
    '''

    if not confmat_test:
        fig, ax = plt.subplots(figsize=(3, 3))
        ax.matshow(confmat_train, cmap=plt.cm.Blues, alpha=0.5)
        for i in range(confmat_train.shape[0]):
            for j in range(confmat_train.shape[1]):
                ax.text(
                    x=j, y=i, s=confmat_train[i, j], va='center', ha='center')

        plt.xlabel('predicted label')
        plt.ylabel('true label')

        plt.tight_layout()
        plt.show()

    else:
        fig, ax = plt.subplots(1, 2, figsize=(6, 6))
        ax[0].matshow(confmat_train, cmap=plt.cm.Blues, alpha=0.5)
        for i in range(confmat_train.shape[0]):
            for j in range(confmat_train.shape[1]):
                ax[0].text(x=j, y=i, s=confmat_train[i, j],
                           va='center', ha='center')

        ax[1].matshow(confmat_test[0], cmap=plt.cm.Blues, alpha=0.5)
        for i in range(confmat_test[0].shape[0]):
            for j in range(confmat_test[0].shape[1]):
                ax[1].text(x=j, y=i, s=confmat_test[0]
                           [i, j], va='center', ha='center')

        ax[0].set_xlabel('predicted label')
        ax[0].set_ylabel('true label')
        ax[0].set_title('TRAIN')

        ax[1].set_xlabel('predicted label')
        ax[1].set_ylabel('true label')
        ax[1].set_title('TEST')

        plt.tight_layout()
        plt.show()


def analyze_train_test_performance(clf, X_train, X_test, y_train, y_test, predict_proba=True):
    '''Analyze Train and Test Performance'''
    train_metrics = {}
    test_metrics = {}
    # get predictions
    y_pred_train = clf.predict(X_train)
    y_pred_test = clf.predict(X_test)

    # get confusion matrices
    confmat_train = confusion_matrix(y_train, y_pred_train)
    confmat_test = confusion_matrix(y_test, y_pred_test)

    # Plot confusion matrices and provide metrics
    print_performance_metrics(confmat_train, confmat_test)
    plot_confusion_matrix(confmat_train, confmat_test)

    try:
        # Plot ROC curve
        y_prob = clf.predict_proba(X_test)[:, 1]
        plot_roc_curve(y_test, y_prob)
    except AttributeError: 
        print("This method has not predict_proba method, ROC cannot be displayed. ")


def get_train_test_performances(clf, X_train, X_test, y_train, y_test):

    train_metrics = {}
    test_metrics = {}
    # get predictions
    y_pred_train = clf.predict(X_train)
    y_pred_test = clf.predict(X_test)

    # get confusion matrices
    confmat_train = confusion_matrix(y_train, y_pred_train)
    confmat_test = confusion_matrix(y_test, y_pred_test)

    acc_train, sen_train, esp_train, ppv_train, fsc_train = get_metrics(
        confmat_train)
    acc_test, sen_test, esp_test, ppv_test, fsc_test = get_metrics(
        confmat_test)

    train_metrics['acc'] = acc_train
    train_metrics['sen'] = sen_train
    train_metrics['esp'] = esp_train
    train_metrics['ppv'] = ppv_train
    train_metrics['fsc'] = fsc_train

    test_metrics['acc'] = acc_test
    test_metrics['sen'] = sen_test
    test_metrics['esp'] = esp_test
    test_metrics['ppv'] = ppv_test
    test_metrics['fsc'] = fsc_test

    return train_metrics, test_metrics


# These are customized functions: should be in utils.py
def hyper_parameters_search(clf, X, y, param_grid, scorer = 'f1', cv=5):
    
    grid = GridSearchCV(clf, param_grid = param_grid, scoring = scorer, cv = cv)
    grid.fit(X, y)

    # print("best mean cross-validation score: {:.3f}".format(grid.best_score_))
    # print("best parameters: {}".format(grid.best_params_))
    
    return grid

def plot_cv_scoring(grid, hyper_parameter, scorer = 'f1', plot_errors = False, log=False):
    
    scores = np.array(grid.cv_results_['mean_test_score'])
    std_scores = grid.cv_results_['std_test_score']
    params = grid.param_grid[hyper_parameter]
    
    if log:
        params = np.log10(params)
    if plot_errors:
        plt.errorbar(params,scores,yerr=std_scores, fmt='o-',ecolor='g')
    else:
        plt.plot(params,scores, 'o-')
    plt.xlabel(hyper_parameter,fontsize=14)
    plt.ylabel(scorer)
    plt.show()
    
# feature importance
def plot_importances(importances, feat_names):
    
    df_importances = pd.Series(importances, index=feat_names)
    
    plt.figure()
    df_importances.plot.bar()
    plt.ylabel("Feature Importance")
    plt.show()

def select_best_features(importances, feat_names, n_features):

    # Combine importances and feature names into a list of tuples
    feature_importance = list(zip(feat_names, importances))
    
    # Sort the list of tuples based on importances in descending order
    sorted_features = sorted(feature_importance, key=lambda x: x[1], reverse=True)
    
    # Select top n_features
    top_features = sorted_features[:n_features]
    
    # Display the results
    print("Top", n_features, "features:")
    for feature, importance in top_features:
        print(f"{feature}: {importance:.4f}")

    return top_features

