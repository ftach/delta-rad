# for each table in delta_rad_tables
    # X_train, X_val, y_train, y_val = get_dataset(delta_rad_csv_path)
    # for each feature_selection algorithm
        # best_features = get_best_features()
        # for i in range(1, 5) # number of features selected
            # select only i features from best_features 
            # for each prediction_algorithm
                # train with selected features in cross validation 
                # use best algo to make predictions 
                # compute roc auc # later compare the auc for each model using a boxplot (to show the distribution of auc for different number of selected features)
                # performance metrics (accuracy, sensitivity, specificity) with confidence intervals 
                # display patient for which prediction got wrong,
                # compute p value between models 

