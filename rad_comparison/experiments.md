# Radiomic and delta-radiomics comparison results so FAR

## Dataset
Classes are imbalanced for 'Récidive Méta' outcome (20 vs 3 patients). SMOTE was tested but did not seem to improve the results. Further investigation is needed using the different feature selection and prediction methods.

## Delta-radiomics computations
It seems that the best results were achieved using coefficient of varation method, not ratio. 

## Models trained on radiomic features
PCA was not used to select features because it is too time-consuming to tune. Near zero variance either because it produces null results. 

### Metrics
ROC-AUC was chosen for cross-validation. Investigation should be carried out to compare with another metric such as F1-score.

## Testing set 
The test set was defined keeping the same proportion of classes as the original dataset.

### Metrics 
Models were evaluated using ROC-AUC, sensitivity and specificity. The aim should be to maximize the sensitivity while keeping a good specificity ie having the best AUC possible.
The main comparisons will be carried out using the ROC-AUC metric. Spe and Sen will be only used in the end.

## Heatmaps
Interesting but too complex to interpret as we have a lot of different feature tables. 

## Boxplots
Idea: Compare the radiomic and delta-radiomic power. Need to reduce the number of models taken into account to make the results more readable.

## Printing 
Best models for each table were identified and printed. See analyze_results.ipynb for more details.

## Best models in term of specificity+senstivity
It seems that only few models are able to reach a good specificity while keeping a good sensitivity (>0.65). 

## Patient-based analysis
Some patients seem to be misclassified frequently for 'Récidive Locale' outcome. Not for 'Décès' outcome.
TODO: Further investigation is needed to understand who and why.

## Performances against number of features 
No trend was observed. But maybe it is because we distinguish the tables.
Joint analysis should be carried out to see if there is a trend. 

## Best feature selection model 
TODO: Analysis could be carried out to see if there is a trend in the best feature selection model.

## Best prediction model
TODO: Analysis could be carried out to see if there is a trend in the best prediction model.

# Radiomics from GIE acquisition
## Testing set 
It was defined randomly using the sklearn train_test_split function with a test size of 0.3. 

# 15/01/2025
## Best feature names
TODO : Add the best feature names for each table.