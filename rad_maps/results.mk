# Radiomic feature maps statistical analysis results

## 12/01/2025
Clustered feature map using k-means clustering. K was set to 4 to have background low, medium and high expression. But background is annoying for proper clustering. The range of the valus used for clustering
is not ideal. 
It is possible to plot the histogram of the clustered maps. 
It is also possible to compute other intensity parameters. Ex: mean, std, cv, skewness, kurtosis, min, max.
Finally, we can plot a delta-radiomic feature map.

## 14/01/2025
Certains paramètres d'intensité sont plus significatifs que d'autres pour prédire un outcome. 

### F1, Récidive locale, Maximum
Significant difference between groups for mean in Récidive Locale patients for original_firstorder_Maximum in ttt_1 fraction. P-value:  0.01523967334277116
Significant difference between groups for cv in Récidive Locale patients for original_firstorder_Maximum in ttt_1 fraction. P-value:  0.004424113982089846
Significant difference between groups for skewness in Récidive Locale patients for original_firstorder_Maximum in ttt_1 fraction. P-value:  0.004828608766055622
Significant difference between groups for kurtosis in Récidive Locale patients for original_firstorder_Maximum in ttt_1 fraction. P-value:  0.019873465528917274 

### F3, Décès, Kurtosis
No significant differences

### Delta-radiomics maps
Significant difference between groups for std in Décès patients for original_firstorder_Skewness in ttt_1_ttt_3 fraction. P-value:  0.02394217584052614
Significant difference between groups for min in Décès patients for original_firstorder_Skewness in ttt_1_ttt_3 fraction. P-value:  0.013257175726916481

