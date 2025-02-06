import analysis_functions as af
import json
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import numpy as np

folder_path = '/home/tachennf/Documents/delta-rad/extracted_radiomics/'
delta_rad_tables = ['rd_f1_f5_gtv.csv']
feat_sel_algo_list = ['ANOVA_PERC'] # , 'RDM_SEL', 'NO_SEL' ##" 'RF', 'ADABOOST', , 'MI_PERC', 'MI_K_BEST', 'NO_SEL', 'RDM_SEL', 'LASSO'
outcome_csv = 'outcomes.csv'
results_file = './json_results/results_ncv_test.json'
pred_algo_list = ['RF', 'ADABOOST', 'LOGREGRIDGE', 'PSVM', 'KNN',  'BAGG', 'MLP', 'QDA']
MAX_FEATURES = 3
outcomes_list = ['Récidive Locale'] # 'Récidive Méta', 
results_dict = json.load(open(results_file))

print(list(results_dict.keys()))
metric_list = ['roc_auc', 'sensitivity', 'specificity']
results = pd.read_json(results_file)

_ = af.get_best_results(results, delta_rad_tables, feat_sel_algo_list, pred_algo_list, 'Récidive Locale', metric='test_auc', k = 5)



