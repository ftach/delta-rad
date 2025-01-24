import analysis_functions as af
import json
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import numpy as np

results_dict = json.load(open('json_results/result_random_run2.json'))
# delta_rad_tables = ['f1_gtv.csv', 'f2_gtv.csv', 'f3_gtv.csv', 'f4_gtv.csv', 'f5_gtv.csv', 'rd_f1_f5_gtv.csv', 'rd_f1_f2_gtv.csv', 'rd_f1_f3_gtv.csv', 'rd_f2_f3_gtv.csv', 'rd_f1_f4_gtv.csv']
# delta_rad_tables = ['gie_1month_gtv.csv', 'simu_gie_gtv.csv', 'simu_onemth_gtv.csv']
delta_rad_tables = ['f1_gtv.csv', 'f5_gtv.csv',
                    'rd_f1_f5_gtv.csv', 'rd_f1_f3_gtv.csv', 'rd_f1_f4_gtv.csv']

feat_sel_algo_list = ['RF', 'ADABOOST', 'ANOVA_PERC', 'ANOVA_K_BEST', 'CHI2_PERC',
                      'CHI2_K_BEST', 'MI_PERC', 'MI_K_BEST', 'NO_SEL', 'RDM_SEL', 'LASSO']
pred_algo_list = ['DT', 'RF', 'ADABOOST', 'PSVM', 'KNN',
                  'LOGREG', 'LOGREGRIDGE', 'BAGG', 'MLP', 'LDA', 'QDA', 'NaiveB']
MAX_FEATURES = 3
outcomes_list = ['Récidive Locale', 'Décès']
metric_list = ['roc_auc', 'sensitivity', 'specificity']

results = pd.read_json('json_results/result_random_run2.json')


top_results = af.get_best_results(results, delta_rad_tables, feat_sel_algo_list,
                                  pred_algo_list, 'Décès', metric='roc_auc', k=1)

nice_tables = ['F1', 'F5', 'F1/F5', 'F1/F3', 'F1/F4']
# nice_tables = ['1 month', 'Simu', 'Delta']

af.compare_roc_auc(top_results, 'Décès', nice_tables, cval=False)
