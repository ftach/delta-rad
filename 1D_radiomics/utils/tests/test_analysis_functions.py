'''Script to test the analysis functions in the 1D_radiomics package.'''

import sys
import os

# Add the project root directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from utils.src import analysis_functions

def test_get_top_results():
    '''Test the get_top_results function.'''
 
    # Mock input data
    results = {
        "table1": {
            "ANOVA": {
                "RF": {
                    "Récidive Locale": {
                        5: {
                            "test_metrics": {"auc": {"values": [0.75, 0.75, 0.75]}},
                            "features": ["feat1", "feat2"],
                            "params": {"param1": 1}
                        }, 
                        10: {
                            "test_metrics": {"auc": {"values": [0.8, 0.8, 0.8]}},
                            "features": ["feat3", "feat4"],
                            "params": {"param1": 2}
                        }}},
                "LASSO": {
                    "Récidive Locale": {
                        5: {
                            "test_metrics": {"auc": {"values": [0.7, 0.7, 0.7]}},
                            "features": ["feat5", "feat6"],
                            "params": {"param1": 3}
                        },
                        10: {
                            "test_metrics": {"auc": {"values": [0.72, 0.72, 0.72]}},
                            "features": ["feat7", "feat8"],
                            "params": {"param1": 4}
                        }
                    }
                }
            }
        }
        
    }
    # Mock parame   ters
    delta_rad_tables = ["table1"]
    feat_sel_algo_list = ["ANOVA"]
    pred_algo_list = ["RF", "LASSO"]
    metric = "test_auc"
    k = 2

    # Expected output
    expected_output = {
        "table1": {
            "0.8": {"feat_sel_algo": "ANOVA", "pred_algo": "RF", "features": ["feat3", "feat4"], "params": {"param1": 2}},
            "0.75": {"feat_sel_algo": "ANOVA", "pred_algo": "RF", "features": ["feat1", "feat2"], "params": {"param1": 1}}
        }
    }
    
    # Call the function
    output = analysis_functions.get_top_results(results, delta_rad_tables, feat_sel_algo_list, pred_algo_list, metric, k)
    # Assert the output matches the expected output
    assert output == expected_output, f"Expected {expected_output}, but got {output}"

    k = 4
    expected_output = {
        "table1": {
            "0.8": {"feat_sel_algo": "ANOVA", "pred_algo": "RF", "features": ["feat3", "feat4"], "params": {"param1": 2}},
            "0.75": {"feat_sel_algo": "ANOVA", "pred_algo": "RF", "features": ["feat1", "feat2"], "params": {"param1": 1}}, 
            "0.72": {"feat_sel_algo": "ANOVA", "pred_algo": "LASSO", "features": ["feat7", "feat8"], "params": {"param1": 4}},
            "0.7": {"feat_sel_algo": "ANOVA", "pred_algo": "LASSO", "features": ["feat5", "feat6"], "params": {"param1": 3}}
        }
    }

    output = analysis_functions.get_top_results(results, delta_rad_tables, feat_sel_algo_list, pred_algo_list, metric, k)
    # Assert the output matches the expected output
    assert output == expected_output, f"Expected {expected_output}, but got {output}"
    
def test_get_top_results_to_plot(): 
    '''Test the get_top_results_to_plot function.'''
    # Mock input data
    results = {
        "table1": {
            "ANOVA": {
                "RF": {
                    "Récidive Locale": {
                        5: {
                            "test_metrics": {"auc": {"values": [0.75, 0.75, 0.75]}},
                            "features": ["feat1", "feat2"],
                            "params": {"param1": 1}
                        }, 
                        10: {
                            "test_metrics": {"auc": {"values": [0.8, 0.8, 0.8]}},
                            "features": ["feat3", "feat4"],
                            "params": {"param1": 2}
                        }}},
                "LASSO": {
                    "Récidive Locale": {
                        5: {
                            "test_metrics": {"auc": {"values": [0.7, 0.7, 0.7]}},
                            "features": ["feat5", "feat6"],
                            "params": {"param1": 3}
                        },
                        10: {
                            "test_metrics": {"auc": {"values": [0.72, 0.72, 0.72]}},
                            "features": ["feat7", "feat8"],
                            "params": {"param1": 4}
                        }
                    }
                }
            }
        }, 
        "table2": {
            "ANOVA": {
                "RF": {
                    "Récidive Locale": {
                        5: {
                            "test_metrics": {"auc": {"values": [0.6, 0.6, 0.6]}},
                            "features": ["feat1", "feat2"],
                            "params": {"param1": 1}
                        }, 
                        10: {
                            "test_metrics": {"auc": {"values": [0.2, 0.2, 0.2]}},
                            "features": ["feat3", "feat4"],
                            "params": {"param1": 2}
                        }}},
                "LASSO": {
                    "Récidive Locale": {
                        5: {
                            "test_metrics": {"auc": {"values": [0.7, 0.7, 0.7]}},
                            "features": ["feat5", "feat6"],
                            "params": {"param1": 3}
                        },
                        10: {
                            "test_metrics": {"auc": {"values": [0.72, 0.72, 0.72]}},
                            "features": ["feat7", "feat8"],
                            "params": {"param1": 4}
                        }
                    }
                }
            }
        }
        
    }
    # Mock parameters
    delta_rad_tables = ["table1", "table2"]
    feat_sel_algo_list = ["ANOVA"]
    pred_algo_list = ["RF", "LASSO"]
    metric = "test_auc"
    k = 2

    # Expected output
    expected_output = {'table1': [0.8, 0.75], 'table2': [0.72, 0.7]}
    top_results = analysis_functions.get_top_results(results, delta_rad_tables, feat_sel_algo_list, pred_algo_list, metric, k)
    output = analysis_functions.get_top_results_to_plot(top_results)

    # Assert the output matches the expected output
    assert output == expected_output, f"Expected {expected_output}, but got {output}"
    
if __name__ == "__main__":
    test_get_top_results()
    test_get_top_results_to_plot()
    print("All tests passed!")