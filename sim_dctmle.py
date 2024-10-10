#####################################################################################
# Simulations for double cross-fit TMLE:
#   contains correct model, main-terms model, and machine learning
#
# Unlike other files, this parallels the simulation process along a number of
#   specified CPU's. The number of CPUs allotted can be adjusted in the parameters
#####################################################################################

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from estimators_dctmle import DoubleCrossfitTMLE
from super_learner import superlearnersetup


##########################
# Setting some parameters
setup = 1  # options include: 1-correct parametric model, 2-main-terms model, 3-machine learning with main-terms
decimal = 4  # decimal places to display in results printed to console
file_path_to_save_results = "dctmle_results"+str(setup)+".csv"  # file path to save all result output
n_diff_splits = 10  # number of different splits to use

##########################
# Reading in data
df = pd.read_csv("statin_sim_data_mini.csv")
truth = -0.1081508
samples = list(df['sim_id'].unique())

if setup == 1:
    # Set-up for correct parametric model
    df['ldl_160'] = np.where(df['ldl_log'] > np.log(160), 1, 0)
    df['age_30'] = df['age'] - 30
    df['age_30_sq'] = (df['age'] - 30)**2
    df['ldl_130'] = np.where(df['ldl_log'] < np.log(130), 5-df['ldl_log'], 0)
    df['age_sqrt'] = np.sqrt(df['age']-39)
    df['risk_exp'] = np.exp(df['risk_score']+1)
    df['ldl_120'] = np.where(df['ldl_log'] > np.log(120), df['ldl_log']**2, 0)

    g_model = 'diabetes + ldl_log + ldl_160 + age_30 + age_30_sq + C(risk_score_cat)'
    q_model = 'statin + statin:ldl_130 + age_sqrt + diabetes + risk_exp + ldl_120'
    g_estimator = LogisticRegression(penalty='none', solver='lbfgs', max_iter=1000)
    q_estimator = LogisticRegression(penalty='none', solver='lbfgs', max_iter=1000)

elif setup == 2:
    # Set-up for main-term parametric model
    g_model = 'diabetes + age + risk_score + ldl_log'
    q_model = 'statin + diabetes + age + risk_score + ldl_log'
    g_estimator = LogisticRegression(penalty='none', solver='lbfgs', max_iter=1000)
    q_estimator = LogisticRegression(penalty='none', solver='lbfgs', max_iter=1000)

elif setup == 3:
    # Set-up for machine learning
    g_model = 'diabetes + age + risk_score + ldl_log'
    q_model = 'statin + diabetes + age + risk_score + ldl_log'
    g_estimator = superlearnersetup(var_type='binary', K=10)
    q_estimator = superlearnersetup(var_type='binary', K=10)

else:
    raise ValueError("Invalid setup choice")

##########################
# Running simulation
bias = []
stderr = []
lcl = []
ucl = []

for i in samples:
    print(f"Processing sample {i}")
    dfs = df.loc[df['sim_id'] == i].copy()
    print(f"Sample size: {len(dfs)}")

    try:
        dcdr = DoubleCrossfitTMLE(dfs, 'statin', 'Y')
        print("DoubleCrossfitTMLE initialized")
        dcdr.treatment_model(g_model, g_estimator, bound=0.01)
        print("Treatment model fitted")
        dcdr.outcome_model(q_model, q_estimator)
        print("Outcome model fitted")
        dcdr.fit(resamples=n_diff_splits, method='median')
        print("TMLE fitted")

        bias.append(dcdr.risk_difference - truth)
        stderr.append(dcdr.risk_difference_se)
        lcl.append(dcdr.risk_difference_ci[0])
        ucl.append(dcdr.risk_difference_ci[1])

    except Exception as e:
        print(f"Error occurred: {str(e)}")        
        bias.append(np.nan)
        stderr.append(np.nan)
        lcl.append(np.nan)
        ucl.append(np.nan)


results = pd.DataFrame()
results['bias'] = bias
results['std'] = stderr
results['lcl'] = lcl
results['ucl'] = ucl
results['cover'] = np.where((results['lcl'] < truth) & (truth < results['ucl']), 1, 0)
results['cld'] = results['ucl'] - results['lcl']

print("============================")
print("DC-TMLE")
print("============================")
print("Mean: ", np.round(np.mean(bias), decimal))
print("RMSE:  ", np.round(np.sqrt(np.mean(results['bias'])**2 + np.std(bias, ddof=1)**2), decimal))
print("ASE:  ", np.round(np.mean(stderr), decimal))
print("ESE:  ", np.round(np.std(bias, ddof=1), decimal))
print("CLD:", np.round(np.mean(results['cld']), decimal))
print("Cover:", np.round(np.mean(results['cover']), decimal))
print("============================")

# results.to_csv(file_path_to_save_results, index=False)
print(results)