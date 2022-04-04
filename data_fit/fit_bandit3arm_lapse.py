"""
fit motor circle task with external data (not simulated)
"""
import sys, os
import numpy as np
import pandas as pd
import stan

sys.path.append('.')
from simulations.sim_bandit3arm_combined import bandit_combined_preprocess_func
from data_fit.fit_bandit3arm_combined import extract_ind_results


if __name__ == "__main__":

   # parse data
    # txt_path = f'./transformed_data/circlemotor/circlemotor_data0.txt'
    txt_path = f'./transformed_data/bandit3arm/bandit3arm_data.txt'
    data_dict = bandit_combined_preprocess_func(txt_path)#, task_params=task_params)
    # print(data_dict)
    model_code = open('./models/bandit3arm_lapse.stan', 'r').read()

    # fit stan model
    posterior = stan.build(program_code=model_code, data=data_dict)
    fit = posterior.sample(num_samples=10, num_chains=2)
    df = fit.to_frame()  # pandas `DataFrame, requires pandas
    
    # individual results
    pars_ind = ['Arew', 'Apun', 'R', 'P', 'xi']    
    df_ind = extract_ind_results(df,pars_ind,data_dict)
    subjID_df=pd.DataFrame(data_dict['subjID'],columns=['subjID'])
    df_ind = subjID_df.join(df_ind)
    
    print(df['mu_Arew'].agg(['mean','var']))
    print(df['mu_Apun'].agg(['mean','var']))

    # saving traces
    pars = ['mu_Arew', 'mu_Apun', 'mu_R', 'mu_P', 'mu_xi']
    df_extracted = df[pars]
    save_dir = './tmp_output/bandit3arm_lapse_mydata_trace/'
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    sfile = save_dir + f'mydata_fit.csv'
    s_ind_file = save_dir + f'mydata_fit_ind.csv'
    df_extracted.to_csv(sfile, index=None)
    df_ind.to_csv(s_ind_file, index=None)
