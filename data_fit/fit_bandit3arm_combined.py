"""
fit motor circle task with external data (not simulated)
"""
import sys, os
import numpy as np
import pandas as pd
import stan

sys.path.append('.')
from simulations.sim_bandit3arm_combined import bandit_combined_preprocess_func


if __name__ == "__main__":
    # task params
    # task_params = {
    #     'x_target': 0, # target circle x
    #     'y_target': 0, # target circle y
    #     'radius': 0.15, # radius
    #     'x_penalty': 0, # penalty circle x
    #     'y_penalty': -0.01, # penalty circle y, small diff to make sure initialisation ok
    #     'penalty_val': -2 # penalty value
    # }

   # parse data
    # txt_path = f'./transformed_data/circlemotor/circlemotor_data0.txt'
    txt_path = f'./transformed_data/bandit3arm/bandit3arm_data.txt'
    data_dict = bandit_combined_preprocess_func(txt_path)#, task_params=task_params)
    model_code = open('./models/bandit3arm_combLR_lapse_decay_b.stan', 'r').read()

    # fit stan model
    posterior = stan.build(program_code=model_code, data=data_dict)
    fit = posterior.sample(num_samples=2000, num_chains=4)
    df = fit.to_frame()  # pandas `DataFrame, requires pandas
    print(df['mu_Arew'].agg(['mean','var']))
    print(df['mu_Apun'].agg(['mean','var']))

    # saving traces
    pars = ['mu_Arew', 'mu_Apun', 'mu_R', 'mu_P', 'mu_xi','mu_d']
    df_extracted = df[pars]
    save_dir = './tmp_output/bandit3arm_combined_mydata_trace/'
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    sfile = save_dir + f'mydata_fit.csv'
    df_extracted.to_csv(sfile, index=None)
