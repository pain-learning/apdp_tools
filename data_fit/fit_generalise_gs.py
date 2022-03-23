"""
fit motor circle task with external data (not simulated)
"""
import sys, os
import numpy as np
import pandas as pd
import stan

sys.path.append('.')
from simulations.sim_generalise_gs import generalise_gs_preprocess_func


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
    txt_path = f'./transformed_data/generalise/generalise_data.txt'
    data_dict = generalise_gs_preprocess_func(txt_path)#, task_params=task_params)
    # print(data_dict)
    model_code = open('./models/generalise_gs.stan', 'r').read() # moved to y changes

    # fit stan model
    posterior = stan.build(program_code=model_code, data=data_dict)
    fit = posterior.sample(num_samples=2000, num_chains=4)
    df = fit.to_frame()  # pandas `DataFrame, requires pandas
    print(df['mu_sigma_a'].agg(['mean','var']))
    print(df['mu_beta'].agg(['mean','var']))

    # saving traces
    pars = ['mu_sigma_a', 'mu_sigma_n', 'mu_eta', 'mu_kappa', 'mu_beta', 'mu_bias']
    df_extracted = df[pars]
    save_dir = './tmp_output/generalise_mydata_trace/'
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    sfile = save_dir + f'mydata_fit.csv'
    df_extracted.to_csv(sfile, index=None)
