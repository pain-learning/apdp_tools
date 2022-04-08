"""
fit motor circle task with external data (not simulated)
"""
import sys, os
import numpy as np
import pandas as pd
import stan

sys.path.append('.')
from simulations.sim_motorcircle_basic import motorcircle_preprocess_func 


if __name__ == "__main__":
    # task params
    task_params = {
        'x_target': 0, # target circle x
        'y_target': 0, # target circle y
        'radius': 0.15, # radius
        'x_penalty': 0, # penalty circle x
        'y_penalty': -0.01, # penalty circle y, small diff to make sure initialisation ok
        'penalty_val': -2 # penalty value
    }

   # parse data
    # txt_path = f'./transformed_data/circlemotor/circlemotor_data0.txt'
    txt_path = f'./transformed_data/circlemotor/circlemotor_data1.txt'
    data_dict = motorcircle_preprocess_func(txt_path, task_params=task_params)
    # print(data_dict)
    model_code = open('./models/motorcircle_realdata.stan', 'r').read() # moved to y changes

    # fit stan model
    posterior = stan.build(program_code=model_code, data=data_dict)
    fit = posterior.sample( num_samples=2000, num_chains=4)
    df = fit.to_frame()  # pandas `DataFrame, requires pandas
    print(df['loss_sens'].agg(['mean','var']))
    print(df['perturb'].agg(['mean','var']))

    # saving traces
    pars = ['loss_sens', 'perturb']
    df_extracted = df[pars]
    save_dir = './data_output/motorcircle_mydata_trace/'
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    sfile = save_dir + f'mydata_fit.csv'
    df_extracted.to_csv(sfile, index=None)