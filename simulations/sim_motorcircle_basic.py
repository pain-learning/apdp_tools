"""
data simulation and fitting for motor circle task (probablistic model)
"""
import sys, os
import numpy as np
import pandas as pd
import stan

def sim_motorcircle(task_params, param_dict, sd_dict, group_name, seed, num_sj=50, num_trial=200, model_name='motorcircle_basic'):
    """simulate with motor circle model for multiple subjects"""
    multi_subject = []
    
    # generate new params
    np.random.seed(seed)
    for sj in range(num_sj):
        sample_params = dict()
        for key in param_dict:
            sample_params[key] = np.random.normal(param_dict[key], sd_dict[key], size=1)[0]
        # print(sample_params)
        df_sj = model_motorcircle(task_params, sample_params, sj, num_trial)
        multi_subject.append(df_sj)
        
    df_out = pd.concat(multi_subject)
    # saving output
    output_dir = './tmp_output/motorcircle_sim/'
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    f_name = model_name+'_'+group_name+'_'+str(seed)
    df_out.to_csv(output_dir+f_name+'.txt', sep='\t', index=False)
    # print(df_out)

def model_motorcircle(task_params, subject_params, subjID=0, num_trial=200):
    """basic model in Trommershauser 2005, single trial
    Rp - positive gain circle (target circle)
    Rn - negative gain circle (penalty circle)
    G - gain
    S - choice strategy"""
    # movement end point (x',y') are distributed according to a spatially isotropic Gaussian with width sigma wrt the mean end point on plane (x,y)
    # derive sim params from task params
    x = task_params['x_target'] + (-task_params['radius']*task_params['penalty_val']/100)*subject_params['loss_sens']
    y = task_params['y_target']
    sigx = subject_params['perturb']* (task_params['x_target']-task_params['x_penalty'])
    sigy = subject_params['perturb']* (task_params['x_target']-task_params['x_penalty'])
    # optimal strategy
    # mean = [subject_params['x'], subject_params['y']]
    # cov = [[subject_params['sigx'],0],[0,subject_params['sigy']]]
    mean = [x, y]
    cov = [[sigx, 0], [0, sigy]]
    x_sim, y_sim = np.random.multivariate_normal(mean, cov, num_trial).T
    # sim trajectory
    df_out = pd.DataFrame({
        'subjID': np.ones(num_trial)*subjID,
        'trial': np.arange(num_trial),
        'x': x_sim,
        'y': y_sim
    })
    return df_out

def motorcircle_preprocess_func(txt_path, task_params=None):
    """parse simulated data for pystan"""
    # Iterate through grouped_data
    subj_group = pd.read_csv(txt_path, sep='\t')

    # Use general_info(s) about raw_data
    subj_ls = np.unique(subj_group['subjID'])
    n_subj = len(subj_ls)
    t_subjs = np.array([subj_group[subj_group['subjID']==x].shape[0] for x in subj_ls])
    t_max = int(max(t_subjs))

    # Initialize (model-specific) data arrays
    x = np.full((n_subj, t_max), 0, dtype=float)
    y = np.full((n_subj, t_max), 0, dtype=float)

    # Write from subj_data to the data arrays
    for s in range(n_subj):
        subj_data = subj_group[subj_group['subjID']==s]
        t = t_subjs[s]
        x[s][:t] = subj_data['x']
        y[s][:t] = subj_data['y']

    # Wrap into a dict for pystan
    data_dict = {
        'N': n_subj,
        'T': t_max,
        # 'Tsubj': t_subjs.tolist(),
        'x': x.tolist(),
        'y': y.tolist(),
    }
    # add task params
    if task_params is not None:
        data_dict.update(task_params)
    # print(data_dict)
    # Returned data_dict will directly be passed to pystan
    return data_dict



if __name__ == "__main__":
    # simulation parameters
    task_near = {
        'x_target': 0, # target circle x
        'y_target': 0, # target circle y
        'radius': 1, # radius
        'x_penalty': -1, # penalty circle x
        'y_penalty': 0, # penalty circle y
        'penalty_val': -200 # penalty value
    }
    # control params
    params_hc = {
        'loss_sens': 0.1, # sensitivity to loss
        'perturb': 0.05, # cov 
    }
    sd_hc = {
        'loss_sens': 0.05, # sensitivity to loss
        'perturb': 0.03, # cov 
    }
    # patients param
    params_pt = {
        'loss_sens': 0.12, # sensitivity to loss
        'perturb': 0.1, # cov 
    }
    sd_pt = {
        'loss_sens': 0.05, # sensitivity to loss
        'perturb': 0.03, # cov 
    }
    ########### simulation
    group_name = sys.argv[1] # pt=patient, hc=control
    seed_num = int(sys.argv[2]) # seed number
    subj_num = int(sys.argv[3]) # subject number to simulate
    trial_num = int(sys.argv[4]) # trial number to simulate

    model_name = 'motorcircle_near'
    if group_name == 'hc':
        # simulate controls
        sim_motorcircle(task_near, params_hc, sd_hc, group_name, seed=seed_num, num_sj=subj_num, num_trial=trial_num, model_name=model_name)
    elif group_name == 'pt':
        # simulate patients
        sim_motorcircle(task_near, params_pt, sd_pt, group_name, seed=seed_num, num_sj=subj_num, num_trial=trial_num, model_name=model_name)
    else:
        print('check group name (hc or pt)')

    # parse simulated data
    txt_path = f'./tmp_output/motorcircle_sim/motorcircle_near_{group_name}_{seed_num}.txt'
    data_dict = motorcircle_preprocess_func(txt_path, task_params=task_near)
    # print(data_dict)
    model_code = open('./models/motorcircle_basic.stan', 'r').read()

    # fit stan model
    posterior = stan.build(program_code=model_code, data=data_dict)
    fit = posterior.sample( num_samples=2000, num_chains=4)
    df = fit.to_frame()  # pandas `DataFrame, requires pandas
    print(df['loss_sens'].agg(['mean','var']))
    print(df['perturb'].agg(['mean','var']))

    # saving traces
    pars = ['loss_sens', 'perturb']
    df_extracted = df[pars]
    save_dir = './tmp_output/motorcircle_trace/'
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    sfile = save_dir + f'{group_name}_sim_{seed_num}.csv'
    df_extracted.to_csv(sfile, index=None)