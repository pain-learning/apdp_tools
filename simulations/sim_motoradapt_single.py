"""
data simulation and fitting for motor adaptation / PomLab task (state space model)
"""
import sys, os
import numpy as np
import pandas as pd
import stan

def sim_motoradapt_single(param_dict, sd_dict, group_name, seed, num_sj=50, num_trial=200, model_name='motoradapt_single', plot=False, plot_raw=False):
    """simulate with state space model for multiple subjects"""
    multi_subject = []
    
    # generate new params
    np.random.seed(seed)
    for sj in range(num_sj):
        sample_params = dict()
        for key in param_dict:
            sample_params[key] = np.random.normal(param_dict[key], sd_dict[key], size=1)[0]
        # print(sample_params)
        df_sj = model_motoradapt_single(sample_params, sj, num_trial)
        multi_subject.append(df_sj)
        
    df_out = pd.concat(multi_subject)
    # plot check
    if plot:
        plot_state(df_out, plot_raw)
    # saving output
    output_dir = './sim_output/motoradapt_sim/'
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    f_name = model_name+'_'+group_name+'_'+str(seed)
    df_out.to_csv(output_dir+f_name+'.txt', sep='\t', index=False)
    # print(df_out)

def sigmoid(x):
    return 1./(1.+np.exp(-x))

def model_motoradapt_single(param_dict, subjID, num_trial=200):
    """state space model, single process model"""
    # rotation schedule (50 baseline trials, then rotation trials, 50 washout trials)
    n_baseline = 3
    rotation = np.concatenate([np.zeros(n_baseline), 15*np.ones(num_trial-n_baseline)])
    # rotation = np.concatenate([np.zeros(50), np.arange(0.,45.,45./50), 45.*np.ones(num_trial-150), np.zeros(50)])

    # initialise
    state_single = 0.
    sim_error = 0.
    # transform params
    A_retention = sigmoid(param_dict['A_retention'])
    B_learning = sigmoid(param_dict['B_learning'])
    norm_sig = np.exp(param_dict['norm_sig'])
    # initialise output
    data_out = []
    # simulate trials
    for t in range(num_trial):
        # trial error, error = current state - perturbation
        sim_error = state_single - rotation[t]
        # update, retained state - learning rate * error
        state_single = A_retention*state_single - B_learning*sim_error
        # output (add noise)
        single_state_rand = np.random.normal(state_single, norm_sig)
        data_out.append([subjID, t, rotation[t], single_state_rand])

    df_out = pd.DataFrame(data_out)
    df_out.columns = ['subjID', 'trial', 'rotation', 'state']

    return df_out

def motoradapt_preprocess_func(txt_path):
    """parse simulated data for pystan"""
    # Iterate through grouped_data
    subj_group = pd.read_csv(txt_path, sep='\t')

    # Use general_info(s) about raw_data
    subj_ls = np.unique(subj_group['subjID'])
    n_subj = len(subj_ls)
    t_subjs = np.array([subj_group[subj_group['subjID']==x].shape[0] for x in subj_ls])
    t_max = max(t_subjs)

    # Initialize (model-specific) data arrays
    rotation = np.full((n_subj, t_max), -1, dtype=int)
    state = np.full((n_subj, t_max), 0, dtype=float)

    # Write from subj_data to the data arrays
    for s in range(n_subj):
        subj_data = subj_group[subj_group['subjID']==s]
        t = t_subjs[s]
        rotation[s][:t] = subj_data['rotation']
        state[s][:t] = subj_data['state']

    # Wrap into a dict for pystan
    data_dict = {
        'N': int(n_subj),
        'T': int(t_max),
        'Tsubj': t_subjs.astype(int),
        'rotation': rotation,
        'state': state,
    }
    # print(data_dict)
    # Returned data_dict will directly be passed to pystan
    return data_dict

def plot_state(df_out, plot_raw=False):
    """plot state from model"""
    fig = plt.subplots(figsize=(6,5))
    if plot_raw:
        for n in range(10):
            df = df_out[df_out['subjID']==n]
            plt.plot(df['state'], label='State')
    else:
        df = df_out
        sns.lineplot(x='trial', y='state', data=df)
    # plt.vlines(50, -10, 60, colors='black', linestyles='--')
    # plt.vlines(max(df['trial'])-50, -10, 60, colors='black', linestyles='--')
    df0 = df[df['subjID']==0]
    plt.plot(df0['rotation'], color='black')
    plt.hlines(0, 0, max(df['trial']), colors='black')
    plt.legend()
    plt.xlabel('Trial')
    plt.ylabel('Simluated reaching direction')
    plt.title('Single state-space model')
    plt.show()

if __name__ == "__main__":
    # healthy control parameters (made up based on Takiyama 2016)
    param_dict_hc = {
        'A_retention': 2.7,  # retention rate 0.92
        'B_learning': -1.3,  # learning rate 0.33
        'norm_sig': .6 # sd of individual trajectory 1.65
    }
    # patient parameters (made up based on Takiyama 2016)
    param_dict_pt = {
        'A_retention': 1.1,  # retention rate 0.81
        'B_learning': -0.1,  # learning rate 0.47
        'norm_sig': .9 # sd of individual trajectory 1
    }
    # healthy control sd
    sd_dict_hc = {
        'A_retention': 0.8,  # retention rate
        'B_learning': 0.6,  # learning rate
        'norm_sig': 1.8 # sd of individual trajectory 
    }
    # patient sd
    sd_dict_pt = {
        'A_retention': 0.8,  # retention rate
        'B_learning': 0.5,  # learning rate
        'norm_sig': 1.6 # sd of individual trajectory 
    }
    
    # parsing cl arguments
    group_name = sys.argv[1] # pt=patient, hc=control
    seed_num = int(sys.argv[2]) # seed number
    subj_num = int(sys.argv[3]) # subject number to simulate
    trial_num = int(sys.argv[4]) # trial number to simulate

    # simulate
    model_name = 'motoradapt_single'
    if group_name == 'hc':
        # simulate hc subjects with given params
        sim_motoradapt_single(param_dict_hc, sd_dict_hc, group_name, seed=seed_num,num_sj=subj_num, num_trial=trial_num, model_name=model_name, plot=False, plot_raw=False)
    elif group_name == 'pt':
        # simulate pt subjects with given params
        sim_motoradapt_single(param_dict_pt, sd_dict_pt, group_name, seed=seed_num, num_sj=subj_num, num_trial=trial_num, model_name=model_name)
    else:
        print('check group name (hc or pt)')

    # parse simulated data
    txt_path = f'./sim_output/motoradapt_sim/motoradapt_single_{group_name}_{seed_num}.txt'
    data_dict = motoradapt_preprocess_func(txt_path)

    # fit stan model
    model_code = open('./models/motoradapt_single.stan', 'r').read()
    posterior = stan.build(program_code=model_code, data=data_dict)
    fit = posterior.sample(num_samples=2000, num_chains=4)
    df = fit.to_frame()  # pandas `DataFrame, requires pandas
    print(df['mu_A'].agg(['mean','var']))
    print(df['mu_B'].agg(['mean','var']))

    # saving traces
    pars = ['mu_A', 'mu_B', 'mu_sig']
    df_extracted = df[pars]
    save_dir = './sim_output/motoradapt_trace/'
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    sfile = save_dir + f'{group_name}_sim_{seed_num}.csv'
    df_extracted.to_csv(sfile, index=None)

    # # fit stan model
    # sm = pystan.StanModel(file='motoradapt_single.stan')
    # fit = sm.sampling(data=data_dict, iter=2000, chains=4)
    # print(fit)

    # # saving
    # pars = ['mu_A', 'mu_B', 'mu_sig']
    # extracted = fit.extract(pars=pars, permuted=True)
    # # print(extracted)
    # sfile = f'./sim_output/motoradapt_sim/{group_name}_sim_{seed_num}.pkl'
    # with open(sfile, 'wb') as op:
    #     tmp = { k: v for k, v in extracted.items() if k in pars } # dict comprehension
    #     pickle.dump(tmp, op)
