"""
data simulation and fitting for 3-arm bandit task (combined model with lapse as used in fapia)
"""
import os, sys
import numpy as np
import pandas as pd
import stan
from scipy.special import softmax

def sim_bandit3arm_combined(param_dict, sd_dict, group_name, seed, num_sj=50, num_trial=200, model_name='bandit3arm_combined'):
    """simulate 3 arm bandit data for multiple subjects"""
    multi_subject = []

    # generate new params
    np.random.seed(seed)
    sample_params = dict()
    for key in param_dict:
        sample_params[key] = np.random.normal(param_dict[key], sd_dict[key], size=1)[0]

    for sj in range(num_sj):
        df_sj = model_bandit3arm_combined(sample_params, sj, num_trial)
        multi_subject.append(df_sj)

    df_out = pd.concat(multi_subject)
    # saving output
    output_dir = './tmp_output/bandit3arm_combined_sim_'+str(num_sj)+'/'
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    f_name = model_name+'_'+group_name+'_'+str(seed)+'_'+str(num_trial)+'_'+str(num_sj)
    df_out.to_csv(output_dir+f_name+'.txt', sep='\t', index=False)
    print(df_out)
    # return df_out

def model_bandit3arm_combined(param_dict, subjID, num_trial=200):
    """simulate 3-arm bandit choices and outcomes"""
    # load reward/pain probabilities
    pun_prob = pd.read_csv('./probs/pain_prob3arm_t240_jakubSmooth_01_28_10_21_01358.csv').values
    rew_prob = pd.read_csv('./probs/reward_prob3arm_t240_jakubSmooth_01_28_10_21_01358.csv').values

    # initialise values
    Qr = np.zeros(3)
    Qp = np.zeros(3)

    # initial probabilities of choosing each deck
    pD = (1/3) * np.ones(3)

    # initialise output
    data_out = []
    # simulate trials
    for t in range(num_trial):
        # select a deck
        tmpDeck = int(np.random.choice(np.arange(3), size=1, p=pD, replace=True))

        # compute tmpRew and tmpPun
        tmpRew = int(np.random.binomial(size=1, n=1, p=rew_prob[t, tmpDeck]))
        tmpPun = -1 * int(np.random.binomial(size=1, n=1, p=pun_prob[t, tmpDeck])) # punishment=-1

        # compute PE and update values
        PEr = param_dict['R']*tmpRew - Qr[tmpDeck]
        PEp = param_dict['P']*tmpPun - Qp[tmpDeck]
        # PEr_fic = -Qr
        # PEp_fic = -Qp

        Qr_chosen, Qp_chosen = [], []
        Qr_chosen = Qr[tmpDeck]
        Qp_chosen = Qp[tmpDeck]

        # update Q with decay rate
        Qr = (1-param_dict['d']) * Qr
        Qp = (1-param_dict['d']) * Qp

        # update Qr and Qp
        # Qr += param_dict['Arew'] * PEr_fic
        # Qp += param_dict['Apun'] * PEp_fic

        # replace Q values of chosen deck with correct values
        Qr[tmpDeck] = Qr_chosen + (param_dict['Arew']+param_dict['Apun']) * PEr
        Qp[tmpDeck] = Qp_chosen + (param_dict['Arew']-param_dict['Apun']) * PEp

        # sum Q
        Qsum = Qr + Qp

        # update pD for next trial
        # pD_pre = np.exp(Qsum) / sum(np.exp(Qsum))
        pD_pre = softmax(Qsum-max(Qsum))

        # xi/lapse
        pD = pD_pre * (1.-param_dict['xi']) + param_dict['xi']/3.

        # output
        data_out.append([subjID, t, tmpDeck+1, int(tmpRew), int(tmpPun)])

    df_out = pd.DataFrame(data_out)
    df_out.columns = ['subjID', 'trial', 'choice', 'gain', 'loss']

    return df_out

def bandit_combined_preprocess_func(txt_path):
    """parse simulated data for pystan"""
    # Iterate through grouped_data
    subj_group = pd.read_csv(txt_path, sep='\t')

    # Use general_info(s) about raw_data
    subj_ls = np.unique(subj_group['subjID'])
    n_subj = len(subj_ls)
    t_subjs = np.array([subj_group[subj_group['subjID']==x].shape[0] for x in subj_ls])
    t_max = max(t_subjs)

    # Initialize (model-specific) data arrays
    rew = np.full((n_subj, t_max), 0, dtype=float)
    los = np.full((n_subj, t_max), 0, dtype=float)
    choice = np.full((n_subj, t_max), -1, dtype=int)

    # Write from subj_data to the data arrays
    for s in range(n_subj):
        subj_data = subj_group[subj_group['subjID']==s]
        t = t_subjs[s]
        rew[s][:t] = subj_data['gain']
        los[s][:t] = -1 * np.abs(subj_data['loss'])  # Use abs
        choice[s][:t] = subj_data['choice']

    # Wrap into a dict for pystan
    data_dict = {
        'N': int(n_subj),
        'T': int(t_max),
        'Tsubj': t_subjs.astype(int),
        'rew': rew,
        'los': los,
        'choice': choice,
    }
    # print(data_dict)
    # Returned data_dict will directly be passed to pystan
    return data_dict

if __name__ == "__main__":
    # healthy control parameters (based on FAPIA, table 5.3 combined model params)
    param_dict_hc = {
        'Arew': 0.569,  # reward learning rate
        'Apun': 0.016,  # punishment learning rate
        'R':    9.417,  # reward sensitivity
        'P':    5.778,  # punishment sensitivity
        'xi':   0.026,  # lapse
        'd':    0.469   # decay
    }

    # assumed patient parameters
    param_dict_pt = {
        'Arew': 0.585,  # reward learning rate
        'Apun': 0.106,  # punishment learning rate
        'R':    12.750,  # reward sensitivity
        'P':    9.953,  # punishment sensitivity
        'xi':   0.105,   # lapse
        'd':    0.603   # decay
    }

    # control sd
    sd_dict_hc= {
        'Arew': 0.178,  # reward learning rate
        'Apun': 0.018,  # punishment learning rate
        'R':    5.031,  # reward sensitivity
        'P':    1.715,  # punishment sensitivity
        'xi':   0.003,   # lapse
        'd':    0.261   # decay
    }

    # patient sd
    sd_dict_pt = {
        'Arew': 0.300,  # reward learning rate
        'Apun': 0.010,  # punishment learning rate
        'R':    5.911,  # reward sensitivity
        'P':    1.577,  # punishment sensitivity
        'xi':   0.013,   # lapse
        'd':    0.311   # decay
    }

    # parsing cl arguments
    group_name = sys.argv[1] # pt=patient, hc=control
    seed_num = int(sys.argv[2]) # seed number
    subj_num = int(sys.argv[3]) # subject number to simulate
    trial_num = int(sys.argv[4]) # trial number to simulate

    model_name = 'bandit3arm_combined'
    if group_name == 'hc':
        # simulate hc subjects with given params
        sim_bandit3arm_combined(param_dict_hc, sd_dict_hc, group_name, seed=seed_num,num_sj=subj_num, num_trial=trial_num, model_name=model_name)
    elif group_name == 'pt':
        # simulate pt subjects with given params
        sim_bandit3arm_combined(param_dict_pt, sd_dict_pt, group_name, seed=seed_num, num_sj=subj_num, num_trial=trial_num, model_name=model_name)
    else:
        print('check group name (hc or pt)')

    # parse simulated data
    txt_path = f'./tmp_output/bandit3arm_combined_sim_{subj_num}/bandit3arm_combined_{group_name}_{seed_num}_{trial_num}_{subj_num}.txt'
    data_dict = bandit_combined_preprocess_func(txt_path)

    # fit stan model
    model_code = open('./models/bandit3arm_combLR_lapse_decay_b.stan', 'r').read()
    posterior = stan.build(program_code=model_code, data=data_dict)
    fit = posterior.sample(num_samples=20, num_chains=1)
    df = fit.to_frame()  # pandas `DataFrame, requires pandas
    print(df['mu_Arew'].agg(['mean','var']))
    print(df['mu_Apun'].agg(['mean','var']))

    # saving traces
    pars = ['mu_Arew', 'mu_Apun', 'mu_R', 'mu_P', 'mu_xi','mu_d']
    df_extracted = df[pars]
    save_dir = './tmp_output/bandit3arm_combined_trace_'+str(subj_num)+'/'
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    sfile = save_dir + f'{group_name}_sim_{seed_num}_{trial_num}_{subj_num}.csv'
    df_extracted.to_csv(sfile, index=None)

    # # fit stan model (pystan2 version)
    # sm = pystan.StanModel(file='bandit3arm_combLR_lapse_decay_b.stan')
    # fit = sm.sampling(data=data_dict, iter=2000, chains=1)
    # print(fit)

    # # saving
    # pars = ['mu_Arew', 'mu_Apun', 'mu_R', 'mu_P', 'mu_xi','mu_d']
    # extracted = fit.extract(pars=pars, permuted=True)
    # # print(extracted)
    # sfile = f'./tmp_output/bandit_combined_sim/{group_name}_sim_{seed_num}.pkl'
    # with open(sfile, 'wb') as op:
    #     tmp = { k: v for k, v in extracted.items() if k in pars } # dict comprehension
    #     pickle.dump(tmp, op)
