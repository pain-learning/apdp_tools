"""
data simulation and fitting for bandit 4-arm task (with lapse in model)
"""
import os, sys
import numpy as np
import pandas as pd
import stan

from sim_bandit4arm_combined import bandit_combined_preprocess_func

def sim_bandit4arm_lapse(param_dict, sd_dict, group_name, seed, num_sj=50, num_trial=200, model_name='bandit4arm_lapse'):
    """simulate 4 arm bandit data for multiple subjects"""
    multi_subject = []
    
    # generate new params
    np.random.seed(seed)
    sample_params = dict()
    for key in param_dict:
        sample_params[key] = np.random.normal(param_dict[key], sd_dict[key], size=1)[0]
    
    for sj in range(num_sj):
        df_sj = model_bandit4arm_lapse(sample_params, sj, num_trial)
        multi_subject.append(df_sj)
        
    df_out = pd.concat(multi_subject)
    # saving output
    output_dir = './tmp_output/bandit_lapse_sim/'
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    f_name = model_name+'_'+group_name+'_'+str(seed)
    df_out.to_csv(output_dir+f_name+'.txt', sep='\t', index=False)
    print(df_out)
    # return df_out

def model_bandit4arm_lapse(param_dict, subjID, num_trial=200):
    """simulate 4-arm bandit choices and outcomes"""
    # load reward/pain probabilities
    pun_prob = pd.read_csv('./probs/pain_prob.csv').values
    rew_prob = pd.read_csv('./probs/reward_prob.csv').values
    
    # initialise values
    Qr = np.zeros(4)
    Qp = np.zeros(4)
    
    # initial probabilities of choosing each deck
    pD = 0.25 * np.ones(4)
    
    # initialise output
    data_out = []
    # simulate trials
    for t in range(num_trial):
        # select a deck
        tmpDeck = int(np.random.choice(np.arange(4), size=1, p=pD, replace=True))

        # compute tmpRew and tmpPun
        tmpRew = int(np.random.binomial(size=1, n=1, p=rew_prob[t, tmpDeck]))
        tmpPun = -1 * int(np.random.binomial(size=1, n=1, p=pun_prob[t, tmpDeck])) # punishment=-1

        # compute PE and update values
        PEr = param_dict['R']*tmpRew - Qr[tmpDeck]
        PEp = param_dict['P']*tmpPun - Qp[tmpDeck]
        PEr_fic = -Qr
        PEp_fic = -Qp

        Qr_chosen, Qp_chosen = [], []
        Qr_chosen = Qr[tmpDeck]
        Qp_chosen = Qp[tmpDeck]
        
        # update Qr and Qp
        Qr += param_dict['Arew'] * PEr_fic
        Qp += param_dict['Apun'] * PEp_fic
        # replace Q values of chosen deck with correct values
        Qr[tmpDeck] = Qr_chosen + param_dict['Arew'] * PEr
        Qp[tmpDeck] = Qp_chosen + param_dict['Apun'] * PEp

        # sum Q
        Qsum = Qr + Qp
        
        # update pD for next trial
        pD_pre = np.exp(Qsum) / sum(np.exp(Qsum))

        # xi/lapse
        pD = pD_pre * (1.-param_dict['xi']) + param_dict['xi']/4.
        
        # output
        data_out.append([subjID, t, tmpDeck+1, int(tmpRew), int(tmpPun)])
        
    df_out = pd.DataFrame(data_out)
    df_out.columns = ['subjID', 'trial', 'choice', 'gain', 'loss']

    return df_out

if __name__ == "__main__":
    # healthy control parameters (based on FAPIA)
    param_dict_hc = {
        'Apun': 0.519,  # punishment learning rate
        'Arew': 0.307,  # reward learning rate
        'R':    9.248,  # reward sensitivity
        'P':    8.643,  # punishment sensitivity
        'xi':   0.018   # lapse
    }

    # assumed patient parameters 
    param_dict_pt = {
        'Apun': 0.648,  # punishment learning rate
        'Arew': 0.337,  # reward learning rate
        'R':    10.583,  # reward sensitivity
        'P':    10.109,  # punishment sensitivity
        'xi':   0.071   # lapse
    }

    # control sd
    sd_dict_hc= {
        'Arew': 0.030,  # reward learning rate
        'Apun': 0.033,  # punishment learning rate
        'R':    0.941,  # reward sensitivity
        'P':    1.248,  # punishment sensitivity
        'xi':   0.006   # lapse
    }

    # patient sd
    sd_dict_pt = {
        'Arew': 0.051,  # reward learning rate
        'Apun': 0.056,  # punishment learning rate
        'R':    1.128,  # reward sensitivity
        'P':    1.129,  # punishment sensitivity
        'xi':   0.015   # lapse
    }

    # parsing cl arguments
    group_name = sys.argv[1] # pt=patient, hc=control
    seed_num = int(sys.argv[2]) # seed number
    subj_num = int(sys.argv[3]) # subject number to simulate
    trial_num = int(sys.argv[4]) # trial number to simulate

    model_name = 'bandit4arm_lapse'
    if group_name == 'hc':
        # simulate hc subjects with given params
        sim_bandit4arm_lapse(param_dict_hc, sd_dict_hc, group_name, seed=seed_num,num_sj=subj_num, num_trial=trial_num, model_name=model_name)
    elif group_name == 'pt':
        # simulate pt subjects with given params
        sim_bandit4arm_lapse(param_dict_pt, sd_dict_pt, group_name, seed=seed_num, num_sj=subj_num, num_trial=trial_num, model_name=model_name)
    else:
        print('check group name (hc or pt)')

    # parse simulated data
    txt_path = f'./tmp_output/bandit_lapse_sim/bandit4arm_lapse_{group_name}_{seed_num}.txt'
    data_dict = bandit_combined_preprocess_func(txt_path)

    # fit stan model
    model_code = open('./models/bandit4arm_lapse.stan', 'r').read()
    posterior = stan.build(program_code=model_code, data=data_dict)
    fit = posterior.sample(num_samples=2000, num_chains=4)
    df = fit.to_frame()  # pandas `DataFrame, requires pandas
    print(df['mu_Arew'].agg(['mean','var']))
    print(df['mu_Apun'].agg(['mean','var']))

    # saving traces
    pars = ['mu_Arew', 'mu_Apun', 'mu_R', 'mu_P', 'mu_xi']
    df_extracted = df[pars]
    save_dir = './tmp_output/bandit_lapse_trace/'
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    sfile = save_dir + f'{group_name}_sim_{seed_num}.csv'
    df_extracted.to_csv(sfile, index=None)


    # # fit
    # # Run the model and store results in "output"
    # output = bandit4arm_lapse('./tmp_output/bandit_sim/'+model_name+'_'+group_name+'_'+str(seed_num)+'.txt', niter=3000, nwarmup=1500, nchain=4, ncore=16)
    
    # # debug
    # print(output.fit)

    # # saving
    # sfile = './tmp_output/bandit_sim/'+group_name+'_sim_'+str(seed_num)+'.pkl'
    # with open(sfile, 'wb') as op:
    #     tmp = { k: v for k, v in output.par_vals.items() if k in ['mu_Arew', 'mu_Apun', 'mu_R', 'mu_P', 'mu_xi'] } # dict comprehension
    #     pickle.dump(tmp, op)


