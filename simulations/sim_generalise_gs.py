"""
data simulation and fitting for generalisation instrumetnal avoidance task (with gen in model)
"""
import sys, os
import numpy as np
import pandas as pd
import stan

def sigmoid(p):
    return 1./(1.+np.exp(-p))

def softmax_perception(s_cue, Q, beta, bias):
    """probability of avoidance given stim and params"""
    avoid_cost = 0.2 # fixed
    if s_cue == 1:
        pred_V = 0.75*Q[s_cue-1] + 0.25*Q[2-1]
    elif s_cue == 3:
        pred_V = 0.75*Q[s_cue-1] + 0.25*Q[2-1]
    elif s_cue == 5:
        pred_V = 0.75*Q[s_cue-1] + 0.25*Q[6-1]
    elif s_cue == 7:
        pred_V = 0.75*Q[s_cue-1] + 0.25*Q[6-1]
    else:
        pred_V = Q[s_cue-1]

    # avoidance probability
    gx = 1./(1. + np.exp(-beta*(0. - pred_V - avoid_cost - bias)))
    return gx

def draw_cue(num_trial):
    """drawing cues based on given probability"""
    trial_type = []
    p_cues = [0.052, 0.264, 0.052, 0.264,
            0.052, 0.264, 0.052]
    for i in range(num_trial):
        smp = np.random.choice(7, p=p_cues)
        trial_type.append(smp+1) # match matlab number
    return trial_type

def model_generalise_gs(param_dict, subjID, group_name, num_trial=190):
    """simulate shapes, avoid actions, and shock outcomes"""
    # load predefined image sequences (38 trials x 5 blocks)
    # trial_type = np.squeeze(pd.read_csv('./probs/generalise_stim.csv').values) # 1:7
    trial_type = draw_cue(num_trial)
    num_state = len(np.unique(trial_type))

    # initialise values (values from AN's VBA code)
    # Q = sigmoid(-0.2) * np.ones(num_state) # 7 possible cues
    Q = np.zeros(num_state) # 7 possible cues
    alpha = sigmoid(0.95) * np.ones(1) # initial learning rate

    # initialise output
    data_out = []
    rt = 0

    # simulate trials
    for t in range(num_trial):
        # a cue is shown (not sure how it's generated)
        s_cue = int(trial_type[t])

        # avoid or not
        p_avoid = softmax_perception(s_cue, Q, param_dict['beta'], param_dict['bias'])
        a = int(np.random.binomial(size=1, n=1, p=p_avoid))
        # a = int(np.random.choice([0,1], size=1,
        #     p=[1-p_avoid, p_avoid], replace=True))
        # print(p_avoid, a)

        # deliver shock or not
        if (s_cue == 2 or s_cue == 4) and a == 0: # if CS+1 or 2 shock trials and no avoidance made
            r = -1
        else:
            r = 0

        # define sensory params
        # mean_theta = 0.25
        # rhos = [0.25-mean_theta, 0.25, 0.25+mean_theta, 0.75-2*mean_theta, 0.75-mean_theta, 0.75, 0.75+mean_theta]
        rhos = np.array([0.0, 0.25, 0.5, 0.25, 0.5, 0.75, 1.0])

        # compute PE and update values
        if a == 0: # did not avoid
            # PE update
            PE = r - Q[s_cue-1]

            # define sigma
            if r == 0: # no shock
                sigma_t = param_dict['sigma_n']
            else: # shock
                sigma_t = param_dict['sigma_a']

            # update Q
            # for s in range(num_state):
            #     rho = rhos[s]
            #     G = 1./np.exp((rho-cue_rho)**2. / (2.*sigma_t**2.))
            #     Q[s] += param_dict['kappa'] * alpha * PE * G
            # current cue rho value
            cue_rho = rhos[s_cue-1] * np.ones(num_state)
            diff2 = (rhos-cue_rho)**2. / (2.*(sigma_t**2.))
            G = 1./np.exp(diff2)
            Q += param_dict['kappa'] * alpha * PE * G

        else: # avoided
            PE = 0.
            Q = Q

        # update alpha
        alpha = param_dict['eta']*np.abs(PE) + (1-param_dict['eta'])*alpha

        # output
        data_out.append([subjID, group_name, t, s_cue, a, rt, r])

    df_out = pd.DataFrame(data_out)
    df_out.columns = ['subjID', 'group', 'trial', 'cue', 'choice', 'rt', 'outcome']
    # print(df_out)
    return df_out

def sim_generalise_gs(param_dict, sd_dict, group_name, seed, num_sj=50, num_trial=190, model_name='generalise_gs'):
    """simulate generalise instrumental avoidance task for multiple subjects"""
    multi_subject = []

    # generate new params
    np.random.seed(seed)
    sample_params = dict()
    for key in param_dict:
        sample_params[key] = np.random.normal(param_dict[key], sd_dict[key], size=1)[0]

    for sj in range(num_sj):
        df_sj = model_generalise_gs(sample_params, sj, group_name, num_trial)
        multi_subject.append(df_sj)

    df_out = pd.concat(multi_subject)
    # saving output
    output_dir = './sim_output/generalise_sim_'+str(num_sj)+'/'
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    f_name = model_name+'_'+group_name+'_'+str(seed)+'_'+str(num_trial)+'_'+str(num_sj)
    df_out.to_csv(output_dir+f_name+'.txt', sep='\t', index=False)
    print(df_out)

def generalise_gs_preprocess_func(txt_path):
    """parse simulated data for pystan"""
    # Iterate through grouped_data
    subj_group = pd.read_csv(txt_path, sep='\t')

    # Use general_info(s) about raw_data
    subj_ls = np.unique(subj_group['subjID'])
    n_subj = len(subj_ls)
    t_subjs = np.array([subj_group[subj_group['subjID']==x].shape[0] for x in subj_ls])
    t_max = max(t_subjs)

    # Initialize (model-specific) data arrays
    cue = np.full((n_subj, t_max), 0, dtype=int)
    choice = np.full((n_subj, t_max), -1, dtype=int)
    outcome = np.full((n_subj, t_max), -1, dtype=float)
    rt = np.full((n_subj, t_max), -1, dtype=float)
    subjID = np.full((n_subj),0,dtype=int)
    group_n = np.full(n_subj,'gr',dtype=object)

    # Write from subj_data to the data arrays
    for s in range(n_subj):
        subj_data = subj_group[subj_group['subjID']==subj_ls[s]]
        t = t_subjs[s]
        cue[s][:t] = subj_data['cue']
        choice[s][:t] = subj_data['choice']
        outcome[s][:t] = -1 * np.abs(subj_data['outcome'])  # Use abs
        subjID[s] = pd.unique(subj_data['subjID'])
        rt[s][:t] = subj_data['rt']
        group_n[s] = pd.unique(subj_data['group'])[0]

    # Wrap into a dict for pystan
    data_dict = {
        'N': int(n_subj),
        'T': int(t_max),
        'Tsubj': t_subjs.astype(int),
        'cue': cue,
        'choice': choice,
        'outcome': outcome,
        'rt': rt,
        'subjID': subjID,
        'group': group_n
    }
    # print(data_dict)
    # Returned data_dict will directly be passed to pystan
    return data_dict

if __name__ == "__main__":
    # parameters AMT dataset (highest 10% approx patient params, low 40% approx as control params)
    # params made up
    param_dict_hc = {
        'sigma_a': 0.45,  # generalisation param for shock
        'sigma_n': 0.06,  # generalisation param for no shock
        'eta':    0.17,     # p_h dynamic learning rate
        'kappa':  0.75,    # p_h dynamic learning rate
        'beta': 9.5,       # softmax beta
        'bias': 0.3      # softmax bias
    }
    # hc sd
    sd_dict_hc = {
        'sigma_a': 0.05,  # generalisation param for shock
        'sigma_n': 0.01,  # generalisation param for no shock
        'eta':    0.1,     # p_h dynamic learning rate
        'kappa':  0.2,    # p_h dynamic learning rate
        'beta': 2,       # softmax beta
        'bias': 0.1      # softmax bias
    }
    # patient params
    param_dict_pt = {
        'sigma_a': 0.85,  # generalisation param for shock
        'sigma_n': 0.03,  # generalisation param for no shock
        'eta':    0.18,     # p_h dynamic learning rate
        'kappa':  0.76,    # p_h dynamic learning rate
        'beta': 4.3,       # softmax beta
        'bias': 0.3      # softmax bias
    }
    # patient sd
    sd_dict_pt = {
        'sigma_a': 0.05,  # generalisation param for shock
        'sigma_n': 0.01,  # generalisation param for no shock
        'eta':    0.10,     # p_h dynamic learning rate
        'kappa':  0.2,    # p_h dynamic learning rate
        'beta': 2,       # softmax beta
        'bias': 0.1      # softmax bias
    }

    # parsing cl arguments
    group_name = sys.argv[1] # pt=patient, hc=control
    seed_num = int(sys.argv[2]) # seed number
    subj_num = int(sys.argv[3]) # subject number to simulate
    trial_num = int(sys.argv[4]) # trial number to simulate

    model_name = 'generalise_gs'
    if group_name == 'hc':
        # simulate hc subjects with given params
        sim_generalise_gs(param_dict_hc, sd_dict_hc, group_name, seed=seed_num,num_sj=subj_num, num_trial=trial_num, model_name=model_name)
    elif group_name == 'pt':
        # simulate pt subjects with given params
        sim_generalise_gs(param_dict_pt, sd_dict_pt, group_name, seed=seed_num, num_sj=subj_num, num_trial=trial_num, model_name=model_name)
    else:
        print('check group name (hc or pt)')

    # parse simulated data
    txt_path = f'./sim_output/generalise_sim_{subj_num}/generalise_gs_{group_name}_{seed_num}_{trial_num}_{subj_num}.txt'
    data_dict = generalise_gs_preprocess_func(txt_path)
    data_dict_gr = data_dict
    data_dict_gr.pop('group') 
                     
    # fit stan model
    model_code = open('./models/generalise_gs.stan', 'r').read()
    posterior = stan.build(program_code=model_code, data=data_dict_gr)
    fit = posterior.sample(num_samples=20, num_chains=4)
    df = fit.to_frame()  # pandas `DataFrame, requires pandas
    print(df['mu_sigma_a'].agg(['mean','var']))
    print(df['mu_beta'].agg(['mean','var']))

    # saving traces
    pars = ['mu_sigma_a', 'mu_sigma_n', 'mu_eta', 'mu_kappa', 'mu_beta', 'mu_bias']
    df_extracted = df[pars]
    save_dir = './sim_output/generalise_trace_'+str(subj_num)+'/'
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    sfile = save_dir + f'{group_name}_sim_{seed_num}_{trial_num}_{subj_num}.csv'
    df_extracted.to_csv(sfile, index=None)


    # # fit stan model
    # sm = pystan.StanModel(file='generalise_gs.stan')
    # fit = sm.sampling(data=data_dict, iter=3000, chains=2)
    # print(fit)

    # # saving
    # pars = ['mu_sigma_a', 'mu_sigma_n', 'mu_eta', 'mu_kappa', 'mu_beta', 'mu_bias']
    # extracted = fit.extract(pars=pars, permuted=True)
    # # print(extracted)
    # sfile = f'./sim_output/generalise_sim/{group_name}_sim_{seed_num}.pkl'
    # with open(sfile, 'wb') as op:
    #     tmp = { k: v for k, v in extracted.items() if k in pars } # dict comprehension
    #     pickle.dump(tmp, op)

    # hbayesdm method
    # # fit
    # # Run the model and store results in "output"
    # output = generalise_gs('./sim_output/generalise_sim/'+model_name+'_'+group_name+'_'+str(seed_num)+'.txt', niter=3000, nwarmup=1500, nchain=4, ncore=16)

    # # debug
    # print(output.fit)

    # # saving
    # sfile = './sim_output/generalise_sim/'+group_name+'_sim_'+str(seed_num)+'.pkl'
    # with open(sfile, 'wb') as op:
    #     tmp = { k: v for k, v in output.par_vals.items() if k in ['mu_sigma_a', 'mu_sigma_n', 'mu_eta', 'mu_kappa', 'mu_beta', 'mu_bias'] } # dict comprehension
    #     pickle.dump(tmp, op)
