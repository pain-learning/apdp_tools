"""
compare hdi from pystan fit traces (depreciated version)
"""
import os, sys
import pickle
import numpy as np
import pandas as pd
from hbayesdm import rhat, print_fit, plot_hdi, hdi
import arviz as az
from matplotlib import pyplot as plt

def comp_hdi_mean(model_name, param_ls, sort=True, draw_idx=50, draws=1000, seed=123):
    """
    compare hdi by drawing simulations (trace means)
    """
    # define sim dir
    output_dir = './tmp_output/'+model_name+'_sim/'
    np.random.seed(seed)
    significant_df = []
    df_out = []
    # find sim results in groups
    for key in param_ls:
        bounds = []
        # random compare n draws
        for comp in range(1,draws):
            # load MCMC traces with matching seeds (not number of subjects)
            hc_file = os.path.join(output_dir, 'hc_sim_'+str(int(np.random.randint(0,draw_idx,1)))+'.pkl')
            pt_file = os.path.join(output_dir, 'pt_sim_'+str(int(np.random.randint(0,draw_idx,1)))+'.pkl')

            if os.path.isfile(hc_file) and os.path.isfile(pt_file):
                # print(hc_file, pt_file)
                with open(hc_file, 'rb') as hc:
                    hc_dict = pickle.load(hc)
                with open(pt_file, 'rb') as pt:
                    pt_dict = pickle.load(pt)

                # calculate lower bounds of simulation using difference
                hdi_bounds = hdi_diff(key, hc_dict, pt_dict)
                # store hdi bounds
                bounds.append(hdi_bounds)
                # store mean
                df_tmp = pd.DataFrame({
                    'param':[key,key], 
                    'param_mean':[np.mean(hc_dict[key]), np.mean(pt_dict[key])],
                    'group':['control','patient'],
                    'hdi_low':[hdi_bounds[0],hdi_bounds[0]], 
                    'hdi_high':[hdi_bounds[1],hdi_bounds[1]],
                    'param_std':[np.std(hc_dict[key]), np.std(pt_dict[key])]})
                df_out.append(df_tmp)
        # percentage of significant draws (ie bounds doesn't encompass 0)
        significant_pc = hdi_stats(key, bounds)
        significant_df.append(significant_pc)
    
    # save significant calculation
    df_sig = pd.DataFrame({'parameter': param_ls,
                        'significant_percent': significant_df},index=None)
    save_dir = './figs/'+model_name+'/'
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    df_sig.to_csv('./figs/'+model_name+'/significance_pc.csv')
    # save df_out
    out = pd.concat(df_out)
    out.to_csv('./figs/'+model_name+'/params.csv',index=None)

def comp_hdi(model_name, param_ls, sort=True, draw_idx=50, draws=1000, seed=123):
    """
    compare hdi by drawing simulations
    """
    # define sim dir
    output_dir = './tmp_output/'+model_name+'_sim/'
    np.random.seed(seed)
    significant_df = []
    # find sim results in groups
    for key in param_ls:
        bounds = []
        c = 0
        # random compare n draws
        for comp in range(1,draws):
            # load MCMC traces with matching seeds (not number of subjects)
            hc_file = os.path.join(output_dir, 'hc_sim_'+str(int(np.random.randint(0,draw_idx,1)))+'.pkl')
            pt_file = os.path.join(output_dir, 'pt_sim_'+str(int(np.random.randint(0,draw_idx,1)))+'.pkl')

            if os.path.isfile(hc_file) and os.path.isfile(pt_file):
                # print(hc_file, pt_file)
                with open(hc_file, 'rb') as hc:
                    hc_dict = pickle.load(hc)
                with open(pt_file, 'rb') as pt:
                    pt_dict = pickle.load(pt)

                # calculate lower bounds of simulation using difference
                hdi_bounds = hdi_diff(key, hc_dict, pt_dict)
                # store hdi bounds
                bounds.append(hdi_bounds)
                c += 1
                # store sum dict
                if sort:
                    if comp == 1:
                        hc_sum = np.sort(hc_dict[key])
                        pt_sum = np.sort(pt_dict[key])
                        diff_sum = np.sort(hc_dict[key] - pt_dict[key])
                    else:
                        hc_sum += np.sort(hc_dict[key])
                        pt_sum += np.sort(pt_dict[key])
                        diff_sum += np.sort(hc_dict[key] - pt_dict[key])
                else:
                    if comp == 1:
                        hc_sum = hc_dict[key]
                        pt_sum = pt_dict[key]
                        diff_sum = hc_dict[key] - pt_dict[key]
                    else:
                        hc_sum += hc_dict[key]
                        pt_sum += pt_dict[key]
                        diff_sum += hc_dict[key] - pt_dict[key]
        # percentage of significant draws (ie bounds doesn't encompass 0)
        significant_pc = hdi_stats(key, bounds)
        significant_df.append(significant_pc)
        # plot hdi 
        plot_hdi_groups(model_name, key, hc_sum/float(draws), pt_sum/float(draws), sort)
        # plot hdi 
        plot_hdi_diff(model_name, key, diff_sum/float(draws), sort)
    # save significant calculation
    df_sig = pd.DataFrame({'parameter': param_ls,
                        'significant_percent': significant_df},index=None)
    df_sig.to_csv('./figs/'+model_name+'/significance_pc.csv')

def hdi_stats(key, hdi_bounds):
    """calculate hdi bounds stats"""
    # print(hdi_bounds)
    dfb = pd.DataFrame(hdi_bounds)
    dfb.columns = ['lower', 'upper']
    significant_sim, significant_neg, significant_pos = 0, 0, 0
    for _,row in dfb.iterrows():
        if np.sign(row['lower']) == np.sign(row['upper']) and row['lower']<0:
            significant_neg += 1
        elif np.sign(row['lower']) == np.sign(row['upper']) and row['lower']>=0:
            significant_pos += 1
    if significant_neg > significant_pos:
        significant_sim = significant_neg
    else:
        significant_sim = significant_pos
    significant_pc = significant_sim/dfb.shape[0]*100.
    print(f'{key} significant %: {significant_pc:.2f}')
    return significant_pc
        

def hdi_diff(param_key, hc_dict, pt_dict):
    """calculate difference between patient and control group mean param traces"""
    # calculate difference between groups
    param_diff = hc_dict[param_key] - pt_dict[param_key]
    # calculate hdi
    hdi_bounds = hdi(param_diff)
    # print(param_key+' hdi range: ', hdi_bounds)
    return hdi_bounds

def plot_hdi_groups(model_name, param_key, hc_dict, pt_dict, sort, credible_interval=0.94, point_estimate='mean', bins='auto', round_to=2):
    """plotting param posterior in groups"""
    # kwargs.setdefault('color', 'black')
    # x = [hc_dict[param_key], pt_dict[param_key]]
    x = [hc_dict, pt_dict]
    leg = [['Control'], ['Patient']]
    colors = ['blue','green']
    fig, axes = plt.subplots(2,1, sharex=True)
    for i in range(2):
        ax = axes[i]
        az.plot_posterior(x[i],
                        ax=ax,
                        kind='hist',
                        credible_interval=credible_interval,
                        point_estimate=point_estimate,
                        bins=bins,
                        round_to=round_to,
                        color=colors[i])
        if i ==0:
            title = param_key
        else:
            title = ''
        ax.set_title(title)
        ax.legend(leg[i])
    # save fig
    save_dir = './figs/'+model_name+'/'
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    if sort:
        save_name = param_key+'_hdi_sorted.png' 
    else:
        save_name = param_key+'_hdi.png'
    fig = ax.get_figure()
    fig.savefig(save_dir+save_name)

def plot_hdi_diff(model_name, param_key, diff_dict, sort, credible_interval=0.94, point_estimate='mean', bins='auto', round_to=2):
    """plotting param posterior in diff"""
    # kwargs.setdefault('color', 'black')
    # x = [hc_dict[param_key], pt_dict[param_key]]
    x = diff_dict
    fig, ax = plt.subplots(1,1, sharex=True)
    az.plot_posterior(x,
                    ax=ax,
                    kind='hist',
                    credible_interval=credible_interval,
                    point_estimate=point_estimate,
                    bins=bins,
                    round_to=round_to,
                    color='black')
    title = param_key
    ax.set_title(title)
    ax.legend(['Control-Patient'])
    # save fig
    save_dir = './figs/'+model_name+'/'
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    if sort:
        save_name = param_key+'_hdi_diff_sorted.png'
    else:
        save_name = param_key+'_hdi_diff.png'
    fig = ax.get_figure()
    fig.savefig(save_dir+save_name)

def plot_violin_params(csv_params, model_name, n_perm):
    """plot violin of param means"""
    import seaborn as sns
    sns.set_theme(style="whitegrid")
    df = pd.read_csv(csv_params)
    df['parameter'] = df['param'].str.slice(3,)
    param_ls = np.unique(df['parameter'])
    n_param = len(param_ls)
    if model_name=='motoradapt':
        fig, ax = plt.subplots(1,n_param,figsize=(2,2.5))
        leg_box = (-1,-0.1)
    elif model_name=='generalise':
        fig, ax = plt.subplots(1,n_param,figsize=(4.5,2.5))
        leg_box = (-2,-0.1)
    else:  
        fig, ax = plt.subplots(1,n_param,figsize=(4,2.5))
        leg_box = (-2, -0.1)
    for n in range(n_param):
        g= sns.violinplot(data=df[df['parameter']==param_ls[n]], x="parameter", y="param_mean", hue="group", split=True, inner="quart", linewidth=1,palette={"patient": "b", "control": ".85"}, ax=ax[n])
        sns.despine(left=True)
        g.set(ylabel=None)
        ax[n].get_legend().remove()
        ax[n].tick_params(axis='y', labelsize=8) 
        if model_name=='motoradapt' and n==2:
            g.set(yticklabels=[])
        g.set(xlabel=None)
    # plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.legend(loc='upper center', bbox_to_anchor=leg_box,
          fancybox=True, shadow=True, ncol=2)
    # if model_name == 'bandit':
    #     # plt.suptitle(f'Simulated data fitted model parameter mean distribution \n ({model_name} task, reward+punishment lapse model)')
    # elif model_name == 'generalise':
    #     # plt.suptitle(f'Simulated data fitted model parameter mean distribution \n ({model_name} task, perceptual+value generalisation model)')
    # elif model_name == 'motoradapt':
    #     # plt.suptitle(f'Simulated data fitted model parameter mean distribution \n (moto adaptation task, single state space model)')
    # save fig
    save_dir = './figs/'+model_name+'/'
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    save_name = 'param_mean.png'
    fig.savefig(save_dir+save_name,bbox_inches='tight',pad_inches=0)

def plot_hdi_permutations(csv_params, model_name, n_perm):
    """plot hdi for all permutations"""
    import seaborn as sns
    sns.set_theme(style="whitegrid")
    df = pd.read_csv(csv_params)
    param_ls = np.unique(df['param'])
    for param in param_ls:
        # fig, ax = plt.subplots(figsize=(5,4))
        fig, ax = plt.subplots(figsize=(3,2.5))
        df_tmp = df[(df['param']==param) & (df['group']=='control')]
        df_tmp_sort = df_tmp.sort_values(by=['hdi_high'],ascending=True)
        fill_indicator = sum(df_tmp_sort['hdi_high']>0)<sum(df_tmp_sort['hdi_high']<=0)
        cnt = 0
        fill_range = []
        for _, row in df_tmp_sort.iterrows():
            plt.vlines(x = cnt, ymin = row['hdi_low'], ymax = row['hdi_high'], colors = 'black', label = 'HDI', linewidth=0.5)
            if fill_indicator: # <0 majority
                if row['hdi_high']>=0:
                    fill_range.append(cnt)
            else:
                if row['hdi_high']<0:
                    fill_range.append(cnt)
            cnt += 1
        plt.fill_between(fill_range, min(df_tmp_sort['hdi_low']), max(df_tmp_sort['hdi_high']), facecolor='gray', alpha=0.5) 
        plt.xlabel(param[3:], fontsize=10)
        plt.ylabel('Control>Patient 95% HDI', fontsize=10)
        plt.yticks(fontsize=8)
        plt.xticks(fontsize=8) 
        # plt.suptitle(f'Control>Patient 95% HDI for parameter estimation \n ({model_name} task, each line represents 1 simulation)')
        plt.title(f'Parameter 95% HDI ({model_name} task)', fontsize=10)
        # save fig
        save_dir = './figs/'+model_name+'/'
        if not os.path.isdir(save_dir):
            os.mkdir(save_dir)
        save_name = f'param_hdi_{param}.png'
        fig.savefig(save_dir+save_name, bbox_inches='tight',pad_inches=0)

if __name__ == "__main__":
    # arg
    model_name = sys.argv[1] # which model sims to compare

    if model_name == 'bandit':
        param_ls = ['mu_Arew', 'mu_Apun', 'mu_R', 'mu_P', 'mu_xi']
    elif model_name == 'bandit_combined':
        param_ls = ['mu_Arew', 'mu_Apun', 'mu_R', 'mu_P', 'mu_xi', 'mu_d']
    elif model_name == 'generalise':
        param_ls = ['mu_sigma_a', 'mu_sigma_n', 'mu_eta', 'mu_kappa', 'mu_beta', 'mu_bias']
    elif model_name == 'motoradapt':
        param_ls = ['mu_A', 'mu_B', 'mu_sig']
    else:
        print('model must be bandit or generalise.')
    n_perm = 1000
    comp_hdi_mean(model_name, param_ls, sort=False, draw_idx=30, draws=n_perm)
    plot_violin_params(f'./figs/{model_name}/params.csv', model_name, n_perm=n_perm)
    plot_hdi_permutations(f'./figs/{model_name}/params.csv', model_name, n_perm)