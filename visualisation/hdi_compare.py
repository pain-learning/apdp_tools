"""
compare hdi from pystan fit traces
"""
import os
import numpy as np
import pandas as pd
import arviz as az
from matplotlib import pyplot as plt
import seaborn as sns

def comp_hdi_mean(model_name, param_ls, sort=True, draw_idx=50, draws=1000, seed=123):
    """
    compare hdi by drawing simulations (trace means)
    """
    # define sim dir
    output_dir = './tmp_output/'+model_name+'_trace/'
    np.random.seed(seed)
    significant_df = []
    df_out = []
    # find sim results in groups
    for key in param_ls:
        bounds = []
        # random compare n draws
        for comp in range(1,draws):
            # load MCMC traces with matching seeds (not number of subjects)
            hc_file = os.path.join(output_dir, 'hc_sim_'+str(int(np.random.randint(0,draw_idx+1,1)))+'.csv')
            pt_file = os.path.join(output_dir, 'pt_sim_'+str(int(np.random.randint(0,draw_idx+1,1)))+'.csv')
            print(hc_file, pt_file)

            if os.path.isfile(hc_file) and os.path.isfile(pt_file):
                hc_dict = pd.read_csv(hc_file)
                pt_dict = pd.read_csv(pt_file)

                # calculate lower bounds of simulation using difference
                hdi_bounds = hdi_diff(key, hc_dict, pt_dict)
                print(hdi_bounds)
                # store hdi bounds
                bounds.append(hdi_bounds)
                # store mean
                df_tmp = pd.DataFrame({
                    'param':[key,key], 
                    'param_mean':[np.mean(hc_dict[key]), np.mean(pt_dict[key])],
                    'group':['control','patient'],
                    'hdi_low':[min(hdi_bounds),min(hdi_bounds)], 
                    'hdi_high':[max(hdi_bounds),max(hdi_bounds)],
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
    hdi_bounds = hdi(param_diff.values)
    # print(param_key+' hdi range: ', hdi_bounds)
    return hdi_bounds

def hdi(x, credible_interval=0.94):
    """compute hdi given np.ndarray of samples and credible interval, return upper and lower bound of interval
    """
    return az.hdi(x, hdi_prob=credible_interval)

def plot_violin_params(csv_params, model_name, n_perm):
    """plot violin of param means"""
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
    # save fig
    save_dir = './figs/'+model_name+'/'
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    save_name = 'param_mean.png'
    fig.savefig(save_dir+save_name,bbox_inches='tight',pad_inches=0)

if __name__ == "__main__":
    model_name = 'motorcircle'
    param_ls = ['loss_sens', 'perturb']
    n_perm = 50
    comp_hdi_mean(model_name, param_ls, sort=False, draw_idx=5, draws=n_perm, seed=0)
    plot_violin_params(f'./figs/{model_name}/params.csv', model_name, n_perm=n_perm)