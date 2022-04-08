"""
fit motor circle task with external data (not simulated)
"""
import sys, os
import numpy as np
import pandas as pd
import stan
import arviz as az
from matplotlib import pyplot as plt
import seaborn as sns

sys.path.append('.')
from simulations.sim_generalise_gs import generalise_gs_preprocess_func
from data_fit.fit_bandit3arm_combined import comp_hdi_mean_data

def extract_ind_results(df,pars_ind,data_dict):
    out_col_names = []
    out_df = np.zeros([2,len(pars_ind)*2])
    i=0
    for ind_par in pars_ind:     
        pattern = r'\A'+ind_par+r'.\d+'

        out_col_names.append(ind_par+'_mean')
        out_col_names.append(ind_par+'_std')
        
        mean_val=df.iloc[:,df.columns.str.contains(pattern)].mean(axis=0).to_frame()
        std_val=df.iloc[:,df.columns.str.contains(pattern)].std(axis=0).to_frame()

        out_df[:,2*i:2*(i+1)] = np.concatenate([mean_val.values,std_val.values],axis=1)
        i+=1
        
    out_df = pd.DataFrame(out_df,columns=out_col_names)
    
    beh_col_names = ['total','avg_rt','std_rt']
    total_np = 600+data_dict['choice'].sum(axis=1,keepdims=True)*(-2)+data_dict['outcome'].sum(axis=1,keepdims=True)*(10)
    avg_rt_np = data_dict['rt'].mean(axis=1,keepdims=True)
    std_rt_np =  data_dict['rt'].std(axis=1,keepdims=True)
    beh_df = pd.DataFrame(np.concatenate([total_np,avg_rt_np,std_rt_np],axis=1),columns=beh_col_names)
    out_df = beh_df.join(out_df)  
        
    return out_df 

def plot_violin_params(csv_params, model_name):
    """plot violin of param means"""
    csv_params = f'./data_output/generalise_mydata/param_statAB.csv'
    model_names = 'generalise'
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
    save_name = 'param_mean'+'_'+str(num_trial)+'_'+str(num_sj)+'.png'
    fig.savefig(save_dir+save_name,bbox_inches='tight',pad_inches=0)

if __name__ == "__main__":
    try:
        groups_comp = sys.argv[1]
        groups_comp = groups_comp.split(",")
    except IndexError:
        # groups_comp = ['None']
        groups_comp = ['']
        
    groups_comp=['A','B']
   # parse data
    # txt_path = f'./transformed_data/circlemotor/circlemotor_data0.txt'
    txt_path = f'./transformed_data/generalise/generalise_data.txt'
    data_dict = generalise_gs_preprocess_func(txt_path)#, task_params=task_params)
    model_code = open('./models/generalise_gs.stan', 'r').read() # moved to y changes
    pars_ind = ['sigma_a', 'sigma_n', 'eta', 'kappa', 'beta', 'bias']
    pars = ['mu_sigma_a', 'mu_sigma_n', 'mu_eta', 'mu_kappa', 'mu_beta', 'mu_bias']
    
    for g in groups_comp:
        group_value = data_dict['group']
        print('Group: '+g)
        if not g=='':
            group_bool = [i for i,x in enumerate([g == val for val in data_dict['group']]) if x]
            group_value = data_dict['group'][group_bool]
            data_dict_gr = {}
            for key, value in data_dict.items():
                if key not in ['N','T','group']:
                    data_dict_gr[key] = value[group_bool]
                elif key not in ['group']:
                    data_dict_gr[key] = value
                else:
                    continue
        else:
            data_dict_gr = data_dict
            data_dict_gr.pop('group')
        
        data_dict_gr['N'] = data_dict_gr['rt'].shape[0]
        
        # fit stan model
        posterior = stan.build(program_code=model_code, data=data_dict_gr)
        fit = posterior.sample(num_samples=10, num_chains=1)
        df = fit.to_frame()  # pandas `DataFrame, requires pandas
        data_dict_gr['group'] = group_value
        
        # individual results
        df_ind = extract_ind_results(df,pars_ind,data_dict_gr)
        subjID_df=pd.DataFrame((data_dict_gr['subjID'],data_dict_gr['group'])).transpose()
        subjID_df.columns = ['subjID','group']
        df_ind = subjID_df.join(df_ind)
    
        print(df['mu_sigma_a'].agg(['mean','var']))
        print(df['mu_beta'].agg(['mean','var']))


        # saving traces
        df_extracted = df[pars]
        save_dir = './data_output/generalise_mydata/'
        
        if not os.path.isdir(save_dir):
            os.mkdir(save_dir)
        sfile = save_dir + f'mydata_fit_group_trace'+g+'.csv'
        s_ind_file = save_dir + f'mydata_fit_ind_est'+g+'.csv'
        df_extracted.to_csv(sfile, index=None)
        df_ind.to_csv(s_ind_file, index=None)
        
    comp_hdi_mean_data('generalise', param_ls=pars, groups_comp=groups_comp)
    plot_violin_params(f'./data_output/generalise_mydata/param_statAB.csv',model_name = 'generalise')
        

