"""
fit motor circle task with external data (not simulated)
"""
import sys, os
import numpy as np
import pandas as pd
import stan
import arviz as az
import nest_asyncio
nest_asyncio.apply()


sys.path.append('.')
from simulations.sim_bandit3arm_combined import bandit_combined_preprocess_func
from visualisation.hdi_compare import hdi, hdi_diff

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
    total_np = 100+data_dict['rew'].sum(axis=1,keepdims=True)+data_dict['los'].sum(axis=1,keepdims=True)
    avg_rt_np = data_dict['rt'].mean(axis=1,keepdims=True)
    std_rt_np =  data_dict['rt'].std(axis=1,keepdims=True)
    beh_df = pd.DataFrame(np.concatenate([total_np,avg_rt_np,std_rt_np],axis=1),columns=beh_col_names)
    out_df = beh_df.join(out_df)

    return out_df

def comp_hdi_mean_data(model_name,param_ls, groups_comp=None):
    """
    compare hdi by drawing simulations (trace means)
    """
    # model_name='bandit3arm_combined'
    # groups_comp=['A','B']
    # groups_comp=['']
    # param_ls = ['mu_Arew', 'mu_Apun', 'mu_R', 'mu_P', 'mu_xi','mu_d']
    # define sim dir
    output_dir = './data_output/'+model_name+'_mydata/'
    
    if groups_comp != ['']:
        gr1_file = os.path.join(output_dir,'mydata_fit_group_trace'+groups_comp[0]+'.csv')
        gr2_file = os.path.join(output_dir,'mydata_fit_group_trace'+groups_comp[1]+'.csv')
        
        gr1_dict = pd.read_csv(gr1_file)
        gr2_dict = pd.read_csv(gr2_file)
    else:
        gr_file = os.path.join(output_dir,'mydata_fit_group_trace.csv')
        gr_dict = pd.read_csv(gr_file)
    df_out = []
    bounds = []
    for key in param_ls:       
        if groups_comp != ['']:
            # calculate lower bounds of simulation using difference
            hdi_bounds = hdi_diff(key, gr1_dict, gr2_dict)
            # store hdi bounds
            bounds.append(hdi_bounds)
            # store mean
            df_tmp = pd.DataFrame({
                'param':[key,key],
                'param_mean':[np.mean(gr1_dict[key]), np.mean(gr2_dict[key])],
                'group':[groups_comp[0],groups_comp[1]],
                'hdi_low':[min(hdi_bounds),min(hdi_bounds)],
                'hdi_high':[max(hdi_bounds),max(hdi_bounds)],
                'param_std':[np.std(gr1_dict[key]), np.std(gr2_dict[key])]})
        else:
            hdi_bounds = hdi(gr_dict[key].values)
            bounds.append(hdi_bounds)
            df_tmp = pd.DataFrame({ 
                'param':[key],
                'param_mean':[np.mean(gr_dict[key])],
                'hdi_low':[min(hdi_bounds)],
                'hdi_high':[max(hdi_bounds)],
                'param_std':[np.std(gr_dict[key])]})
        df_out.append(df_tmp)
    df_out = pd.concat(df_out)
    df_out.to_csv(output_dir+'param_stat'+''.join(groups_comp)+'.csv',index=None)
    if groups_comp != ['']:
        df_hdi_diff = pd.DataFrame(bounds,columns=['hdi_low','hdi_high']) 
        df_hdi_diff = df_hdi_diff.join(pd.DataFrame(param_ls,columns=['param']) )
        df_hdi_diff.to_csv(output_dir+'hdi_diff_stat'+''.join(groups_comp)+'.csv',index=None)
    


if __name__ == "__main__":
   # parse data
    # txt_path = f'./transformed_data/circlemotor/circlemotor_data0.txt'
    try:
        groups_comp = sys.argv[1]
        groups_comp = groups_comp.split(",")
    except IndexError:
        # groups_comp = ['None']
        groups_comp = ['']

    # print(groups_comp)
    groups_comp = 'A,B'
    groups_comp=groups_comp.split(",")
    groups_comp = ['']


    txt_path = f'./transformed_data/bandit3arm/bandit3arm_data.txt'
    data_dict = bandit_combined_preprocess_func(txt_path)#, task_params=task_params)
    model_code = open('./models/bandit3arm_combLR_lapse_decay_b.stan', 'r').read()
    pars_ind = ['Arew', 'Apun', 'R', 'P', 'xi','d']
    pars = ['mu_Arew', 'mu_Apun', 'mu_R', 'mu_P', 'mu_xi','mu_d']
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

        print(df['mu_Arew'].agg(['mean','var']))
        print(df['mu_Apun'].agg(['mean','var']))


        # saving traces
        df_extracted = df[pars]
        save_dir = './data_output/bandit3arm_combined_mydata/'
        if not os.path.isdir(save_dir):
            os.mkdir(save_dir)
        sfile = save_dir + f'mydata_fit_group_trace'+g+'.csv'
        s_ind_file = save_dir + f'mydata_fit_ind_est'+g+'.csv'
        df_extracted.to_csv(sfile, index=None)
        df_ind.to_csv(s_ind_file, index=None)
        
    comp_hdi_mean_data('bandit3arm_combined', param_ls=pars, groups_comp=groups_comp)
