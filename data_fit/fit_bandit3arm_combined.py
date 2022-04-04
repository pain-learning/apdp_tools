"""
fit motor circle task with external data (not simulated)
"""
import sys, os
import numpy as np
import pandas as pd
import stan
import nest_asyncio
nest_asyncio.apply()


sys.path.append('.')
from simulations.sim_bandit3arm_combined import bandit_combined_preprocess_func

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


if __name__ == "__main__":
   # parse data
    # txt_path = f'./transformed_data/circlemotor/circlemotor_data0.txt'
    txt_path = f'./transformed_data/bandit3arm/bandit3arm_data.txt'
    data_dict = bandit_combined_preprocess_func(txt_path)#, task_params=task_params)
    model_code = open('./models/bandit3arm_combLR_lapse_decay_b.stan', 'r').read()

    # fit stan model
    posterior = stan.build(program_code=model_code, data=data_dict)
    fit = posterior.sample(num_samples=10, num_chains=2)
    df = fit.to_frame()  # pandas `DataFrame, requires pandas
    
    # individual results
    pars_ind = ['Arew', 'Apun', 'R', 'P', 'xi','d']    
    df_ind = extract_ind_results(df,pars_ind,data_dict)
    subjID_df=pd.DataFrame(data_dict['subjID'],columns=['subjID'])
    df_ind = subjID_df.join(df_ind)
           
    print(df['mu_Arew'].agg(['mean','var']))
    print(df['mu_Apun'].agg(['mean','var']))

    # saving traces
    pars = ['mu_Arew', 'mu_Apun', 'mu_R', 'mu_P', 'mu_xi','mu_d']
    
    df_extracted = df[pars]    
    
    save_dir = './tmp_output/bandit3arm_combined_mydata_trace/'
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    sfile = save_dir + f'mydata_fit.csv'
    s_ind_file = save_dir + f'mydata_fit_ind.csv'
    df_extracted.to_csv(sfile, index=None)
    df_ind.to_csv(s_ind_file, index=None)
