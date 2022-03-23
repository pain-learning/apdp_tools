"""
Transform task Pavlovia data to csv files suitable for fitting
"""

from asyncore import file_dispatcher
import os, sys
import pandas as pd
from pandas.errors import EmptyDataError

def load_pavlovia(task_name, task_dir, n_trials, output_dir='./transformed_data'):
    """load pavlovia repo of the task and the data within"""
    # check if this has data dir
    
    task_data_dir = os.path.join(task_dir, 'data')
    if not os.path.isdir(task_data_dir):
        raise ValueError('No Pavlovia data directory found.')
    else:
        # load pavlovia data in df, excluding data from pc tests ('_xxx.csv'), also csv shouldn't be empty
        f_list = []
        for f in os.listdir(task_data_dir):
            f_path = os.path.join(task_data_dir, f)
            # print(f_path)
            if not f.startswith('_') and f.endswith('.csv'):

                try:
                    # print(f'{f}')
                    tmp = pd.read_csv(f_path)                    
                except:
                    print(f'{f} is empty, skipping.')
                    continue
                else:
                    try:
                        # tmp[['trials.thisTrialN']]
                        if (n_trials != -1) and (len(tmp[['trials.thisTrialN']].dropna()) !=n_trials): #check if data is complete
                            print(f'{f} is not complete, skipping')                        
                        else:
                            print(f'{f} is ok')
                            print (len(tmp[['trials.thisTrialN']].dropna()))
                            f_list.append(f)
                    except:
                        print(f'{f} is not complete, skipping')                        

        # f_list = [f for f in os.listdir(task_data_dir) if not f.startswith('_') and f.endswith('.csv')]
        # sort data file according to date
        f_list_sorted = sorted(f_list, key=split_filename)
    print(f_list_sorted)
    
    # load to panda df
    df_ls = []
    df_pid_subjID = []
    id_count = 1
    for f in f_list_sorted: #create unique subject ID + renumber trials
        csv_path = os.path.join(task_data_dir, f)
        df = pd.read_csv(csv_path)
        df['subjID'] = id_count
        df.loc[~df['trials.thisTrialN'].isna(),'trials.thisTrialN'] = (list(range(1,1+len(df.loc[~df['trials.thisTrialN'].isna(),'trials.thisTrialN'])))) #renumber trials
        df_ls.append(df)
        df_pid_subjID.append(df[['subjID','participant']])
        id_count += 1
    df_out = pd.concat(df_ls)
    
    # mapping from PID to subjID
    df_pid_subjID = pd.concat(df_pid_subjID)
    df_pid_subjID.drop_duplicates('subjID',inplace=True)
    df_pid_subjID.fillna('XSUB',inplace=True)
    pidsub_path = os.path.join(output_dir, task_name+'_PID_subjID.txt')
    df_pid_subjID.to_csv(pidsub_path, index=None, sep='\t')
    
    # transform data and save
    if 'generalise' in task_name:
        transform_generalise(df_out, output_dir=output_dir)
    elif 'bandit3arm' in task_name:
        transform_bandit3arm(df_out, output_dir=output_dir)
    elif 'circlemotor' in task_name:
        transform_motorcircle(df_out, output_dir=output_dir)
    else:
        raise ValueError('Data transform for the task not yet implemented. \nPlease use the following: bandit4arm, generalise, circlemotor.\nAlternatively, you can write your own data conversion pipeline following the example functions here, to match the data format in simulated data.')

def split_filename(f_name):
    """split string of data csv filename"""
    from datetime import datetime
    fn_split = f_name.split('_')
    hrs = fn_split[-1].split('.')[0]
    ts = fn_split[-2] + '_' + hrs
    ts_dt = datetime.strptime(ts, "%Y-%m-%d_%Hh%M")
    return ts_dt

def transform_generalise(df, output_dir):
    """transform df into compatible csv for the generalisation task"""
    # df=df_out
    print(df.columns)
    
    # extracting useful cols
    df_sub = df[['subjID', 'trials.thisTrialN','cue', 'choice', 'rt', 'outcome']]
    
    # rename cols
    df_sub.rename(columns={'trials.thisTrialN': 'trial'}, inplace=True)
    
    # drop na
    df_sub.dropna(subset=['trial'], inplace=True)
 
    # convert all to int
    df_sub.fillna(999,inplace=True)
    df_sub[['trial','choice','cue','outcome']] = df_sub[['trial','choice','cue','outcome']].astype(int)

    # saving tsv
    output_path = os.path.join(output_dir, 'generalise_data.txt')
    df_sub.to_csv(output_path, index=None, sep='\t')
    
    # print status
    print('\ngeneralisation task data conversion done.')

def transform_bandit3arm(df, output_dir):
    """transform df into compatible csv for the bandit3arm task"""
    # df=df_out
    print(df.columns)
    
    # extracting useful cols
    df_sub = df[['subjID', 'trials.thisTrialN', 'choice','rt', 'gain','loss']]
    
    # rename cols
    df_sub.rename(columns={'trials.thisTrialN': 'trial'}, inplace=True)
    
    # drop na
    df_sub.dropna(subset=['trial'], inplace=True)

    # convert all to int
    df_sub.fillna(999,inplace=True)
    df_sub[['trial','choice','gain','loss']] = df_sub[['trial','choice','gain','loss']].astype(int)

    # saving tsv
    output_path = os.path.join(output_dir, 'bandit3arm_data.txt')
    df_sub.to_csv(output_path, index=None, sep='\t')
    
    # print status
    print('\nbandit task data conversion done.')

def transform_motorcircle(df, output_dir):
    """transform df into compatible csv for circlemotor task"""
    print(df.columns)
    # extracting useful cols
    df_sub = df[['subjID', 'trials.thisTrialN','touch_resp.x', 'touch_resp.y', 'solid_y', 'empty_y']]
    # df_sub = df[['subjID', 'trials.thisTrialN','touch_resp.x', 'touch_resp.y']]
    # rename cols
    df_sub.rename(columns={'trials.thisTrialN': 'trial', 'touch_resp.x': 'x', 'touch_resp.y': 'y'}, inplace=True)
    # df_sub.rename(columns={'trials.thisTrialN': 'trial', 'touch_resp.x': 'x', 'touch_resp.y': 'y'}, inplace=True)
    # drop na
    df_sub.dropna(subset=['x'], inplace=True)
    # saving tsv
    output_path = os.path.join(output_dir, 'circlemotor_data.txt')
    df_sub.to_csv(output_path, index=None, sep='\t')
    # print status
    print('\ncirclemotor task data conversion done.')

# run
if __name__ == "__main__":
    # parsing cl arguments
    task_name = sys.argv[1] # name of task
    task_dir = sys.argv[2] # path to the task repo
    try: 
        n_trials = int(sys.argv[3]) # how many trials are expected (check if data is complete)
    except IndexError:
        n_trials = -1
      
    
    # make outputdir
    output_dir = './transformed_data'
    output_task_dir = os.path.join(output_dir, task_name)
    if not os.path.isdir(output_task_dir):
        os.mkdir(output_task_dir)

    # load data and convert
    load_pavlovia(task_name, task_dir, n_trials, output_dir=output_task_dir)