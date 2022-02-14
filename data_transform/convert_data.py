"""
Transform task Pavlovia data to csv files suitable for fitting
"""

from asyncore import file_dispatcher
import os, sys
import pandas as pd
from pandas.errors import EmptyDataError

def load_pavlovia(task_name, task_dir, output_dir='./transformed_data'):
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
            if not f.startswith('_') and f.endswith('.csv'):
                try:
                    tmp = pd.read_csv(f_path)
                except:
                    print(f'{f} is empty, skipping.')
                    continue
                else:
                    f_list.append(f)

        # f_list = [f for f in os.listdir(task_data_dir) if not f.startswith('_') and f.endswith('.csv')]
        # sort data file according to date
        f_list_sorted = sorted(f_list, key=split_filename)
    # print(f_list_sorted)
    # load to panda df
    df_ls = []
    id_count = 0
    for f in f_list_sorted:
        csv_path = os.path.join(task_data_dir, f)
        df = pd.read_csv(csv_path)
        df['subjID'] = id_count
        df_ls.append(df)
        id_count += 1
    df_out = pd.concat(df_ls)
    # transform data and save
    if 'generalise' in task_name:
        transform_generalise(df_out, output_dir=output_dir)
    elif 'bandit4arm' in task_name:
        transform_bandit4arm(df_out, output_dir=output_dir)
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
    print(df.columns)
    # extracting useful cols
    df_sub = df[['subjID', 'trials.thisTrialN','CS', 'touch_resp.time', 'outcome']]
    # rename cols
    df_sub.rename(columns={'trials.thisTrialN': 'trial', 'CS': 'cue', 'touch_resp.time': 'choice'}, inplace=True)
    # drop na
    df_sub.dropna(subset=['outcome'], inplace=True)
    # convert touch to choice
    df_sub.loc[df_sub['choice']>0, 'choice'] = 1
    df_sub.loc[df_sub['choice']!=1, 'choice'] = 0
    # convert all to int
    df_sub = df_sub.astype(int)
    # saving tsv
    output_path = os.path.join(output_dir, 'generalise_data.txt')
    df_sub.to_csv(output_path, index=None, sep='\t')
    # print status
    print('\ngeneralisation task data conversion done.')

def transform_bandit4arm(df, output_dir):
    """transform df into compatible csv for the bandit4arm task"""
    print(df.columns)
    # extracting useful cols
    df_sub = df[['subjID', 'trials.thisTrialN', 'touch_resp.clicked_name', 'corr']]
    # rename cols
    df_sub.rename(columns={'trials.thisTrialN': 'trial', 'touch_resp.clicked_name': 'cue'}, inplace=True)
    # drop na
    df_sub.dropna(subset=['cue'], inplace=True)
    # convert touch to choice, check to make sure it matches generated probs
    df_sub['cue'] = df_sub['cue'].map({
        'button_upper_left':    1,
        'button_upper_right':   2,
        'button_lower_left':    3,
        'button_lower_right':   4
    })
    # convert wins to gain col
    df_sub['gain'] = df_sub['corr'].map({
        0:0,    # nothing
        1:1,    # win
        11:1,   # win and loss
        -1:0    # loss
    })
    # convert loss to loss
    df_sub['loss'] = df_sub['corr'].map({
        0:0,    # nothing
        1:0,    # win
        11:-1,   # win and loss
        -1:-1    # loss
    })
    # removing corr col
    df_sub = df_sub.drop('corr', axis=1)
    # convert all to int
    df_sub = df_sub.astype(int)
    # saving tsv
    output_path = os.path.join(output_dir, 'bandit4arm_data.txt')
    df_sub.to_csv(output_path, index=None, sep='\t')
    # print status
    print('\nbandit task data conversion done.')

def transform_motorcircle(df, output_dir):
    """transform df into compatible csv for circlemotor task"""
    print(df.columns)
    # extracting useful cols
    df_sub = df[['subjID', 'trials.thisTrialN','touch_resp.x', 'touch_resp.y']]
    # rename cols
    df_sub.rename(columns={'trials.thisTrialN': 'trial', 'touch_resp.x': 'x', 'touch_resp.y': 'y'}, inplace=True)
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
    
    # make outputdir
    output_dir = './transformed_data'
    output_task_dir = os.path.join(output_dir, task_name)
    if not os.path.isdir(output_task_dir):
        os.mkdir(output_task_dir)

    # load data and convert
    load_pavlovia(task_name, task_dir, output_dir=output_task_dir)