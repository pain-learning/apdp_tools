"""
Transform generalisation task Pavlovia data for tools
"""

import os, sys
import pandas as pd

def load_pavlovia(task_dir, output_dir='./transformed_data'):
    """load pavlovia repo of the task and the data within"""
    # check if this has data dir
    task_data_dir = os.path.join(task_dir, 'data')
    if not os.path.isdir(task_data_dir):
        raise ValueError('No Pavlovia data directory found.')
    else:
        # load pavlovia data in df
        f_list = [f for f in os.listdir(task_data_dir) if 'PARTICIPANT' in f and f.endswith('.csv')]
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
    transform_generalise(df_out, output_dir=output_dir)

def split_filename(f_name):
    """split string of data csv filename"""
    from datetime import datetime
    fn_split = f_name.split('_')
    hrs = fn_split[4].split('.')[0]
    ts = fn_split[3] + '_' + hrs
    ts_dt = datetime.strptime(ts, "%Y-%m-%d_%Hh%M")
    return ts_dt

def transform_generalise(df, output_dir):
    """transform df into compatible csv"""
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

if __name__ == "__main__":
    # parsing cl arguments
    task_dir = sys.argv[1] # path to the task repo
    
    # make outputdir
    output_dir = './transformed_data'
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    # load data
    load_pavlovia(task_dir, output_dir=output_dir)