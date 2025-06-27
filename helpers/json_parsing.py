import pandas as pd
import numpy as np
import ijson
import os
import time

inpath = 'raw_data/'
outpath = 'processed_json'
json_files = [f for f in os.listdir(inpath) if f.endswith('.json')]
json_files

def process_json(file,inpath,outpath):
    file_name = os.path.splitext(file)[0]
    file_path = os.path.join(inpath,file)
    j = 1
    k = ijson.items(open(file_path, 'rb'),'item')
    df_list = []
    for i, item in enumerate(k):
        cname = item[0]
        for sid,sess in item[1].items():
            for sessid, act in sess.items():
                actions = [act[i][0] for i in range(0,len(act))]
                timestamps= [act[i][1] for i in range(0,len(act))]
                temp_df = pd.DataFrame({'course_id':[cname]*len(actions),
                         'username':[sid]*len(actions),
                         'session_id': [sessid]*len(actions),
                          'action' : actions,
                          'time' : timestamps
                         }
                        )
                df_list.append(temp_df)
        if i % 100 ==0:
            outfile = f'{file_name}_{j}.csv'
            out = os.path.join(outpath,outfile)
            intermediary_df = pd.concat(df_list)
            intermediary_df.to_csv(out,index=False)
            print(f'wrote: {out}')
            df_list = []
            j +=1
    j +=1
    outfile = f'{file_name}_{j}.csv'
    out = os.path.join(outpath,outfile)
    intermediary_df = pd.concat(df_list)
    intermediary_df.to_csv(out,index=False)
    print(f'wrote: {out}')

def process_csv(file):
    chunksize = 100000
    tfr = pd.read_csv(file, chunksize=chunksize, iterator=True)
    df = pd.concat(tfr, ignore_index=True)
    df['time'] = pd.to_datetime(df['time'])
    df = df.sort_values(['username','course_id','time'])
    df_smaller = df.groupby(['username','course_id']).nth([0,-1]).reset_index()
    return df_smaller

for file in json_files:
    start = time.time()
    print(f'starting {file}')
    process_json(file,inpath,outpath)
    stop = time.time()
    duration = stop-start
    print(f'{file} took {duration/60} minutes to process')

path = 'processed_json'
csv_files = [f for f in os.listdir(path) if f.endswith('.csv')]

temp_dfs = []
for i, file in enumerate(csv_files):
    temp_dfs.append(process_csv(os.path.join(path,file)))

full_df = pd.concat(temp_dfs)
full_df = full_df.sort_values(['username','course_id','time'])
full_df = full_df.assign(days_spent=full_df.groupby(['username','course_id']).time.apply(lambda x: x - x.iloc[0]))
small_df = full_df.groupby(['username','course_id']).nth([-1]).reset_index()
