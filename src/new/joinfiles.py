import pandas as pd
import glob
import os

def combine_dataset_files(foldername):
    '''
    Get all csv files from folder datasetgroups/foldername, combine them and
    save combined df as a csv.
    '''
    dfs = read_csvs_from_folder(foldername)
    df = combine_datasets(dfs)
    nruns = list(df["run"])[-1] + 1
    df.to_csv(f'datasets/{foldername}_nruns={nruns}.csv', index=False)

def read_csvs_from_folder(foldername):
    folder_path = 'datasetgroups/' + foldername
    csv_files = glob.glob(os.path.join(folder_path, '*.csv'))
    dataframes = [pd.read_csv(file) for file in csv_files]
    return dataframes

def combine_datasets(dfs):
    '''
    Run numbers must be adjusted, because they start from zero in each dataset.
    The 'runs' columns must be offset by number of runs in previous files.
    '''
    run_offset = 0

    dfsnew = []
    for i, df in enumerate(dfs):
        if i > 0:
            df['run'] += run_offset

        run_offset += df['run'].nunique()
        dfsnew.append(df)

    combined_df = pd.concat(dfsnew, ignore_index=True)
    return combined_df

    
foldername = 'tiger_t=50_cs=5'
combine_dataset_files(foldername)
