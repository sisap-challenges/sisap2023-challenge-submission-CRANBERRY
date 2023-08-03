import argparse
import h5py
import numpy as np
import os
from pathlib import Path
import subprocess
from urllib.request import urlretrieve
import time 

def download(src, dst):
    if not os.path.exists(dst):
        os.makedirs(Path(dst).parent, exist_ok=True)
        print('downloading %s -> %s...' % (src, dst))
        urlretrieve(src, dst)

def prepare(root_data_folder, kind, size):
    dataset_url = "https://sisap-23-challenge.s3.amazonaws.com/SISAP23-Challenge"
    data_file_dict = {
        "dataset_orig": [os.path.join(root_data_folder, 'Dataset', 'Dataset'), f"laion2B-en-clip768v2-n={size}.h5"],
        "dataset": [os.path.join(root_data_folder, 'Dataset', 'Dataset'), f"laion2B-en-clip768v2-n={size}.h5"],
        "query_orig": [os.path.join(root_data_folder, 'Dataset', 'Query'), f"private-queries-gold-10k-clip768v2.h5"],
        "query": [os.path.join(root_data_folder, 'Dataset', 'Query'), f"private-queries-gold-10k-clip768v2.h5"]
    }

    for version, file_spec in data_file_dict.items():
        result_file_path = os.path.join(file_spec[0], file_spec[1])
        if version.startswith("query") or (not os.path.exists(result_file_path)):
            download(f"{dataset_url}/{file_spec[1]}", result_file_path)
        else:
            print(f"File '{result_file_path}' already exists, skipping download.")

    return data_file_dict

def store_results(dst, algo, kind, D, I, buildtime, querytime, params, size):
    os.makedirs(Path(dst).parent, exist_ok=True)
    f = h5py.File(dst, 'w')
    f.attrs['algo'] = algo
    f.attrs['data'] = kind
    f.attrs['buildtime'] = buildtime
    f.attrs['querytime'] = querytime
    f.attrs['size'] = size
    f.attrs['params'] = params
    f.create_dataset('knns', I.shape, dtype=I.dtype)[:] = I
    f.create_dataset('dists', D.shape, dtype=D.dtype)[:] = D
    f.close()

def run(root_data_folder, kind, key, size="100K", k=30, buildindex=True):
    
    # Download the dataset and query files
    data_file_dict = prepare(root_data_folder, kind, size)
    
    dataset_orig_path = os.path.join(data_file_dict['dataset_orig'][0], data_file_dict['dataset_orig'][1])
    dataset_path = os.path.join(data_file_dict['dataset'][0], data_file_dict['dataset'][1])
    query_orig_path = os.path.join(data_file_dict['query_orig'][0], data_file_dict['query_orig'][1])
    query_path = os.path.join(data_file_dict['query'][0], data_file_dict['query'][1])

    # Run Java-based implementation (building the index + searching)...
    print(f"running Java-based implementation (building the index + searching)...")
    dataset_size_dict = {"100K": "100000", "300K": "300000", "10M": "10000000", "30M": "30000000", "100M": "100000000"}
    print(f'* params: {dataset_orig_path}, {dataset_path}, {query_orig_path}, {query_path}, {dataset_size_dict[size]}, {"true" if buildindex else "false"}, {str(k)}')
    start_time = time.time()
    subprocess.check_output(['java', '-Xmx500g', '-jar', os.path.join(os.getcwd(), 'VMTrials', 'target', 'VMTrials-1.0-SNAPSHOT-jar-with-dependencies.jar'), dataset_orig_path, dataset_path, query_orig_path, query_path, dataset_size_dict[size], "true" if buildindex else "false", str(k)], universal_newlines=True)
    print(f"done in {time.time() - start_time}s")

    # Convert output params to dictionary
    output_params_file_path = os.path.join(root_data_folder, 'Result', f"{data_file_dict['dataset_orig'][1]}_{data_file_dict['query_orig'][1]}_run_params.csv")
    with open(output_params_file_path, 'r') as f:
        output_params = f.read()
        output_params = dict(item.split(":") for item in output_params.strip().split(";"))
        print(f"output_params: {output_params}")
    algo = 'CRANBERRY'
    buildtime = output_params['buildtime']
    querytime = output_params['querytime']
    if 'params' in output_params:
        params = output_params['params']
    else:
        params = ''
    
    # Convert CSV-based results to .h5 format
    import shutil
    algorithm_result_dir = os.path.join(root_data_folder, 'Result')
    result_dir = 'result'
    result_dst = os.path.join(result_dir, kind, size, f'{algo}.h5')
    if not os.path.exists(result_dir):
        os.makedirs(result_dir, exist_ok=True)
    result_file = f"{data_file_dict['dataset_orig'][1]}_{data_file_dict['query_orig'][1]}.csv"
    result_file_path = os.path.join(result_dir, result_file)
    print(f'processing result file: {result_file_path}')
    shutil.copyfile(os.path.join(algorithm_result_dir, result_file), result_file_path)

    # Parse the result file with Pandas
    import pandas as pd
    df = pd.read_csv(result_file_path, header=None, skiprows=0, sep=';')
    print(f'* result file shape: {df.shape}')
    #df = df.drop(df.columns[0], axis=1) # Remove the first column with row numbers
    I = df.copy().applymap(lambda x: x.split(':')[0]).astype(int).to_numpy()
    D = df.copy().applymap(lambda x: x.split(':')[1]).astype(float).to_numpy()
    print(f"* shapes of 'knns' and 'dists' structures: {I.shape}; {D.shape}")
    
    store_results(result_dst, algo, kind, D, I, buildtime, querytime, params, size)
    print(f'.h5 result file successfully created: {result_dst}')

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--size",
        default="100K"
    )
    parser.add_argument(
        "--k",
        default=30,
    )
    parser.add_argument(
        "--buildindex",
        default=True,
    )
    args = parser.parse_args()
    assert args.size in ["100K", "300K", "10M", "30M", "100M"]

    root_data_folder = "Similarity_search"
    run(root_data_folder, "clip768v2", "emb", size=args.size, k=args.k, buildindex=args.buildindex)
