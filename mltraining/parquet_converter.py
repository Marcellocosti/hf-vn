import pandas as pd
import numpy as np
import argparse

import yaml
import awkward as ak
import uproot
import os
import sys
script_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(script_dir, '..', 'utils'))
from utils import logger, get_centrality_bins

def converter(file, table, query, cols_to_keep, cols_to_add, label):

    # Pick data tree, convert to pandas dataframe, and save as parquet
    with upopen(file) as f:
        keys = f.keys()

        matched_keys = [key for key in keys if table in key and len(key.split('/')) == 2]
        dfs = []
        for key in matched_keys:
            logger(f"Reading {key}", level='INFO')
            array = f[key].arrays(library="ak")  # get awkward array
            df = ak.to_dataframe(array)          # convert to pandas DataFrame
            if (len(df) == 0):
                logger(f"Empty DataFrame for key: {key}", level='WARNING')
            dfs.append(df)

        # Print all columns before filtering
        if cols_to_keep is not None:
            dfs = [df[cols_to_keep] for df in dfs]
        full_df = pd.concat(dfs, ignore_index=True)

        # Add new columns
        if cols_to_add is not None:
            for col in cols_to_add:
                print(f"Adding new column: {col['name']} with formula: {col['formula']}")
                full_df[col['name']] = full_df.eval(col['formula'])

    # Create parquet file name
    parquet_file = file.replace('.root', f'_{table}_{label}.parquet')
    if query != "":
        print(f"Applying query: {query}")
        queried_df = full_df.query(query)
        queried_df.to_parquet(parquet_file)
        logger(f"Saved {label} parquet file: {parquet_file} with entries {len(queried_df)}", level='INFO')
    else:
        full_df.to_parquet(parquet_file)
        logger(f"Saved {label} parquet file: {parquet_file} with entries {len(full_df)}", level='INFO')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic datasets for ML training.")
    parser = argparse.ArgumentParser(description='Arguments to pass')
    parser.add_argument('cfgFileName', metavar='text', default='cfg_prepare_inputs.yml', help='config file name for ml')
    args = parser.parse_args()

    _, (cent_min, cent_max) = get_centrality_bins(args.cent_class)
    if args.data != "AO2D_data.root":
        converter(args.data, args.table, cent_min, cent_max, queries=["fM > 1.6 and fM < 2.5"], labels=["data"])
    if args.mc != "AO2D_mc.root":
        converter(args.mc, args.table, cent_min, cent_max, queries=["fM > 1.6 and fM < 2.5 and fOriginMcRec == 1", "fM > 1.6 and fM < 2.5 and fOriginMcRec == 2"], labels=["prompt", "fd"])
