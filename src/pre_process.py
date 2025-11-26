'''
This script is used to pre-process a/multi large AnRes.root for the BDT training:
    - split the input by pT
    - obtain the sigma from prompt enhance sample
python3 pre_process.py config.yml AnRes_1.root AnRes_2.root --pre --sigma  
'''
import os
import sys
import yaml
import numpy as np
import array
from ROOT import TFile, TObject
import argparse
import gc
from alive_progress import alive_bar
import concurrent.futures
script_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(f"{script_dir}/")
sys.path.append(f"{script_dir}/../utils/")
from utils import get_centrality_bins, make_dir_root_file, logger
from sparse_dicts import get_sparse_dict

def check_existing_outputs(file_path):
    """
    Check if output file already exists, and open it accordingly.

    Args:
        file_path (str): path to the output ROOT file.

    Returns:
        tuple: (out_file, write_opt) where out_file is the opened TFile and write_opt is the write option.
    """
    if os.path.exists(file_path):
        logger(f"    Updating file: {file_path}")
        out_file = TFile(file_path, 'update')
        write_opt = TObject.kOverwrite
    else:
        logger(f"    Creating file: {file_path}")
        out_file = TFile.Open(file_path, 'recreate')
        write_opt = 0 # Standard

    return out_file, write_opt

def get_inputs(file, full_cfg, sparse_cfg, debug=False):
    """Load a single sparse and axes info
    
    Args:
        file (TFile): input ROOT file
        full_cfg (dict): full configuration dictionary
        sparse_cfg (dict): sparse configuration dictionary
        debug (bool, optional): print debug info. Defaults to False.
    
    Returns:
        tuple: (sparse, axes) where sparse is the loaded sparse histogram and axes is the dictionary of axes information
    """

    print(f"Loading data sparse {sparse_cfg['path']} from file {file.GetName()}")
    axes = get_sparse_dict(sparse_cfg['name'], full_cfg['Dmeson'])
    sparse = file.Get(sparse_cfg['path'])
    logger(f"Sparse {sparse} loaded from {sparse_cfg['path']}", level='WARNING')
    if full_cfg['Dmeson'] == 'Dzero':
        # TODO: safety checks for Dmeson reflecton and secondary peak
        if sparse_cfg['name'] == "RecoPrompt":
            sparse.GetAxis(axes['Origin']).SetRange(2, 2)       # select prompt
            sparse.GetAxis(axes['CandType']).SetRange(1, 2)     # select signal
        elif sparse_cfg['name'] == "RecoFD":
            sparse.GetAxis(axes['Origin']).SetRange(3, 3)       # select non-prompt
            sparse.GetAxis(axes['CandType']).SetRange(1, 2)     # select signal
        elif sparse_cfg['name'] == "RecoRefl":
            sparse.GetAxis(axes['CandType']).SetRange(3, 4)     # select reflection
        elif sparse_cfg['name'] == "RecoReflPrompt":
            sparse.GetAxis(axes['CandType']).SetRange(3, 4)     # select reflection
            sparse.GetAxis(axes['Origin']).SetRange(2, 2)       # select prompt   
        elif sparse_cfg['name'] == "RecoReflFD":
            sparse.GetAxis(axes['CandType']).SetRange(3, 4)     # select reflection
            sparse.GetAxis(axes['Origin']).SetRange(3, 3)       # select FD
        elif sparse_cfg['name'] == "GenPrompt":
            sparsesGen['GenPrompt'][i_file].GetAxis(axes_dict['GenPrompt']['Origin']).SetRange(2, 2)  # select prompt
        elif sparse_cfg['name'] == "GenFD":
            sparsesGen['GenFD'][i_file].GetAxis(axes_dict['GenFD']['Origin']).SetRange(3, 3)  # select non-prompt
        else:
            logger(f"Unknown sparse type for Dzero {sparse_cfg['name']}", level='ERROR')

    logger("Sparses loaded", level='INFO')
    if debug:
        print('\n###############################################################')
        for key, value in axes.items():
            logger(f"    {key}: {value}", level='DEBUG')
        print('###############################################################\n')

    return sparse, axes

def process_sparse(i_file, infile, full_cfg, sparse_cfg, prep_out_dir, input_out_dir):
    """
    Process a single sparse from an input file for all pt bins according to the configuration.
    
    Args:
        i_file (int): index of the input file
        infile (TFile): input ROOT file
        full_cfg (dict): full configuration dictionary
        sparse_cfg (dict): sparse configuration dictionary
        prep_out_dir (str): output directory for pre-processed files
        input_out_dir (str): sub-directory for the specific input configuration
    """

    logger(f'[Data] Processing file {i_file}, {infile.GetName()}')
    sparse, axes = get_inputs(infile, full_cfg, sparse_cfg, True)

    # Only for flow with SP, not for correlations (applied in O2Physics)
    if axes.get('Cent') is not None:
        cent_min, cent_max = get_centrality_bins(full_cfg['centrality'])[1]
        logger(f"Applying cent cut to sparse {sparse} with value {cent_min} -- {cent_max}", "INFO")
        sparse.GetAxis(axes['Cent']).SetRangeUser(cent_min, cent_max)

    pt_mins, pt_maxs = full_cfg['ptbins'][:-1], full_cfg['ptbins'][1:]
    bkg_maxs = full_cfg['preprocess']['bkg_cuts']
    axes_to_keep, rebin = sparse_cfg["axes"]['names'], sparse_cfg["axes"]['rebin']
    sparse_type, sparse_path = sparse_cfg['name'], sparse_cfg['path']
    sparse_dir, sparse_name = sparse_path.split('/')[0], sparse_path.split('/')[1]
    print(f"sparse_type: {sparse_type}, sparse_path: {sparse_path}, sparse_dir: {sparse_dir}, sparse_name: {sparse_name}")

    for pt_min, pt_max, bkg_max in zip(pt_mins, pt_maxs, bkg_maxs):
        print(f"Processing pT bin {pt_min} - {pt_max} with bkg max {bkg_max} ...")
        # Create output file
        out_file_dir = f"{prep_out_dir}/preprocess/{int(pt_min*10)}_{int(pt_max*10)}/{input_out_dir}"
        os.makedirs(out_file_dir, exist_ok=True)
        out_file_path = f'{out_file_dir}/AnalysisResults_{i_file}.root'
        out_file, write_opt = check_existing_outputs(out_file_path)

        sparse.GetAxis(axes.get('PtTrig', axes.get('Pt'))).SetRangeUser(pt_min, pt_max) # PtTrig for correlations, Pt for SP flow
        if axes.get('ScoreBkg') is not None: # Skip sparses for generated info
            sparse.GetAxis(axes['ScoreBkg']).SetRangeUser(0, bkg_max)
        proj_axes = [axes[ax_to_keep] for ax_to_keep in axes_to_keep]
        proj_sparse = sparse.Projection(len(proj_axes), array.array('i', proj_axes), 'O')
        proj_sparse.SetName(sparse.GetName())
        proj_sparse = proj_sparse.Rebin(array.array('i', rebin))
        make_dir_root_file(sparse_type, out_file)
        out_file.cd(sparse_type)
        proj_sparse.Write(f"hSparse{sparse_type}", write_opt)
        out_file.Delete(f"hSparse{sparse_type}" + ";*")
        proj_sparse.Delete()
        del proj_sparse

        gc.collect()

        out_file.Close()
        logger(f'----> Finished processing pT bin {pt_min} - {pt_max} for {i_file}, sparse: {sparse_cfg["name"]}\n', "INFO")

    del sparse
    gc.collect()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Arguments")
    parser.add_argument('config', metavar='text', 
                        default='config.yml', help='configuration file')
    args = parser.parse_args()

    with open(args.config, 'r') as cfg_pre:
        full_cfg = yaml.safe_load(cfg_pre)

    output_dir = full_cfg['outdirPrep'] if full_cfg.get("outdirPrep") else full_cfg['outdir']

    for input_cfg in full_cfg['preprocess']['inputs']:
        logger(f"##### Skimming {input_cfg['outdir']} #####")
        files = [TFile.Open(fp, 'r') for fp in input_cfg['files']]
        for sparse_cfg in input_cfg['sparses']:
            print(f"output_dir: {output_dir}")
            print(f"input_cfg['outdir']: {input_cfg['outdir']}")
            with concurrent.futures.ThreadPoolExecutor(full_cfg['preprocess']['workers']) as executor:
                tasks_data = [executor.submit(process_sparse, i_file, file, full_cfg, sparse_cfg, output_dir, f"{input_cfg['outdir']}/jobs") for i_file, file in enumerate(files)]
        [TFile.Close(file) for file in files]

        # use hadd to merge the results in the directories for the different pT bins
        for directory in os.listdir(f"{output_dir}/preprocess/"):
            prep_dir = f"{output_dir}/preprocess/{directory}/{input_cfg['outdir']}"
            files_to_merge = [f"./jobs/{file}" for file in os.listdir(f"{prep_dir}/jobs") if file.startswith("AnalysisResults_") and file.endswith(".root")]
            files_to_merge_str = ' '.join(files_to_merge)
            merged_file = f"{prep_dir}/AnalysisResults_pt_{directory}.root"
            log_merge = f"{prep_dir}/log_merge.txt"
            hadd = f"hadd -f {merged_file} {files_to_merge_str} > {log_merge}"
            logger(f"\n\nRunning command: {hadd}", "INFO")
            try:
                os.system(f"cd {prep_dir} && {hadd}")
            except Exception as e:
                logger(f"Error while creating hadd command: {e}", "ERROR")
                os.system(f"rm {merged_file}") # Remove the partiall merged file
                os.system(f"mv {log_merge} {prep_dir}/log_merge_error.txt")

        logger(f"Finished processing {input_cfg['outdir']}\n\n")
