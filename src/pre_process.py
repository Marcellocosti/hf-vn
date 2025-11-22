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

    for pt_min, pt_max, bkg_max in zip(pt_mins, pt_maxs, bkg_maxs):

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
        proj_sparse.Write(sparse_name, write_opt)
        out_file.Delete(sparse_name + ";*")
        proj_sparse.Delete()
        del proj_sparse

        if sparse_cfg.get('resolution') is not None:
            logger(f"Loading resolution from {sparse_cfg['resolution']}", level='INFO')
            reso_file = TFile.Open(sparse_cfg["resolution"], 'r')
            det_A = full_cfg.get('detA', 'FT0c')
            det_B = full_cfg.get('detB', 'FV0a')
            det_C = full_cfg.get('detC', 'TPCtot')
            resolution = reso_file.Get(f'{det_A}_{det_B}_{det_C}/histo_reso_delta_cent')
            resolution.SetDirectory(0)
            reso_file.Close()
            resolution.Write()
        gc.collect()

        out_file.Close()
        logger(f'   Finished processing pT bin {pt_min} - {pt_max} for {i_file}, {file_path}, sparse: {sparse_cfg["name"]}\n\n')

    del sparse
    gc.collect()
    infile.Close()

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
            with concurrent.futures.ThreadPoolExecutor(full_cfg['preprocess']['workers']) as executor:
                tasks_data = [executor.submit(process_sparse, i_file, file, full_cfg, sparse_cfg, output_dir, input_cfg['outdir']) for i_file, file in enumerate(files)]
        [TFile.Close(file) for file in files]
        logger(f"Finished processing {input_cfg['outdir']}\n\n")
