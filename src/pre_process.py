'''
This script is used to pre-process a/multi large AnRes.root for the BDT training:
    - split the input by pT
    - obtain the sigma from prompt enhance sample
python3 pre_process.py config_pre.yml AnRes_1.root AnRes_2.root --pre --sigma  
'''
import os
import sys
import yaml
import numpy as np
import array
import ROOT
from ROOT import TFile, TObject
import argparse
import gc
import itertools
from alive_progress import alive_bar
import concurrent.futures
script_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(f"{script_dir}/")
sys.path.append(f"{script_dir}/../utils/")
from utils import get_centrality_bins, make_dir_root_file, logger
from sparse_dicts import get_sparses_data, get_sparses_mc

def check_existing_outputs(ptmin, ptmax, outputDir, stage):
    pt_str = f'pt_{int(ptmin*10)}_{int(ptmax*10)}'
    outFilePath = f'{outputDir}/preprocess/AnalysisResults_{pt_str}.root'
    if os.path.exists(outFilePath):
        logger(f"    [{stage}] Updating file: {outFilePath}")
        outFile = TFile(outFilePath, 'update')
        write_opt = TObject.kOverwrite
    else:
        logger(f"    [{stage}] Creating file: {outFilePath}")
        outFile = TFile.Open(outFilePath, 'recreate')
        write_opt = 0 # Standard

    return outFile, write_opt

def process_pt_bin_data(cfg, ptmin, ptmax, bkg_max_cut, debugPreprocessFile, outputDir, sparses, axes, resolution):
    logger(f'[Data] Processing pT bin {ptmin} - {ptmax}')
    
    # Force recreate of the output file when data are reprocessed, other operations are lightweight
    pt_str = f'pt_{int(ptmin*10)}_{int(ptmax*10)}'
    outFilePath = f'{outputDir}/preprocess/AnalysisResults_{pt_str}.root'
    logger(f"\t\t[Data] Creating file: {outFilePath}")
    outFile = TFile.Open(outFilePath, 'recreate')

    axes_to_keep = cfg["axes"]['names']
    rebin = cfg["axes"]['rebin']

    sparse_type, sparse_path = cfg['name'], cfg['path']
    with alive_bar(len(sparses), title=f'[INFO] \t\t[Data] Processing {sparse_type}, {sparse_path}', bar='smooth') as bar:
        logger(f'\t\t[Data] Processing dataset: {sparse_type}')
        for iSparse, sparse in enumerate(sparses):
            sparse.GetAxis(axes.get('PtTrig', axes['Pt'])).SetRangeUser(ptmin, ptmax) # PtTrig for correlations, Pt for SP flow
            sparse.GetAxis(axes['score_bkg']).SetRangeUser(0, bkg_max_cut)
            proj_axes = [axes[axtokeep] for axtokeep in axes_to_keep]
            proj_sparse = sparse.Projection(len(proj_axes), array.array('i', proj_axes), 'O')
            proj_sparse.SetName(sparse.GetName())
            proj_sparse = proj_sparse.Rebin(array.array('i', rebin))
            if iSparse == 0:
                merged_sparse_pt = proj_sparse.Clone()
                proj_sparse.Delete()  # Delete the original projection to save memory
                del proj_sparse
                gc.collect()
                make_dir_root_file(f'{pt_str}/{sparse_path}', debugPreprocessFile)
                logger(f'\t[Data] Writing sparse for {sparse_path} with {merged_sparse_pt.GetNdimensions()} dimensions')
                debugPreprocessFile.cd(f'{pt_str}/{sparse_path}')
                for iDim in range(merged_sparse_pt.GetNdimensions()):
                    merged_sparse_pt.Projection(iDim).Write(axes_to_keep[iDim], TObject.kOverwrite)
            else:
                merged_sparse_pt.Add(proj_sparse)
                proj_sparse.Delete()  # Delete the original projection to save memory
                del proj_sparse
                gc.collect()
            print(f"\t\t[Data] After adding sparse {iSparse}, merged_sparse_pt.GetEntries() = {merged_sparse_pt.GetEntries()}", flush=True)
            bar()
        sparse_dir, sparse_name = sparse_path.split('/')[0], sparse_path.split('/')[1]
        make_dir_root_file(f'Data_{sparse_type}/{sparse_dir}', outFile)
        logger(f'\t[Data] Writing sparse for {sparse_name} with {merged_sparse_pt.GetNdimensions()} dimensions')
        outFile.cd(f'Data_{sparse_type}/{sparse_dir}')
        merged_sparse_pt.Write(sparse_name, TObject.kOverwrite)
        merged_sparse_pt.Delete()
        del merged_sparse_pt
        if resolution is not None:
            resolution.Write()
        gc.collect()

    outFile.Close()
    logger(f'[Data] Finished processing pT bin {ptmin} - {ptmax}\n\n')

def process_pt_bin_mc(config, ptmin, ptmax, centmin, centmax, bkg_max_cut, debugPreprocessFile, outputDir, reco_sparses, gen_sparses, sparse_axes):
    logger(f'[MC] Processing pT bin {ptmin} - {ptmax}, cent {centmin}-{centmax}')
    # outFilePath = f'{outputDir}/preprocess/AnalysisResults_pt_{int(ptmin*10)}_{int(ptmax*10)}.root'
    # if os.path.exists(outFilePath):
    #     print(f"    [MC] Updating file: {outFilePath}")
    #     outFile = TFile(outFilePath, 'update')
    #     write_opt = TObject.kOverwrite
    # else:
    #     print(f"    [MC] Creating file: {outFilePath}")
    #     outFile = TFile.Open(outFilePath, 'recreate')
    #     write_opt = 0 # Standard

    outFile, write_opt = check_existing_outputs(ptmin, ptmax, outputDir, "MC")

    axes_reco = list(config['preprocess']["axes_reco"].keys())
    rebin_reco = list(config['preprocess']["axes_reco"].values())
    axes_gen = list(config['preprocess']["axes_gen"].keys())
    rebin_gen = list(config['preprocess']["axes_gen"].values())

    # cut on pt and bkg on all the reco and gen sparses
    make_dir_root_file('MC/Reco/', outFile)
    for key, sparse_type in reco_sparses.items():
        for sparse in sparse_type:
            sparse.GetAxis(sparse_axes[key]['Pt']).SetRangeUser(ptmin, ptmax)
            sparse.GetAxis(sparse_axes[key]['score_bkg']).SetRangeUser(0, bkg_max_cut)
    for key, sparse_type in reco_sparses.items():
        for iSparse, sparse in enumerate(sparse_type):
            cloned_sparse = sparse.Clone()
            proj_axes = [sparse_axes[key][axtokeep] for axtokeep in axes_reco if axtokeep in sparse_axes[key]] # Different axes for reco and gen allowed
            proj_sparse = cloned_sparse.Projection(len(proj_axes), array.array('i', proj_axes), 'O')
            proj_sparse.SetName(f"{cloned_sparse.GetName()}_{iSparse}")
            proj_sparse = proj_sparse.Rebin(array.array('i', rebin_reco))

            if iSparse == 0:
                processed_sparse = proj_sparse.Clone()
                make_dir_root_file(f'pt_{int(ptmin*10)}_{int(ptmax*10)}/MC/Reco/{key}', debugPreprocessFile)
                debugPreprocessFile.cd(f'pt_{int(ptmin*10)}_{int(ptmax*10)}/MC/Reco/{key}')
                for iDim in range(processed_sparse.GetNdimensions()):
                    try:
                        processed_sparse.Projection(iDim).Write(axes_reco[iDim], TObject.kOverwrite)
                    except Exception as e:
                        print(f"⚠️ Exception at iDim={iDim}: {e}", flush=True)
            else:
                processed_sparse.Add(proj_sparse)
        outFile.cd('MC/Reco/')
        processed_sparse.SetName(f'h{key}')
        processed_sparse.Write(f'h{key}', write_opt)
        del processed_sparse

    make_dir_root_file('MC/Gen/', outFile)
    for key, sparse_type in gen_sparses.items():
        [sparse.GetAxis(sparse_axes[key]['Pt']).SetRangeUser(ptmin, ptmax) for sparse in sparse_type]
    for key, sparse_type in gen_sparses.items():
        for iSparse, sparse in enumerate(sparse_type):
            cloned_sparse = sparse.Clone()
            proj_axes = [sparse_axes[key][axtokeep] for axtokeep in axes_gen if axtokeep in sparse_axes[key]]
            proj_sparse = cloned_sparse.Projection(len(proj_axes), array.array('i', proj_axes), 'O')
            proj_sparse.SetName(f"{cloned_sparse.GetName()}_{iSparse}")
            proj_sparse = proj_sparse.Rebin(array.array('i', rebin_gen))

            if iSparse == 0:
                processed_sparse = proj_sparse.Clone()
                make_dir_root_file(f'pt_{int(ptmin*10)}_{int(ptmax*10)}/MC/Gen/{key}', debugPreprocessFile)
                debugPreprocessFile.cd(f'pt_{int(ptmin*10)}_{int(ptmax*10)}/MC/Gen/{key}')
                for iDim in range(processed_sparse.GetNdimensions()):
                    processed_sparse.Projection(iDim).Write(axes_gen[iDim], TObject.kOverwrite)
            else:
                processed_sparse.Add(proj_sparse)
        outFile.cd('MC/Gen/')
        processed_sparse.Write(f'h{key}', write_opt)
        del processed_sparse
    outFile.Close()
    logger(f'[MC] Finished processing pT bin {ptmin} - {ptmax}\n\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Arguments")
    parser.add_argument('config_pre', metavar='text', 
                        default='config_pre.yml', help='configuration file')
    args = parser.parse_args()

    logger(f'Using configuration file: {args.config_pre}')
    with open(args.config_pre, 'r') as cfgPre:
        full_cfg = yaml.safe_load(cfgPre)

    # Load the full_cfguration
    ptmins = full_cfg['ptbins'][:-1]
    ptmaxs = full_cfg['ptbins'][1:] 

    cfg_op = full_cfg['operations']

    if full_cfg['preprocess']['data_type'] == 'SP':
        # For Correlations, centrality cut is applied in O2Physics
        centmin, centmax = get_centrality_bins(full_cfg['centrality'])[1]

    if full_cfg.get("outdirPrep") and full_cfg["outdirPrep"] != "":
        outputDir = full_cfg['outdirPrep']
    else:
        outputDir = full_cfg['outdir']
    os.makedirs(f'{outputDir}/preprocess', exist_ok=True)
    if os.path.exists(f'{outputDir}/preprocess/DebugPreprocess.root'):
        logger(f'File {outputDir}/preprocess/DebugPreprocess.root already exists, updating it.')
        debugPreprocessFile = TFile.Open(f'{outputDir}/preprocess/DebugPreprocess.root', 'update')
    else:
        logger(f'Creating file {outputDir}/preprocess/DebugPreprocess.root')
        debugPreprocessFile = TFile(f'{outputDir}/preprocess/DebugPreprocess.root', 'recreate')

    bkg_maxs = full_cfg['preprocess']['bkg_cuts']
    max_workers = full_cfg['preprocess']['workers'] # hyperparameter

    if cfg_op["preprocess_data"]:
        logger("##### Skimming Data #####")
        file_paths = full_cfg['preprocess']['inputs_data'] if full_cfg['preprocess'].get('inputs_data') \
                     else [f for f in os.listdir(full_cfg['preprocess']["inputs_data_dir"]) if f.endswith(".root")]
        files = [TFile.Open(f, 'r') for f in file_paths]
        for sparse_cfg in full_cfg['preprocess']['sparses_data']:
            data_sparses, sparse_axes, resolution = get_sparses_data(files, full_cfg, sparse_cfg, True)
            ### Centrally cut on centrality and max of bkg scores
            print(f"data_sparses: {data_sparses}")
            for sparse in data_sparses:
                logger(f"Applying bkg cut to sparse {sparse} with value {max(bkg_maxs)}", "INFO")
                sparse.GetAxis(sparse_axes['score_bkg']).SetRangeUser(0, max(bkg_maxs))
                if 'Correl' not in sparse_cfg['name']:  # Only for flow with SP, not for correlations (applied in O2Physics)
                    logger(f"Applying cent cut to sparse {sparse} with value {centmin} -- {centmax}", "INFO")
                    sparse.GetAxis(sparse_axes['cent']).SetRangeUser(centmin, centmax)
                else:
                    logger(f"Skipping cent cut for correlations sparse {sparse}", "INFO")
            with concurrent.futures.ThreadPoolExecutor(max_workers) as executor:
                tasks_data = [executor.submit(process_pt_bin_data, sparse_cfg, ptmin, ptmax, bkg_maxs[iPt], debugPreprocessFile, outputDir,
                                                                   data_sparses, sparse_axes, resolution) for iPt, (ptmin, ptmax) in enumerate(zip(ptmins, ptmaxs))]
        [file.Close() for file in files]
        logger("Finished processing data")

    if cfg_op.get("preprocess_mc") and config['preprocess'].get('mc'):
        logger("##### Skimming Monte Carlo #####")
        reco_sparses, gen_sparses, sparse_axes = get_sparses_mc(config, data_type, cfg_op.get("preprocess_mc", False), True)
        ### Centrally cut on centrality and max of bkg scores
        for key, sparse_type in reco_sparses.items():
            [sparse.GetAxis(sparse_axes[key]['Cent']).SetRangeUser(centmin, centmax) for sparse in sparse_type]
            [sparse.GetAxis(sparse_axes[key]['ScoreBkg']).SetRangeUser(0, max(bkg_maxs)) for sparse in sparse_type]
        for key, sparse_type in gen_sparses.items():
            [sparse.GetAxis(sparse_axes[key]['Cent']).SetRangeUser(centmin, centmax) for sparse in sparse_type]
        with concurrent.futures.ThreadPoolExecutor(max_workers) as executor:
            tasks_mc = [executor.submit(process_pt_bin_mc, config, ptmin, ptmax, centmin, centmax, bkg_maxs[iPt], 
                                                           debugPreprocessFile, outputDir, reco_sparses, gen_sparses, sparse_axes) for iPt, (ptmin, ptmax) in enumerate(zip(ptmins, ptmaxs))]
        logger("Finished processing MC")

    if not cfg_op["preprocess_data"] and not cfg_op["preprocess_mc"]:
        logger("No data or mc pre-processing enabled. Exiting.", level='ERROR')
    debugPreprocessFile.Close()
