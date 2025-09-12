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
from sparse_dicts import get_sparses

def check_existing_outputs(ptmin, ptmax, outputDir, stage):
    outFilePath = f'{outputDir}/preprocess/AnalysisResults_pt_{int(ptmin*10)}_{int(ptmax*10)}.root'
    if os.path.exists(outFilePath):
        logger(f"    [{stage}] Updating file: {outFilePath}")
        outFile = TFile(outFilePath, 'update')
        write_opt = TObject.kOverwrite
    else:
        logger(f"    [{stage}] Creating file: {outFilePath}")
        outFile = TFile.Open(outFilePath, 'recreate')
        write_opt = 0 # Standard

    return outFile, write_opt

def write_pt_bin_reso(ptmin, ptmax, outputDir, resolutions, data_sparses):
    # outFilePath = f'{outputDir}/preprocess/AnalysisResults_pt_{int(ptmin*10)}_{int(ptmax*10)}.root'
    # if os.path.exists(outFilePath):
    #     print(f"    [Reso] Updating file: {outFilePath}")
    #     outFile = TFile(outFilePath, 'update')
    #     write_opt = TObject.kOverwrite
    # else:
    #     print(f"    [Reso] Creating file: {outFilePath}")
    #     outFile = TFile.Open(outFilePath, 'recreate')
    #     write_opt = 0 # Standard

    outFile, write_opt = check_existing_outputs(ptmin, ptmax, outputDir, "Reso")

    for key, _ in data_sparses.items():
        make_dir_root_file(f'Data_{key}', outFile)
        outFile.cd(f'Data_{key}')
        resolutions[f"Reso_{key}"].Write('hResolution', write_opt)

    outFile.Close()
    logger(f'[Reso] Finished processing pT bin {ptmin} - {ptmax}\n\n')

def process_pt_bin_data(config, ptmin, ptmax, centmin, centmax, bkg_max_cut, debugPreprocessFile, outputDir, data_sparses, sparse_axes):
    logger(f'[Data] Processing pT bin {ptmin} - {ptmax}, cent {centmin}-{centmax}')
    
    # Force recreate of the output file when data are reprocessed, other operations are lightweight
    outFilePath = f'{outputDir}/preprocess/AnalysisResults_pt_{int(ptmin*10)}_{int(ptmax*10)}.root'
    logger(f"\t\t[Data] Creating file: {outFilePath}")
    outFile = TFile.Open(outFilePath, 'recreate')

    axes_data = config['preprocess']["axes_data"]['axis_names']
    rebin_data = config['preprocess']["axes_data"]['rebin_factors']
    
    for key, dataset_sparses in data_sparses.items():
        logger(f'\t\t[Data] Processing dataset: {key}')
        with alive_bar(len(dataset_sparses), title=f'[INFO] \t\t[Data] Processing {key}', bar='smooth') as bar:
            for iSparse, sparse in enumerate(dataset_sparses):
                sparse.GetAxis(sparse_axes['Pt']).SetRangeUser(ptmin, ptmax)
                sparse.GetAxis(sparse_axes['ScoreBkg']).SetRangeUser(0, bkg_max_cut)
                proj_axes = [sparse_axes[axtokeep] for axtokeep in axes_data]
                proj_sparse = sparse.Projection(len(proj_axes), array.array('i', proj_axes), 'O')
                proj_sparse.SetName(sparse.GetName())
                proj_sparse = proj_sparse.Rebin(array.array('i', rebin_data))
                if iSparse == 0:
                    merged_sparse_pt = proj_sparse.Clone()
                    proj_sparse.Delete()  # Delete the original projection to save memory
                    del proj_sparse
                    gc.collect()
                    make_dir_root_file(f'pt_{int(ptmin*10)}_{int(ptmax*10)}/{key}', debugPreprocessFile)
                    logger(f'\t[Data] Writing sparse for {key} with {merged_sparse_pt.GetNdimensions()} dimensions')
                    debugPreprocessFile.cd(f'pt_{int(ptmin*10)}_{int(ptmax*10)}/{key}')
                    for iDim in range(merged_sparse_pt.GetNdimensions()):
                        merged_sparse_pt.Projection(iDim).Write(axes_data[iDim], TObject.kOverwrite)
                else:
                    merged_sparse_pt.Add(proj_sparse)
                    proj_sparse.Delete()  # Delete the original projection to save memory
                    del proj_sparse
                    gc.collect()
                print(f"\t\t[Data] After adding sparse {iSparse}, merged_sparse_pt.GetEntries() = {merged_sparse_pt.GetEntries()}", flush=True)
                bar()
        make_dir_root_file(f'Data_{key}', outFile)
        logger(f'\t[Data] Writing sparse for {key} with {merged_sparse_pt.GetNdimensions()} dimensions')
        outFile.cd(f'Data_{key}')
        merged_sparse_pt.Write('hSparseFlowCharm', TObject.kOverwrite)
        merged_sparse_pt.Delete()
        del merged_sparse_pt
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

def pre_process_data_mc(config):

    # Load the configuration
    ptmins = config['ptbins'][:-1]
    ptmaxs = config['ptbins'][1:] 
    centmin, centmax = get_centrality_bins(config['centrality'])[1]

    # Load the ThnSparse
    data_sparses, reco_sparses, gen_sparses, sparse_axes, resolutions = get_sparses(config, config["operations"]["preprocess_data"], 
                                                                                    config["operations"]["preprocess_mc"], True)
    if config.get("outdirPrep") and config["outdirPrep"] != "":
        outputDir = config['outdirPrep']
    else:
        outputDir = config['outdir']
    os.makedirs(f'{outputDir}/preprocess', exist_ok=True)
    if os.path.exists(f'{outputDir}/preprocess/DebugPreprocess.root'):
        logger(f'File {outputDir}/preprocess/DebugPreprocess.root already exists, updating it.')
        debugPreprocessFile = TFile.Open(f'{outputDir}/preprocess/DebugPreprocess.root', 'update')
    else:
        logger(f'Creating file {outputDir}/preprocess/DebugPreprocess.root')
        debugPreprocessFile = TFile(f'{outputDir}/preprocess/DebugPreprocess.root', 'recreate')

    bkg_maxs = config['preprocess']['bkg_cuts']
    max_workers = config['preprocess']['workers'] # hyperparameter
    if config["operations"]["preprocess_data"] and config['preprocess'].get('data'):
        logger("##### Skimming Data #####")
        ### Centrally cut on centrality and max of bkg scores
        for key, dataset_sparses in data_sparses.items():
            for sparse in dataset_sparses:
                sparse.GetAxis(sparse_axes['Flow']['Cent']).SetRangeUser(centmin, centmax)
                sparse.GetAxis(sparse_axes['Flow']['ScoreBkg']).SetRangeUser(0, max(bkg_maxs))
        with concurrent.futures.ThreadPoolExecutor(max_workers) as executor:
            tasks_data = [executor.submit(process_pt_bin_data, config, ptmin, ptmax, centmin, centmax, bkg_maxs[iPt], 
                                                               debugPreprocessFile, outputDir, data_sparses, sparse_axes['Flow']) for iPt, (ptmin, ptmax) in enumerate(zip(ptmins, ptmaxs))]
        logger("Finished processing data")
        with concurrent.futures.ThreadPoolExecutor(max_workers) as executor:
            tasks_reso = [executor.submit(write_pt_bin_reso, ptmin, ptmax, outputDir, resolutions, data_sparses) for ptmin, ptmax in zip(ptmins, ptmaxs)]
        logger("Finished processing resolutions")

    if config["operations"].get("preprocess_mc") and config['preprocess'].get('mc'):
        logger("##### Skimming Monte Carlo #####")
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

    if not config["operations"]["preprocess_data"] and not config["operations"]["preprocess_mc"]:
        logger("No data or mc pre-processing enabled. Exiting.", level='ERROR')
    debugPreprocessFile.Close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Arguments")
    parser.add_argument('config_pre', metavar='text', 
                        default='config_pre.yml', help='configuration file')
    args = parser.parse_args()

    logger(f'Using configuration file: {args.config_pre}')
    with open(args.config_pre, 'r') as cfgPre:
        config = yaml.safe_load(cfgPre)

    pre_process_data_mc(config)
