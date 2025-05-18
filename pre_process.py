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
import itertools
import concurrent.futures
script_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(f"{script_dir}/")
sys.path.append(f"{script_dir}/utils/")
from utils import get_centrality_bins, make_dir_root_file
from sparse_dicts import get_sparses

def pre_process_data_mc(config):

    # Load the configuration    
    ptmins = config['ptbins'][:-1]
    ptmaxs = config['ptbins'][1:] 
    centmin, centmax = get_centrality_bins(config['centrality'])[1]

    # Load the ThnSparse
    data_sparses, reco_sparses, gen_sparses, sparse_axes, resolutions = get_sparses(config, True)
    outputDir = config['out_dir']
    os.makedirs(f'{outputDir}/preprocess', exist_ok=True)

    if os.path.exists(f'{outputDir}/preprocess/DebugPreprocess.root'):
        print(f'Updating file: {outputDir}/preprocess/DebugPreprocess.root')
        debugPreprocessFile = TFile.Open(f'{outputDir}/preprocess/DebugPreprocess.root', 'update')
    else:
        print(f'Creating file: {outputDir}/preprocess/DebugPreprocess.root')
        debugPreprocessFile = TFile(f'{outputDir}/preprocess/DebugPreprocess.root', 'recreate')

    def process_pt_bin_data(config, ptmin, ptmax, centmin, centmax, bkg_max_cut):
        print(f'[Data] Processing pT bin {ptmin} - {ptmax}, cent {centmin}-{centmax}')
        outFilePath = f'{outputDir}/preprocess/AnalysisResults_pt_{int(ptmin*10)}_{int(ptmax*10)}.root'
        if os.path.exists(outFilePath):
            print(f"    Updating file: {outFilePath}")
            outFile = TFile(outFilePath, 'update')
            write_opt = TObject.kOverwrite
        else:
            print(f"    Creating file: {outFilePath}")
            outFile = TFile.Open(outFilePath, 'recreate')
            write_opt = 0 # Standard

        axes_data, rebin_data = [], []
        for ax, rebin in config['preprocess']["axes_data"].items(): 
            axes_data.append(ax)
            rebin_data.append(rebin)
        for key, dataset_sparses in data_sparses.items():
            for iSparse, sparse in enumerate(dataset_sparses):
                sparse.GetAxis(sparse_axes['Flow']['Pt']).SetRangeUser(ptmin, ptmax)
                sparse.GetAxis(sparse_axes['Flow']['score_bkg']).SetRangeUser(0, bkg_max_cut)
                proj_axes = [sparse_axes["Flow"][axtokeep] for axtokeep in axes_data]
                proj_sparse = sparse.Projection(len(proj_axes), array.array('i', proj_axes), 'O')
                proj_sparse.SetName(sparse.GetName())
                proj_sparse = proj_sparse.Rebin(array.array('i', rebin_data))
                
                if iSparse == 0:
                    merged_sparse_pt = proj_sparse.Clone()
                    make_dir_root_file(f'pt_{int(ptmin*10)}_{int(ptmax*10)}/{key}', debugPreprocessFile)
                    debugPreprocessFile.cd(f'pt_{int(ptmin*10)}_{int(ptmax*10)}/{key}')
                    for iDim in range(merged_sparse_pt.GetNdimensions()):
                        merged_sparse_pt.Projection(iDim).Write(axes_data[iDim], TObject.kOverwrite)
                else:
                    merged_sparse_pt.Add(proj_sparse)
        
            make_dir_root_file(f'Data_{key}', outFile)
            outFile.cd(f'Data_{key}')
            merged_sparse_pt.Write('hSparseFlowCharm', write_opt)
            resolutions[f"Reso_{key}"].Write('hResolution', write_opt)
            del merged_sparse_pt

        outFile.Close()
        print(f'[Data] Finished processing pT bin {ptmin} - {ptmax}\n\n')
        
    def process_pt_bin_mc(config, ptmin, ptmax, centmin, centmax, bkg_max_cut):
        print(f'[MC] Processing pT bin {ptmin} - {ptmax}, cent {centmin}-{centmax}')
        outFilePath = f'{outputDir}/preprocess/AnalysisResults_pt_{int(ptmin*10)}_{int(ptmax*10)}.root'
        if os.path.exists(outFilePath):
            print(f"    Updating file: {outFilePath}")
            outFile = TFile(outFilePath, 'update')
            write_opt = TObject.kOverwrite
        else:
            print(f"    Creating file: {outFilePath}")
            outFile = TFile.Open(outFilePath, 'recreate')
            write_opt = 0 # Standard

        axes_reco, rebin_reco, axes_gen, rebin_gen = [], [], [], []
        for ax, rebin in config['preprocess']["axes_reco"].items(): 
            axes_reco.append(ax)
            rebin_reco.append(rebin)
        for ax, rebin in config['preprocess']["axes_gen"].items(): 
            axes_gen.append(ax)
            rebin_gen.append(rebin)

        # cut on pt and bkg on all the reco and gen sparses
        make_dir_root_file('MC/Reco/', outFile)
        for key, sparse_type in reco_sparses.items():
            [sparse.GetAxis(sparse_axes[key]['Pt']).SetRangeUser(ptmin, ptmax) for sparse in sparse_type] 
            [sparse.GetAxis(sparse_axes[key]['score_bkg']).SetRangeUser(0, bkg_max_cut) for sparse in sparse_type]
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
                        processed_sparse.Projection(iDim).Write(axes_reco[iDim], TObject.kOverwrite)
                else:
                    processed_sparse.Add(proj_sparse)
            outFile.cd('MC/Reco/')
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
                proj_sparse = proj_sparse.Rebin(array.array('i', rebin_reco))

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
        print(f'[MC] Finished processing pT bin {ptmin} - {ptmax}\n\n')

    bkg_maxs = config['preprocess']['bkg_cuts']
    max_workers = config['preprocess']['workers'] # hyperparameter
    if config["operations"]["preprocess_data"] and config['preprocess'].get('data'):
        print("##### Skimming data #####")
        ### Centrally cut on centrality and max of bkg scores
        for key, dataset_sparses in data_sparses.items():
            for sparse in dataset_sparses:
                sparse.GetAxis(sparse_axes['Flow']['cent']).SetRangeUser(centmin, centmax)
                sparse.GetAxis(sparse_axes['Flow']['score_bkg']).SetRangeUser(0, max(bkg_maxs))
        with concurrent.futures.ThreadPoolExecutor(max_workers) as executor:
            tasks = [executor.submit(process_pt_bin_data, config, ptmin, ptmax, centmin, centmax, bkg_maxs[iPt]) for iPt, (ptmin, ptmax) in enumerate(zip(ptmins, ptmaxs))]
            for task in concurrent.futures.as_completed(tasks):
                task.result()
        print("Finished processing data")

    if config["operations"]["preprocess_mc"] and config['preprocess'].get('mc'):
        print("##### Skimming Monte Carlo #####")
        ### Centrally cut on centrality and max of bkg scores
        for key, sparse_type in reco_sparses.items():
            [sparse.GetAxis(sparse_axes[key]['cent']).SetRangeUser(centmin, centmax) for sparse in sparse_type]
            [sparse.GetAxis(sparse_axes[key]['score_bkg']).SetRangeUser(0, max(bkg_maxs)) for sparse in sparse_type]
        for key, sparse_type in gen_sparses.items():
            [sparse.GetAxis(sparse_axes[key]['cent']).SetRangeUser(centmin, centmax) for sparse in sparse_type]
        with concurrent.futures.ThreadPoolExecutor(max_workers) as executor:
            tasks = [executor.submit(process_pt_bin_mc, config, ptmin, ptmax, centmin, centmax, bkg_maxs[iPt]) for iPt, (ptmin, ptmax) in enumerate(zip(ptmins, ptmaxs))]
            for task in concurrent.futures.as_completed(tasks):
                task.result()

        print("Finished processing MC")

    if not config["operations"]["preprocess_data"] and not config["operations"]["preprocess_mc"]:
        print("No data or mc pre-processing enabled. Exiting.")
        sys.exit()
    
    debugPreprocessFile.Close()
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Arguments")
    parser.add_argument('config_pre', metavar='text', 
                        default='config_pre.yml', help='configuration file')
    args = parser.parse_args()

    print(f'Using configuration file: {args.config_pre}')
    with open(args.config_pre, 'r') as cfgPre:
        config = yaml.safe_load(cfgPre)

    pre_process_data_mc(config)
