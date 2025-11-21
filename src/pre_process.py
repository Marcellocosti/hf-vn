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

def check_existing_outputs(outFilePath):
    if os.path.exists(outFilePath):
        logger(f"    Updating file: {outFilePath}")
        outFile = TFile(outFilePath, 'update')
        write_opt = TObject.kOverwrite
    else:
        logger(f"    Creating file: {outFilePath}")
        outFile = TFile.Open(outFilePath, 'recreate')
        write_opt = 0 # Standard

    return outFile, write_opt

def process_sparse_data(ifile, file_path, full_cfg, sparse_cfg, out_dir):
    logger(f'[Data] Processing file {ifile}, {file_path}')
    infile = TFile.Open(file_path, 'r')
    sparse, axes, resolution = get_sparses_data(infile, full_cfg, sparse_cfg, True)
    print(f"sparse inside process_sparse_data: {sparse}")
    # Only for flow with SP, not for correlations (applied in O2Physics)
    if 'Correl' not in sparse_cfg['name']:
        logger(f"Applying cent cut to sparse {sparse} with value {centmin} -- {centmax}", "INFO")
        sparse.GetAxis(axes['cent']).SetRangeUser(centmin, centmax)
    print(f"sparse after cent cut: {sparse}")
    ptmins = full_cfg['ptbins'][:-1]
    ptmaxs = full_cfg['ptbins'][1:]
    bkg_maxs = full_cfg['preprocess']['bkg_cuts']
    axes_to_keep, rebin = sparse_cfg["axes"]['names'], sparse_cfg["axes"]['rebin']
    sparse_type, sparse_path = sparse_cfg['name'], sparse_cfg['path']
    print(f"axes_to_keep: {axes_to_keep}, rebin: {rebin}")
    for iPt, (ptmin, ptmax, bkgmax) in enumerate(zip(ptmins, ptmaxs, bkg_maxs)):
        # Force recreate of the output file when data are reprocessed, other operations are lightweight
        pt_str = f'pt_{int(ptmin*10)}_{int(ptmax*10)}'
        os.makedirs(f'{out_dir}/preprocess/{pt_str}', exist_ok=True)
        outFilePath = f'{out_dir}/preprocess/{pt_str}/AnalysisResults_{ifile}.root'
        logger(f"\t\t[Data] Creating file: {outFilePath}")
        # outFile = TFile.Open(outFilePath, 'recreate')
        outFile, write_opt = check_existing_outputs(outFilePath)
        print(f"outFile: {outFile}, write_opt: {write_opt}")
        # quit()
        print(f"sparse before pt and bkg cut: {sparse}")


        sparse.GetAxis(axes.get('PtTrig', axes.get('Pt'))).SetRangeUser(ptmin, ptmax) # PtTrig for correlations, Pt for SP flow
        print(f"sparse after pt cut: {sparse}")
        sparse.GetAxis(axes['score_bkg']).SetRangeUser(0, bkgmax)
        print(f"sparse after bkg cut: {sparse}")
        proj_axes = [axes[axtokeep] for axtokeep in axes_to_keep]
        print(f"proj_axes: {proj_axes}")
        proj_sparse = sparse.Projection(len(proj_axes), array.array('i', proj_axes), 'O')
        print(f"proj_sparse before rebin: {proj_sparse}")
        # proj_sparse.SetDirectory(0)
        print(f"proj_sparse after SetDirectory(0): {proj_sparse}")
        proj_sparse.SetName(sparse.GetName())
        print(f"proj_sparse after SetName: {proj_sparse}")
        proj_sparse = proj_sparse.Rebin(array.array('i', rebin))
        print(f"proj_sparse after Rebin: {proj_sparse}")
        # proj_sparse.SetDirectory(0)
        print(f"proj_sparse after second SetDirectory(0): {proj_sparse}")
        sparse_dir, sparse_name = sparse_path.split('/')[0], sparse_path.split('/')[1]
        print(f"sparse_dir: {sparse_dir}, sparse_name: {sparse_name}")
        print(f"Creating directory: Data_{sparse_type}/{sparse_dir} in output file")
        make_dir_root_file(f'Data_{sparse_type}/{sparse_dir}', outFile)
        print(f"sparse after make_dir_root_file: {sparse}")
        logger(f'\t[Data] Writing sparse for {sparse_name} with {proj_sparse.GetNdimensions()} dimensions')
        outFile.cd(f'Data_{sparse_type}/{sparse_dir}')
        # proj_sparse.Write(sparse_name, TObject.kOverwrite)
        proj_sparse.Write(sparse_name, write_opt)
        outFile.Delete(sparse_name + ";*")
        proj_sparse.Delete()
        del proj_sparse
        if resolution is not None:
            resolution.Write()
        gc.collect()

        outFile.Close()
        logger(f'[Data] Finished processing pT bin {ptmin} - {ptmax} for {ifile}, {file_path}\n\n')

    del sparse
    gc.collect()
    infile.Close()


def process_pt_bin_mc(config, ptmin, ptmax, centmin, centmax, bkg_max_cut, outputDir, reco_sparses, gen_sparses, sparse_axes):
    logger(f'[MC] Processing pT bin {ptmin} - {ptmax}, cent {centmin}-{centmax}')

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

    cfg_op = full_cfg['operations']
    if not cfg_op.get("preprocess_data") and not cfg_op.get("preprocess_mc"):
        logger("No data or mc pre-processing enabled. Exiting.", level='ERROR')
        sys.exit(1)

    if full_cfg['preprocess']['data_type'] == 'SP':
        # For Correlations, centrality cut is applied in O2Physics
        centmin, centmax = get_centrality_bins(full_cfg['centrality'])[1]

    if full_cfg.get("outdirPrep") and full_cfg["outdirPrep"] != "":
        outputDir = full_cfg['outdirPrep']
    else:
        outputDir = full_cfg['outdir']
    os.makedirs(f'{outputDir}/preprocess', exist_ok=True)

    bkg_maxs = full_cfg['preprocess']['bkg_cuts']
    max_workers = full_cfg['preprocess']['workers'] # hyperparameter

    sparse_paths = []
    if cfg_op.get("preprocess_data"):
        logger("##### Skimming Data #####")
        for sparse_cfg in full_cfg['preprocess']['sparses_data']:
            sparse_paths.append(f"Data_{sparse_cfg['name']}/{sparse_cfg['path']}")
            file_paths = sparse_cfg['inputs'] if isinstance(sparse_cfg['inputs'], list) \
                        else [f for f in os.listdir(sparse_cfg["inputs_dir"]) if f.endswith(".root")]
            print(f"file_paths = {file_paths}")
            with concurrent.futures.ThreadPoolExecutor(max_workers) as executor:
                tasks_data = [executor.submit(process_sparse_data, ifile, file_path, full_cfg, sparse_cfg, outputDir) for ifile, file_path in enumerate(file_paths)]
            # ### Centrally cut on centrality and max of bkg scores
            # print(f"data_sparses = {data_sparses}")
            # for sparse in data_sparses:
            #     logger(f"Applying bkg cut to sparse {sparse} with value {max(bkg_maxs)}", "INFO")
            #     sparse.GetAxis(sparse_axes['score_bkg']).SetRangeUser(0, max(bkg_maxs))
            #     if 'Correl' not in sparse_cfg['name']:  # Only for flow with SP, not for correlations (applied in O2Physics)
            #         logger(f"Applying cent cut to sparse {sparse} with value {centmin} -- {centmax}", "INFO")
            #         sparse.GetAxis(sparse_axes['cent']).SetRangeUser(centmin, centmax)
            #     else:
            #         logger(f"Skipping cent cut for correlations sparse {sparse}", "INFO")
            # # quit()
            # with concurrent.futures.ThreadPoolExecutor(max_workers) as executor:
            #     tasks_data = [executor.submit(process_pt_bin_data, sparse_cfg, ptmin, ptmax, bkg_maxs[iPt], outputDir,
            #                                                        data_sparses, sparse_axes, resolution) for iPt, (ptmin, ptmax) in enumerate(zip(ptmins, ptmaxs))]
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
                                                           outputDir, reco_sparses, gen_sparses, sparse_axes) for iPt, (ptmin, ptmax) in enumerate(zip(ptmins, ptmaxs))]
        logger("Finished processing MC")

    # # Project the sparses for debugging
    # for iPt, (ptmin, ptmax) in enumerate(zip(ptmins, ptmaxs)):
    #     pt_str = f'pt_{int(ptmin*10)}_{int(ptmax*10)}'
    #     outFilePath = f'{outputDir}/preprocess/AnalysisResults_{pt_str}.root'
    #     debugPreprocessFile = TFile.Open(outFilePath.replace('.root', '_debug.root'), 'recreate')
    #     debugPreprocessFile.cd()
    #     make_dir_root_file(pt_str, debugPreprocessFile)
    #     debugPreprocessFile.cd(pt_str)
    #     for sparse_path in sparse_paths:
    #         dir_name, sparse_task, sparse_name = sparse_path.split('/')[0], sparse_path.split('/')[1], sparse_path.split('/')[2]
    #         infileprep = TFile(f"{outputDir}/preprocess/AnalysisResults_{pt_str}.root")
    #         sparsetoDebug = infileprep.Get(sparse_path)
    #         make_dir_root_file(f"{pt_str}/{dir_name}", debugPreprocessFile)
    #         debugPreprocessFile.cd(f"{pt_str}/{dir_name}")
    #         for iDim in range(sparsetoDebug.GetNdimensions()):
    #             proj_debug = sparsetoDebug.Projection(iDim)
    #             proj_debug.Write(f'Proj_{sparse_name}_Dim{iDim}')
    #         infileprep.Close()
    #     debugPreprocessFile.Close()
