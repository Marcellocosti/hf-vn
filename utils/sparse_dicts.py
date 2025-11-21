import sys
from alive_progress import alive_bar
import os
from ROOT import TFile # pyright: ignore # type: ignore
sys.path.append("./")
from utils import logger, get_centrality_bins

def get_sparses_dicts_mc(sparse_name, dmeson, beforeDMesonPR = False):
    
    axes_dict = {}
    if dmeson == 'Dzero':
        axes_dict['RecoPrompt'] = {
            'ScoreBkg': 0,
            'score_prompt': 1,
            'ScoreFD': 2,
            'Mass': 3,
            'Pt': 4,
            'Y': 5,
            'CandType': 6,
            'PtBMoth': 7,
            'Origin': 8,
            'NPvContr': 9,
            'Cent': 10,
            'Occ': 11,
        }
        axes_dict['RecoFD'] = axes_dict['RecoPrompt']
        axes_dict['RecoRefl'] = axes_dict['RecoPrompt']
        axes_dict['RecoReflPrompt'] = axes_dict['RecoPrompt']
        axes_dict['RecoReflFD'] = axes_dict['RecoPrompt']
        axes_dict['GenPrompt'] = {
            'Pt': 0,
            'PtBMoth': 1,
            'Y': 2,
            'Origin': 3,
            'NPvContr': 4,
            'Cent': 5,
            'Occ': 6
        }
        axes_dict['GenFD'] = {
            'Pt': 0,
            'PtBMoth': 1,
            'Y': 2,
            'Origin': 3,
            'NPvContr': 4,
            'Cent': 5,
            'Occ': 6
        }
    elif dmeson == 'Dplus':
        axes_dict['RecoPrompt'] = {
            'Mass': 0,
            'Pt': 1,
            'ScoreBkg': 2,
            'score_prompt': 3,
            'ScoreFD': 4,
            'Cent': 5,
            'Occ': 6,
        }
        axes_dict['RecoFD'] = {
            'Mass': 0,
            'Pt': 1,
            'ScoreBkg': 2,
            'score_prompt': 3,
            'ScoreFD': 4,
            'Cent': 5,
            'Occ': 6,
            'PtBMoth': 7,
            'FlagBHad': 8,
        }
        axes_dict['GenPrompt'] = {
            'Pt': 0,
            'Y': 1,
            'Cent': 2,
            'Occ': 3
        }
        axes_dict['GenFD'] = {
            'Pt': 0,
            'Y': 1,
            'Cent': 2,
            'PtBMoth': 3,
            'FlagBHad': 4,
        }
    elif dmeson == 'Ds':
        axes_dict['RecoPrompt'] = {
            'Mass': 0,
            'Pt': 1,
            'Cent': 3,  # Check number 2
            'NPvContr': 4,
            'ScoreBkg': 5,
            'score_prompt': 6,
            'ScoreFD': 7,
            'Occ': 8,
        }
        axes_dict['RecoFD'] = {
            'Mass': 0,
            'Pt': 1,
            'Cent': 2,
            'ScoreBkg': 3,
            'score_prompt': 4,
            'ScoreFD': 5,
            'PtBMoth': 6,
            'FlagBHad': 7,
            'Occ': 8
        }
        axes_dict['GenPrompt'] = {
            'Pt': 0,
            'Y': 1,
            'NPvContr': 2,
            'Cent': 3,
            'Occ': 4
        }
        axes_dict['GenFD'] = {
            'Pt': 0,
            'Y': 1,
            'Cent': 2,
            'PtBMoth': 3,
            'FlagBHad': 4,
            'Occ': 5
        }
    else:
        logger(f"Data type {data_type} not recognized", level='ERROR')

    return axes_dict

def get_sparses_dicts_data(sparse_name):

    print(f"Getting sparse dict for {sparse_name}")
    print(f"sparse_name: {sparse_name} vs CorrelMaps")
    if sparse_name == "CorrelMaps":
        return {
                'PoolBin': 0,
                'PtTrig': 1,
                'PtAssoc': 2,
                'DeltaEta': 3,
                'DeltaPhi': 4,
                'Mass': 5,
                'score_bkg': 6,
                'score_FD': 7
                }
    elif sparse_name == "CorrelTrig":
        return {
                'Mass': 0,
                'PtTrig': 1,
                'score_bkg': 2,
                'score_FD': 3
                }
    elif sparse_name == "FlowSP":
        return {
                'Mass': 0,
                'Pt': 1,
                'cent': 2,
                'sp': 3,
                'score_bkg': 4,
                'score_FD': 5,
                'occ': 6
                }
    else:
        logger(f"Sparse type {sparse_name} not recognized", level='ERROR')

def get_pt_preprocessed_sparses(config, iPt):
    
    logger("Loading preprocessed sparses", level='INFO')
    sparsesFlow, sparsesReco, sparsesGen, axes_dict, resolutions = {}, {}, {}, {}, {}
    pre_cfg = config['preprocess']
    
    # Find preprocess config of sparse with name "FlowSP" (this is the one to be projected)
    for sparse_cfg in pre_cfg['sparses_data']:
        if sparse_cfg['name'] == 'FlowSP':
            sparse_proj_cfg = sparse_cfg
            break
    print(f"\n\naxes_dict: {axes_dict}")
    ptmin = config["ptbins"][iPt]
    ptmax = config["ptbins"][iPt+1]

    if config.get("outdirPrep") and config["outdirPrep"] != "":
        infileprep = TFile(f"{config['outdirPrep']}/preprocess/{int(ptmin*10)}_{int(ptmax*10)}/AnalysisResults.root")
    else:
        infileprep = TFile(f"{config['outdir']}/preprocess/{int(ptmin*10)}_{int(ptmax*10)}/AnalysisResults.root")

    if config["operations"].get("proj_data"):
        inputs_dir = f"Data_FlowSP/hf-task-flow-charm-hadrons"
        sparse_data_name = f"Data_{sparse_proj_cfg['name']}/{sparse_proj_cfg['path']}"
        axes_dict['Flow'] = {ax: iax for iax, ax in enumerate(sparse_proj_cfg['axes']['names'])}
        sparsesFlow[sparse_proj_cfg['name']] = infileprep.Get(sparse_data_name)
        resolutions[f'Reso_{sparse_proj_cfg["name"]}'] = infileprep.Get(f'{inputs_dir}/histo_reso_delta_cent')

    if config["operations"].get("proj_mc"):
        subdir = infileprep.Get("MC/Reco")
        for key in subdir.GetListOfKeys():
            obj = key.ReadObj()
            sparsesReco[key.GetName()[1:]] = obj
            axes_dict[key.GetName()[1:]] = {ax: iax for iax, ax in enumerate(pre_cfg['axes_reco'].keys())}

        subdir = infileprep.Get("MC/Gen")
        for key in subdir.GetListOfKeys():
            obj = key.ReadObj()
            sparsesGen[key.GetName()[1:]] = obj
            axes_dict[key.GetName()[1:]] = {ax: iax for iax, ax in enumerate(pre_cfg['axes_gen'].keys())}

    infileprep.Close()

    return sparsesFlow, sparsesReco, sparsesGen, axes_dict, resolutions

def get_sparses_data(file, full_cfg, sparse_cfg, debug=False):
    """Load the sparses and axes infos

    Args:
        config (dict): the flow config dictionary
        get_data (bool, optional): load data sparses. Defaults to True.
        get_mc (bool, optional): load mc sparses. Defaults to True.
        debug (bool, optional): print debug info. Defaults to False.

    Outputs:
        sparsesFlow: thnSparse in the flow task
        sparsesReco: thnSparse of reco level from the D meson task
        sparsesGen: thnSparse of gen level from the D meson task
        axes_dict (dict): dictionary of the axes for each sparse
    """

    print(f"Loading data sparse {sparse_cfg['path']} from file {file.GetName()}")
    axes = get_sparses_dicts_data(sparse_cfg['name'])
    sparse = file.Get(sparse_cfg['path'])

    resolution = None
    if full_cfg['preprocess']['data_type'] == 'SP':
        logger(f"Loading resolution from {sparse_cfg['resolution']}", level='INFO')
        resofile = TFile.Open(sparse_cfg["resolution"], 'r')
        det_A = full_cfg.get('detA', 'FT0c')
        det_B = full_cfg.get('detB', 'FV0a')
        det_C = full_cfg.get('detC', 'TPCtot')
        _, (centMin, centMax) = get_centrality_bins(full_cfg["centrality"])
        print(f"cent_{centMin}_{centMax}/{det_A}_{det_B}_{det_C}/histo_reso_delta_cent")
        resolution = resofile.Get(f'{det_A}_{det_B}_{det_C}/histo_reso_delta_cent')
        resolution.SetDirectory(0)
        resofile.Close()

    logger("Sparses loaded", level='INFO')
    if debug:
        print('\n')
        print('###############################################################')
        for key, value in axes.items():
            logger(f"    {key}: {value}", level='DEBUG')
        print('###############################################################')
        print('\n')

    print(f"Returning sparses: {sparse}")
    return sparse, axes, resolution

def get_sparses_mc(config, get_mc=None, debug=False, MCBeforePRDplus=False):
    """Load the sparses and axes infos

    Args:
        config (dict): the flow config dictionary
        get_data (bool, optional): load data sparses. Defaults to True.
        get_mc (bool, optional): load mc sparses. Defaults to True.
        debug (bool, optional): print debug info. Defaults to False.

    Outputs:
        sparsesFlow: thnSparse in the flow task
        sparsesReco: thnSparse of reco level from the D meson task
        sparsesGen: thnSparse of gen level from the D meson task
        axes_dict (dict): dictionary of the axes for each sparse
    """

    sparsesReco, sparsesGen = {}, {}
    print(f"MCBeforePRDplus: {MCBeforePRDplus}")
    axes_dict = get_sparses_dicts_mc(config, config['Dmeson'], MCBeforePRDplus)
    pre_cfg = config['preprocess'] if config.get('preprocess') else config

    print(f"Loading mc sparse from: {pre_cfg['mc']}")
    infiletask = [TFile(pre_cfg['mc'])] if isinstance(pre_cfg['mc'], str) else [TFile(pre_cfg['mc']) for file in pre_cfg['mc']]

    if config['Dmeson'] == 'Dzero':
        sparseD0Path = 'hf-task-d0/hBdtScoreVsMassVsPtVsPtBVsYVsOriginVsD0Type'
        sparsesReco['RecoPrompt'] = [file.Get(sparseD0Path) for file in infiletask]
        for ifile in range(len(sparsesReco['RecoPrompt'])):
            sparsesReco['RecoPrompt'][ifile].GetAxis(axes_dict['RecoPrompt']['origin']).SetRange(2, 2)    # make sure it is prompt
            sparsesReco['RecoPrompt'][ifile].GetAxis(axes_dict['RecoPrompt']['cand_type']).SetRange(1, 2) # make sure it is signal

        sparsesReco['RecoFD'] = [file.Get(sparseD0Path) for file in infiletask]
        for ifile in range(len(sparsesReco['RecoFD'])):
            sparsesReco['RecoFD'][ifile].GetAxis(axes_dict['RecoPrompt']['origin']).SetRange(3, 3)       # make sure it is non-prompt
            sparsesReco['RecoFD'][ifile].GetAxis(axes_dict['RecoPrompt']['cand_type']).SetRange(1, 2)    # make sure it is signal

        sparsesReco['RecoRefl'] = [file.Get(sparseD0Path) for file in infiletask]
        for ifile in range(len(sparsesReco['RecoRefl'])):
            sparsesReco['RecoRefl'][ifile].GetAxis(axes_dict['RecoPrompt']['cand_type']).SetRange(3, 4)  # make sure it is reflection

        sparsesReco['RecoReflPrompt'] = [file.Get(sparseD0Path) for file in infiletask]
        for ifile in range(len(sparsesReco['RecoReflPrompt'])):
            sparsesReco['RecoReflPrompt'][ifile].GetAxis(axes_dict['RecoPrompt']['cand_type']).SetRange(3, 4)  # make sure it is reflection
            sparsesReco['RecoReflPrompt'][ifile].GetAxis(axes_dict['RecoPrompt']['origin']).SetRange(2, 2)       # make sure it is prompt   

        sparsesReco['RecoReflFD'] = [file.Get(sparseD0Path) for file in infiletask]
        for ifile in range(len(sparsesReco['RecoReflFD'])):
            sparsesReco['RecoReflFD'][ifile].GetAxis(axes_dict['RecoPrompt']['cand_type']).SetRange(3, 4)    # make sure it is reflection
            sparsesReco['RecoReflFD'][ifile].GetAxis(axes_dict['RecoPrompt']['origin']).SetRange(3, 3)       # make sure it is FD
        #TODO: safety checks for Dmeson reflecton and secondary peak

        sparsesGen['GenPrompt'] = [file.Get('hf-task-d0/hSparseAcc') for file in infiletask]
        for ifile in range(len(sparsesGen['GenPrompt'])):
            sparsesGen['GenPrompt'][ifile].GetAxis(axes_dict['GenPrompt']['origin']).SetRange(2, 2)  # make sure it is prompt

        sparsesGen['GenFD'] = [file.Get('hf-task-d0/hSparseAcc') for file in infiletask]
        for ifile in range(len(sparsesGen['GenFD'])):
            sparsesGen['GenFD'][ifile].GetAxis(axes_dict['GenFD']['origin']).SetRange(3, 3)  # make sure it is non-prompt
    elif config['Dmeson'] == 'Dplus':
        sparsesReco['RecoFD']     = [file.Get('hf-task-dplus/hSparseMassFD') for file in infiletask]
        sparsesReco['RecoPrompt'] = [file.Get('hf-task-dplus/hSparseMassPrompt') for file in infiletask]
        sparsesGen['GenPrompt']   = [file.Get('hf-task-dplus/hSparseMassGenPrompt') for file in infiletask]
        sparsesGen['GenFD']       = [file.Get('hf-task-dplus/hSparseMassGenFD') for file in infiletask]

    elif config['Dmeson'] == 'Ds':
        sparsesReco['RecoPrompt'] = [file.Get('hf-task-ds/MC/Ds/Prompt/hSparseMass') for file in infiletask]
        sparsesReco['RecoFD']     = [file.Get('hf-task-ds/MC/Ds/NonPrompt/hSparseMass') for file in infiletask]
        sparsesGen['GenPrompt']   = [file.Get('hf-task-ds/MC/Ds/Prompt/hSparseGen') for file in infiletask]
        sparsesGen['GenFD']       = [file.Get('hf-task-ds/MC/Ds/NonPrompt/hSparseGen') for file in infiletask]

    [infile.Close() for infile in infiletask]

    logger("Sparses loaded", level='INFO')
    if debug:
        print('\n')
        print('###############################################################')
        for key, value in axes_dict.items():
            logger(f"{key}:", level='DEBUG')
            for sub_key, sub_value in value.items():
                logger(f"    {sub_key}: {sub_value}", level='DEBUG')
        print('###############################################################')
        print('\n')
    return sparsesReco, sparsesGen, axes_dict
