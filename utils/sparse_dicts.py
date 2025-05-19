import ROOT
import yaml
from ROOT import TFile # pyright: ignore # type: ignore

def get_sparses_dicts(config):
    
    axes_dict = {}
    ### Data sparse dictionary
    axes_dict['Flow'] = {
        'Mass': 0,
        'Pt': 1,
        'cent': 2,
        'sp': 3,
        'score_bkg': 4,
        'score_FD': 5,
        'occ': 6
    }
    ### MC sparse dictionary
    if config['Dmeson'] == 'Dzero':
        axes_dict['RecoPrompt'] = {
            'score_bkg': 0,
            'score_prompt': 1,
            'score_FD': 2,
            'Mass': 3,
            'Pt': 4,
            'y': 5,
            'cand_type': 6,
            'pt_bmoth': 7,
            'origin': 8,
            'npvcontr': 9,
            'cent': 10,
            'occ': 11,
        }
        axes_dict['RecoFD'] = axes_dict['RecoPrompt']
        axes_gen = {
            'Pt': 0,
            'pt_bmoth': 1,
            'y': 2,
            'origin': 3,
            'npvcontr': 4,
            'cent': 5,
            'occ': 6
        }
    elif config['Dmeson'] == 'Dplus':
        axes_dict['RecoPrompt'] = {
            'Mass': 0,
            'Pt': 1,
            'score_bkg': 2,
            'score_prompt': 3,
            'score_FD': 4,
            'cent': 5,
            'occ': 6,
        }
        axes_dict['RecoFD'] = {
            'Mass': 0,
            'Pt': 1,
            'score_bkg': 2,
            'score_prompt': 3,
            'score_FD': 4,
            'cent': 5,
            'occ': 6,
            'pt_bmoth': 7,
            'flag_bhad': 8,
        }
        axes_dict['GenPrompt'] = {
            'Pt': 0,
            'y': 1,
            'cent': 2,
            'occ': 3
        }
        axes_dict['GenFD'] = {
            'Pt': 0,
            'y': 1,
            'cent': 2,
            'occ': 3,
            'pt_bmoth': 4,
            'flag_bhad': 5,
        }

    elif config['Dmeson'] == 'Ds':
        axes_dict['RecoPrompt'] = {
            'Mass': 0,
            'Pt': 1,
            'cent': 3,  # Check number 2
            'npvcontr': 4,
            'score_bkg': 5,
            'score_prompt': 6,
            'score_FD': 7,
            'occ': 8,
        }
        axes_dict['RecoFD'] = {
            'Mass': 0,
            'Pt': 1,
            'cent': 2,
            'score_bkg': 3,
            'score_prompt': 4,
            'score_FD': 5,
            'pt_bmoth': 6,
            'flag_bhad': 7,
            'occ': 8
        }
        axes_dict['GenPrompt'] = {
            'Pt': 0,
            'y': 1,
            'npvcontr': 2,
            'cent': 3,
            'occ': 4
        }
        axes_dict['GenFD'] = {
            'Pt': 0,
            'y': 1,
            'cent': 2,
            'pt_bmoth': 3,
            'flag_bhad': 4,
            'occ': 5
        }

    return axes_dict

def get_pt_preprocessed_sparses(config, iPt):
    
    print("Loading preprocessed sparses")
    sparsesFlow, sparsesReco, sparsesGen, axes_dict, resolutions = {}, {}, {}, {}, {}
    pre_cfg = config['preprocess']
    axes_dict['Flow'] = {ax: iax for iax, (ax, _) in enumerate(pre_cfg['axes_data'].items())}
    ptmin = config["ptbins"][iPt]
    ptmax = config["ptbins"][iPt+1]

    infileprep = TFile(f"{config["out_dir"]}/preprocess/AnalysisResults_pt_{int(ptmin*10)}_{int(ptmax*10)}.root")

    if config["operations"].get("proj_data"):
        for key, dataset in pre_cfg["data"].items():
            sparsesFlow[f'Flow_{key}'] = infileprep.Get(f'Data_Flow_{key}/hSparseFlowCharm')
            resolutions[f'Reso_Flow_{key}'] = infileprep.Get(f'Data_Flow_{key}/hResolution')

    if config["operations"].get("proj_mc"):
        subdir = infileprep.Get("MC/Reco")
        for key in subdir.GetListOfKeys():
            obj = key.ReadObj()
            sparsesReco[key.GetName()[1:]] = obj
            axes_dict[key.GetName()[1:]] = {ax: iax for iax, (ax, _) in enumerate(pre_cfg['axes_reco'].items())}

        subdir = infileprep.Get("MC/Gen")
        for key in subdir.GetListOfKeys():
            obj = key.ReadObj()
            sparsesGen[key.GetName()[1:]] = obj
            axes_dict[key.GetName()[1:]] = {ax: iax for iax, (ax, _) in enumerate(pre_cfg['axes_gen'].items())}

    infileprep.Close()

    return sparsesFlow, sparsesReco, sparsesGen, axes_dict, resolutions

def get_sparses(config, debug=False):
    """Load the sparses and axes infos

    Args:
        config (dict): the flow config dictionary
        debug (bool, optional): print debug info. Defaults to False.

    Outputs:
        sparsesFlow: thnSparse in the flow task
        sparsesReco: thnSparse of reco level from the D meson task
        sparsesGen: thnSparse of gen level from the D meson task
        axes_dict (dict): dictionary of the axes for each sparse
    """

    print("Getting sparses")

    sparsesFlow, sparsesReco, sparsesGen, resolutions = {}, {}, {}, {}
    axes_dict = get_sparses_dicts(config)
    
    pre_cfg = config['preprocess']
    if config["operations"].get("preprocess_data"):
        infileflow = []
        resolutions = {}
        for name, dataset in pre_cfg['data'].items():
            infileflow = [TFile(dataset["files"])] if isinstance(dataset["files"], str) else [TFile(file) for file in dataset["files"]]
            sparsesFlow[f'Flow_{name}'] = [infile.Get('hf-task-flow-charm-hadrons/hSparseFlowCharm') for infile in infileflow]
            [infile.Close() for infile in infileflow]
            resofile = TFile.Open(dataset["resolution"], 'r')
            det_A = config.get('detA', 'FT0c')
            det_B = config.get('detB', 'FV0a')
            det_C = config.get('detC', 'TPCtot')
            resolutions[f"Reso_Flow_{name}"] = resofile.Get(f'{det_A}_{det_B}_{det_C}/histo_reso_delta_cent')
            resolutions[f"Reso_Flow_{name}"].SetDirectory(0)
            resofile.Close()

    if config["operations"].get("preprocess_mc"):
        print(f"Loading mc sparse from: {pre_cfg['mc']}")
        infiletask = [TFile(pre_cfg['mc'])] if isinstance(pre_cfg['mc'], str) else [TFile(pre_cfg['mc']) for file in pre_cfg['mc']]

        if config['Dmeson'] == 'Dzero':
            sparseD0Path = 'hf-task-d0/hBdtScoreVsMassVsPtVsPtBVsYVsOriginVsD0Type'
            sparsesReco['RecoPrompt'] = infiletask.Get(sparseD0Path)
            sparsesReco['RecoPrompt'].GetAxis(axes_reco['origin']).SetRange(2, 2)    # make sure it is prompt
            sparsesReco['RecoPrompt'].GetAxis(axes_reco['cand_type']).SetRange(1, 2) # make sure it is signal
            axes_dict['RecoPrompt'] = axes_reco

            sparsesReco['RecoFD'] = infiletask.Get(sparseD0Path)
            sparsesReco['RecoFD'].GetAxis(axes_reco['origin']).SetRange(3, 3)       # make sure it is non-prompt
            sparsesReco['RecoFD'].GetAxis(axes_reco['cand_type']).SetRange(1, 2)    # make sure it is signal
            axes_dict['RecoFD'] = axes_reco

            sparsesReco['RecoRefl'] = infiletask.Get(sparseD0Path)
            sparsesReco['RecoRefl'].GetAxis(axes_reco['cand_type']).SetRange(3, 4)  # make sure it is reflection
            axes_dict['RecoRefl'] = axes_reco

            sparsesReco['RecoReflPrompt'] = infiletask.Get(sparseD0Path)
            sparsesReco['RecoReflPrompt'].GetAxis(axes_reco['cand_type']).SetRange(3, 4)    # make sure it is reflection
            sparsesReco['RecoReflPrompt'].GetAxis(axes_reco['origin']).SetRange(2, 2)       # make sure it is prompt
            axes_dict['RecoReflPrompt'] = axes_reco

            sparsesReco['RecoReflFD'] = infiletask.Get(sparseD0Path)
            sparsesReco['RecoReflFD'].GetAxis(axes_reco['cand_type']).SetRange(3, 4)    # make sure it is reflection
            sparsesReco['RecoReflFD'].GetAxis(axes_reco['origin']).SetRange(3, 3)       # make sure it is FD
            axes_dict['RecoReflFD'] = axes_reco
            #TODO: safety checks for Dmeson reflecton and secondary peak

            sparsesGen['GenPrompt'] = infiletask.Get('hf-task-d0/hSparseAcc')
            sparsesGen['GenPrompt'].GetAxis(axes_gen['origin']).SetRange(2, 2)  # make sure it is prompt
            axes_dict['GenPrompt'] = axes_gen

            sparsesGen['GenFD'] = infiletask.Get('hf-task-d0/hSparseAcc')
            sparsesGen['GenFD'].GetAxis(axes_gen['origin']).SetRange(3, 3)  # make sure it is non-prompt
            axes_dict['GenFD'] = axes_gen
            #TODO: safety checks for Dmeson reflecton and secondary peak
        elif config['Dmeson'] == 'Dplus':
            if isinstance(infiletask, list):
                sparsesReco['RecoFD'] = [file.Get('hf-task-dplus/hSparseMassFD') for file in infiletask]
                sparsesReco['RecoPrompt'] = [file.Get('hf-task-dplus/hSparseMassPrompt') for file in infiletask]
                sparsesGen['GenPrompt'] = [file.Get('hf-task-dplus/hSparseMassGenPrompt') for file in infiletask]
                sparsesGen['GenFD'] = [file.Get('hf-task-dplus/hSparseMassGenFD') for file in infiletask]

        elif config['Dmeson'] == 'Ds':
            sparsesReco['RecoPrompt'] = infiletask.Get('hf-task-ds/MC/Ds/Prompt/hSparseMass')
            sparsesReco['RecoFD'] = infiletask.Get('hf-task-ds/MC/Ds/NonPrompt/hSparseMass')
            sparsesGen['GenPrompt'] = infiletask.Get('hf-task-ds/MC/Ds/Prompt/hSparseGen')
            sparsesGen['GenFD'] = infiletask.Get('hf-task-ds/MC/Ds/NonPrompt/hSparseGen')

        [infile.Close() for infile in infiletask]

    print(f"Loaded sparses!")
    if debug:
        print('\n')
        print('###############################################################')
        for key, value in axes_dict.items():
            print(f"{key}:")
            for sub_key, sub_value in value.items():
                print(f"    {sub_key}: {sub_value}")
        print('###############################################################')
        print('\n')
    return sparsesFlow, sparsesReco, sparsesGen, axes_dict, resolutions
