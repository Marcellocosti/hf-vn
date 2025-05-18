'''
Script to project the MC distributions and apply the pt weights from the AnRes.root of Dtask
python3 proj_thn_mc.py config_flow.yml config_cutset.yml -o path/to/output -s text
                                                        --ptWeights path/to/file histName 
                                                        --ptWeightsB path/to/file histName
'''
import ROOT
import uproot
import yaml
import argparse
import sys
import os
from ROOT import TFile, TObject
from alive_progress import alive_bar
from scipy.interpolate import InterpolatedUnivariateSpline
sys.path.append(f"{os.path.dirname(os.path.abspath(__file__))}/utils")
from sparse_dicts import get_pt_preprocessed_sparses 
from utils import reweight_histo, get_vn_versus_mass

ROOT.TH1.AddDirectory(False)

def proj_data(sparses_dict, reso_dict, axes, inv_mass_bins, proj_scores, writeopt):

    proj_vars = ['Mass', 'sp']
    if proj_scores:
        proj_vars += ['score_FD', 'score_bkg']
    proj_axes = [axes['Flow'][var] for var in proj_vars]

    for var, ax in zip(proj_vars, proj_axes):
        for isparse, (_, sparse) in enumerate(sparses_dict.items()):
            # REVIEW: in case the Potential memory leak
            hist_var_temp = sparse.Projection(ax)
            hist_var_temp.SetName(f'hist_{var}_{isparse}')
            if isparse == 0:
                hist_var = hist_var_temp.Clone(f'hist_{var}')
                hist_var.Reset()

            hist_var.Add(hist_var_temp)

        hist_var.Write(f'h{var}_data', writeopt)

    hist_vn_sp = get_vn_versus_mass(sparses_dict, reso_dict, inv_mass_bins, axes['Flow']['Mass'], axes['Flow']['sp'])

def proj_mc_reco(sparsesReco, ptWeights, ptWeightsB, Bspeciesweights, sPtWeights, sPtWeightsB, writeopt):

    for key, sparse in sparsesReco.items():
        if key != 'RecoPrompt' and key != 'RecoFD':
            for iProjVar in ('Mass', 'Pt'):
                sparse.Projection(axes[key][iProjVar]).Write(f'h{key}{iProjVar}')

    hMassPrompt = sparsesReco['RecoPrompt'].Projection(axes['RecoPrompt']['Mass'])
    hMassPrompt.SetName(f'hPromptMass_{ptMin}_{ptMax}')
    hMassFD = sparsesReco['RecoFD'].Projection(axes['RecoFD']['Mass'])
    hMassFD.SetName(f'hFDMass_{ptMin}_{ptMax}')

    ### project pt prompt
    hPtPrompt = sparsesReco['RecoPrompt'].Projection(axes['RecoPrompt']['Pt'])
    if ptWeights:
        hPtPrompt = reweight_histo(hPtPrompt, sPtWeights, 'hPromptPt') 
    ### project pt FD
    if ptWeightsB:
        if Bspeciesweights:
            hPtFD = reweight_histo(sparsesReco['RecoFD'].Projection(axes['RecoFD']['Pt'], axes['RecoFD']['pt_bmoth'], axes['RecoFD']['flag_bhad']), 
                                   sPtWeightsB, 'hFDPt', Bspeciesweights)
        else:
            hPtFD = reweight_histo(sparsesReco['RecoFD'].Projection(axes['RecoFD']['pt_bmoth'], axes['RecoFD']['Pt']), sPtWeightsB, 'hFDPt') # 2D projection: Projection(ydim, xdim)
    elif ptWeights:
        hPtFD = reweight_histo(sparsesReco['RecoFD'].Projection(axes['RecoFD']['Pt']), sPtWeights, 'hFDPt')
    elif Bspeciesweights:
        hPtFD = reweight_histo(sparsesReco['RecoFD'].Projection(axes['RecoFD']['flag_bhad'], axes['RecoFD']['Pt']), [], 'hFDPt', Bspeciesweights) # 2D projection: Projection(ydim, xdim)
    else:
        hPtFD = sparsesReco['RecoFD'].Projection(axes['RecoFD']['Pt'])

    ## write the output 
    hMassPrompt.Write('hPromptMass', writeopt)
    hMassFD.Write('hFDMass', writeopt)
    hPtPrompt.Write('hPromptPt', writeopt)
    hPtFD.Write('hFDPt', writeopt)

def proj_mc_gen(sparsesGen, ptWeights, ptWeightsB, Bspeciesweights, sPtWeights, sPtWeightsB, writeopt):

    for key, sparse in sparsesGen.items():
        if key != 'GenPrompt' and key != 'GenFD':
            sparse.Projection(axes[key]['Pt']).Write(f'h{key}Pt')

    ### prompt
    hGenPtPrompt = sparsesGen['GenPrompt'].Projection(axes['GenPrompt']['Pt'])
    if ptWeights:
        hGenPtPrompt = reweight_histo(hGenPtPrompt, sPtWeights, 'hPromptGenPt')
    ### FD
    if ptWeightsB:
        if Bspeciesweights:
            hGenPtFD = reweight_histo(sparsesGen['GenFD'].Projection(axes['GenFD']['Pt'], axes['GenFD']['pt_bmoth'], axes['GenFD']['flag_bhad']), 
                                      sPtWeightsB, 'hFDGenPt', Bspeciesweights)
        else:
            hGenPtFD = reweight_histo(sparsesGen['GenFD'].Projection(axes['GenFD']['pt_bmoth'], axes['GenFD']['Pt']), sPtWeightsB, 'hFDGenPt') # 2D projection: Projection(ydim, xdim)
    elif ptWeights:
        hGenPtFD = reweight_histo(sparsesGen['GenFD'].Projection(axes['GenFD']['Pt']), sPtWeights, 'hFDGenPt')
    elif Bspeciesweights:
        hGenPtFD = reweight_histo(sparsesGen['GenFD'].Projection(axes['GenFD']['flag_bhad'], axes['GenFD']['Pt']), [], 'hFDPt', Bspeciesweights) # 2D projection: Projection(ydim, xdim)
    else:
        hGenPtFD = sparsesGen['GenFD'].Projection(axes['GenFD']['Pt'])

    ## write the output
    hGenPtPrompt.Write('hPromptGenPt', writeopt)
    hGenPtFD.Write('hFDGenPt', writeopt)

def pt_weights_info(ptweights, ptweightsB):
    """Get pt weights and return weights flags with spline

    Args:
        ptweights (list): [file path, histogram name] for pt weights
        ptweightsB (list): [file path, histogram name] for B pt weights

    Outputs:
        ptWeights (bool): ptWeights flag
        ptWeightsB (bool): ptWeightsB flag
        Bspeciesweights (str): B species weights # TODO
        sPtWeights (spline): Spline for ptWeights interpolation
        sPtWeightsB (spline): Spline for ptWeightsB weights interpolation
    """

    # REVIEW: the ptWeights inputed is a list, but the ptWeights outputed is a TH1D object
    # and actually ptweights is used as a flag
        # compute info for pt weights
    if ptweights != []:
        with uproot.open(ptweights[0]) as f:
            hPtWeights = f[ptweights[1]]
            bins = hPtWeights.axis(0).edges()
            ptCentW = [(bins[iBin]+bins[iBin+1])/2 for iBin in range(len(bins)-1)]
            sPtWeights = InterpolatedUnivariateSpline(ptCentW, hPtWeights.values())
        ptWeights = True
    else:
        print('\033[91m WARNING: pt weights will not be provided! \033[0m')
        ptWeights = False
        sPtWeights = None

    if ptweightsB != []:
        with uproot.open(ptweightsB[0]) as f:
            hPtWeightsB = f[ptweightsB[1]]
            bins = hPtWeightsB.axis(0).edges()
            ptCentWB = [(bins[iBin]+bins[iBin+1])/2 for iBin in range(len(bins)-1)]
            sPtWeightsB = InterpolatedUnivariateSpline(ptCentWB, hPtWeightsB.values())
        ptWeightsB = True
    else:
        print('\033[91m WARNING: B weights will not not be provided! \033[0m')
        ptWeightsB = False
        sPtWeightsB = None

    if config.get('Bspeciesweights'):
        Bspeciesweights = config['Bspeciesweights']
    else:
        print('\033[91m WARNING: B species weights will not be provided! \033[0m')
        Bspeciesweights = None
    
    return ptWeights, ptWeightsB, Bspeciesweights, sPtWeights, sPtWeightsB

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Arguments")
    parser.add_argument("config", metavar="text",
                        default="config.yaml", help="flow configuration file")
    parser.add_argument('--cutsetConfig', "-cc", metavar='text', type=str, nargs='?',
                        const=None, default='cutsetConfig.yaml',
                        help='Optional cutset configuration file (default: cutsetConfig.yaml)')
    parser.add_argument("--correlated", "-c", action="store_true", 
                        help="Produce yml files for correlated cuts")
    args = parser.parse_args()

    with open(args.config, 'r') as ymlCfgFile:
        config = yaml.load(ymlCfgFile, yaml.FullLoader)
    operations = config["operations"]

    with open(args.cutsetConfig, 'r') as ymlCutSetFile:
        cutSetCfg = yaml.load(ymlCutSetFile, yaml.FullLoader)
        iCut = f"{int(cutSetCfg['icutset']):02d}"
    cutVars = cutSetCfg['cutvars']

    method = "correlated" if args.correlated else "combined"
    outfilename = config["out_dir"] + f'/cutvar_{config["suffix"]}_{method}/proj/proj_{iCut}'
    create_new_file = True
    write_opt_data = 0
    write_opt_mc = 0
    if operations["proj_data"] and operations["proj_mc"]:
        print(f"Creating new file and project data and mc!")
        outfile = TFile(outfilename + '.root', 'RECREATE')
    else:
        projFiles = [f'{outfilename}*' for file in os.listdir(f'{config["out_dir"]}/cutvar_test_correlated/proj') if file.endswith('.root')]
        if len(projFiles) == 0:
            print(f"No existing previous projections, creating new file and project data ({operations["proj_data"]}) or mc ({operations["proj_mc"]:})!")
            outfile = TFile(outfilename + '.root', 'RECREATE')
        else:
            create_new_file = False
            print(f"Found previous projections, updating existing file!")
            outfile = TFile.Open(outfilename + '.root', 'UPDATE')
            if operations["proj_data"]:
                write_opt_data = TObject.kOverwrite 
            if operations["proj_mc"]:
                write_opt_mc = TObject.kOverwrite

    # # compute info for pt weights
    ptweightsPath = config["projections"]["ptweightspath"]
    ptWeights, ptWeightsB, Bspeciesweights, sPtWeights, sPtWeightsB = None, None, None, None, None
    #     ptWeights, ptWeightsB, Bspeciesweights, sPtWeights, sPtWeightsB = pt_weights_info(args.ptweights, args.ptweightsB)

    with alive_bar(len(cutSetCfg['Pt']['min']), title='Processing pT bins') as bar:
        for iPt, (ptMin, ptMax) in enumerate(zip(cutSetCfg['Pt']['min'], cutSetCfg['Pt']['max'])):
            print(f'Projecting distributions for {ptMin:.1f} < pT < {ptMax:.1f} GeV/c')
            sparsesFlow, sparsesReco, sparsesGen, axes, resolutions = get_pt_preprocessed_sparses(config, iPt)
            ptdir = f'pt_{int(ptMin*10)}_{int(ptMax*10)}'
            if create_new_file:
                print(f"creating new directory: {ptdir}")
                outfile.mkdir(ptdir)
            outfile.cd(ptdir)

            # Cut on centrality and pt on data applied in the preprocessing
            if operations["proj_data"]:
                for key, sparse in sparsesFlow.items():
                    sparse.GetAxis(axes['Flow']['score_FD']).SetRangeUser(cutVars['score_FD']['min'][iPt], cutVars['score_FD']['max'][iPt])
                    sparse.GetAxis(axes['Flow']['score_bkg']).SetRangeUser(cutVars['score_bkg']['min'][iPt], cutVars['score_bkg']['max'][iPt])
                proj_data(sparsesFlow, resolutions, axes, config["projections"]['inv_mass_bins'][iPt], config["projections"].get('storeML'), write_opt_data)
                print(f"Projected data!")
            else:
                print("Kept data from previous projections!")

            # Cut on centrality on mc applied in the preprocessing
            if operations["proj_mc"]:
                for key, iSparse in sparsesReco.items():
                    iSparse.GetAxis(axes[key]['score_FD']).SetRangeUser(cutVars['score_FD']['min'][iPt], cutVars['score_FD']['max'][iPt])
                    iSparse.GetAxis(axes[key]['score_bkg']).SetRangeUser(cutVars['score_bkg']['min'][iPt], cutVars['score_bkg']['max'][iPt])

                proj_mc_reco(sparsesReco, ptWeights, ptWeightsB, Bspeciesweights, sPtWeights, sPtWeightsB, write_opt_mc)
                print("Projected mc reco!")
                proj_mc_gen(sparsesGen, ptWeights, ptWeightsB, Bspeciesweights, sPtWeights, sPtWeightsB, write_opt_mc)
                print("Projected mc gen!")
            else:
                print("Kept mc from previous projections!")

            bar()
    
    outfile.Close()
