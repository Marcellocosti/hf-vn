'''
Script to create a yaml file with a set of cuts for ML
python3 make_yaml_for_ml.py config_flow.yml -o path/to/output -s text

'''
import yaml
import argparse
import os
import numpy as np
import sys
script_dir = os.path.dirname(os.path.abspath(__file__))  # Get script's directory
sys.path.append(os.path.abspath(os.path.join(script_dir, '..')))  # Append parent directory

def pad_to_length(list, target_len):
    '''
        Function to pad a list to a target length
        Args:
            lst (list): list to be padded
            target_len (int): target length of the list
        Returns:
            list: padded list
    '''
    list_length = len(list)
    len_offset = target_len - list_length
    return list + [list[-1]] * len_offset if list_length < target_len else list

def make_yaml(flow_config, outputdir, correlated):
    '''
        Function to create a yaml file with a set of cuts for ML
        Args:
            flow_config (str): path to the flow config file
            outputdir (str): path to the output directory
    '''
    with open(flow_config, 'r') as f:
        cfg = yaml.safe_load(f)

    ptmins = cfg['ptbins'][:-1]
    ptmaxs = cfg['ptbins'][1:]
    nPtBins = len(ptmins)

    cfg_cutvar = cfg['cut_variation']
    if correlated:
        sig = cfg_cutvar['corr_bdt_cut']['sig']
        sig_cuts_lower = [list(np.arange(sig['min'][i], sig['max'][i], sig['step'][i])) for i in range(nPtBins)]
        sig_cuts_upper = [[1.0] * len(cuts) for cuts in sig_cuts_lower]
    else:
        sig = cfg_cutvar['uncorr_bdt_cut']['sig']
        sig_cuts_lower = [sig[i][:-1] for i in range(nPtBins)]
        sig_cuts_upper = [sig[i][1:]  for i in range(nPtBins)]

    # Determine the maximum number of cut sets across pt bins and pad all to uniform length
    maxCutSets = max(len(cuts) for cuts in sig_cuts_lower)
    sig_cuts_lower = [pad_to_length(cuts, maxCutSets) for cuts in sig_cuts_lower]
    sig_cuts_upper = [pad_to_length(cuts, maxCutSets) for cuts in sig_cuts_upper]
    # Transpose: convert [iPt][iCut] → [iCut][iPt]
    sig_cuts_lower = list(map(list, zip(*sig_cuts_lower)))
    sig_cuts_upper = list(map(list, zip(*sig_cuts_upper)))

    if correlated:
        bkg_cuts_upper = [cfg_cutvar['corr_bdt_cut']['bkg_max']] * maxCutSets
    else:
        bkg_cuts_upper = [pad_to_length(cuts, maxCutSets) for cuts in cfg_cutvar['uncorr_bdt_cut']['bkg_max']]
        bkg_cuts_upper  = list(map(list, zip(*bkg_cuts_upper)))

    os.makedirs(f'{outputdir}/config', exist_ok=True)
    for iCut, (bkg_maxs, fd_mins, fd_maxs) in enumerate(zip(bkg_cuts_upper, sig_cuts_lower, sig_cuts_upper)):
        bkg_max = list(map(float, bkg_maxs))
        fd_min  = list(map(float, fd_mins))
        fd_max  = list(map(float, fd_maxs))

        if len(fd_min) != len(ptmins) or len(fd_max) != len(ptmins) or len(bkg_max) != len(ptmins):
            raise ValueError(f"Length of fd_min or fd_max or bkg_max does not match length of ptmins: {len(fd_min)} != {len(ptmins)}")

        combinations = {
            'icutset': iCut,
            'cutvars': {
                'Pt': {'min': ptmins, 'max': ptmaxs},
                'score_bkg': {'min': [0.0] * len(ptmins), 'max': bkg_max},
                'score_FD': {'min': fd_min, 'max': fd_max},
            },
            'fitrangemin': cfg['MassMin'],
            'fitrangemax': cfg['MassMax'],
        }

        with open(f'{outputdir}/config/cutset_{iCut:02}.yml', 'w') as file:
            yaml.dump(combinations, file, default_flow_style=False)

    print(f'Yaml files are saved in {outputdir}/config')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Arguments')
    parser.add_argument('flow_config', metavar='text', default='config_flow.yml')
    parser.add_argument("--outputdir", "-o", metavar="text", default=".", help="output directory")
    parser.add_argument("--correlated", "-c", action="store_true", help="Produce yml files for correlated cuts")
    args = parser.parse_args()

    make_yaml(args.flow_config, args.outputdir, args.correlated)
