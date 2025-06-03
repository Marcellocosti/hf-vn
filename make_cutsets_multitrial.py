import yaml
import ROOT
import copy
import os
import numpy as np
import argparse
import itertools
import copy
from alive_progress import alive_bar # type: ignore

def get_reference_config_pt_bin(config_flow, iPtBin):
    # Deepcopy to ensure that we are working with a copy and not modifying the original config_flow
    pt_bin_config = copy.deepcopy(config_flow)

    # Separate bin-specific keys (those with lists as values) from common settings
    bin_specific_keys = {key: value for key, value in pt_bin_config.items() if isinstance(value, list)}

    # Keep the common settings (those that are not bin-specific)
    for key, value in pt_bin_config.items():
        if key not in bin_specific_keys:
            pt_bin_config[key] = value  # Keep common settings

    # Assign bin-specific values
    for key, values in bin_specific_keys.items():
        if key == "axestokeep":
            pt_bin_config[key] = values
        elif key != "axestokeep" and key != "flow_files":
            # this means it is a general setting
            if len(values) < len(config_flow["ptmins"]) and key.startswith("Templs"):
                pt_bin_config[key] = values
            else:
                pt_bin_config[key] = [values[iPtBin]]

    # Modify cut_variation values but leave the original config_flow intact
    if pt_bin_config.get("cut_variation"):
        pt_bin_config["cut_variation"]["corr_bdt_cut"]["bkg_max"] = [pt_bin_config["cut_variation"]["corr_bdt_cut"]["bkg_max"][iPtBin]]
        pt_bin_config["cut_variation"]["corr_bdt_cut"]["sig"]["max"] = [pt_bin_config["cut_variation"]["corr_bdt_cut"]["sig"]["max"][iPtBin]]
        pt_bin_config["cut_variation"]["corr_bdt_cut"]["sig"]["min"] = [pt_bin_config["cut_variation"]["corr_bdt_cut"]["sig"]["min"][iPtBin]]
        pt_bin_config["cut_variation"]["corr_bdt_cut"]["sig"]["step"] = [pt_bin_config["cut_variation"]["corr_bdt_cut"]["sig"]["step"][iPtBin]]
        pt_bin_config["cut_variation"]["uncorr_bdt_cut"]["bkg_max"] = [pt_bin_config["cut_variation"]["uncorr_bdt_cut"]["bkg_max"][iPtBin]]
        pt_bin_config["cut_variation"]["uncorr_bdt_cut"]["sig"] = [pt_bin_config["cut_variation"]["uncorr_bdt_cut"]["sig"][iPtBin]]
    
    return pt_bin_config

def modify_yaml_bdt(config_flow, config_mod, output_dir):
    
    with open(config_flow, 'r') as CfgFlow:
        cfg_flow = yaml.safe_load(CfgFlow)
    with open(config_mod, 'r') as CfgMod:
        cfg_mod = yaml.safe_load(CfgMod)

    os.makedirs(f"{output_dir}/", exist_ok=True)
    os.makedirs(f"{output_dir}/config_history/", exist_ok=True)

    with open(os.path.join(f"{output_dir}/config_history/", f"config_reference.yml"), 'w') as out_file:
        yaml.dump(cfg_flow, out_file, default_flow_style=False)
    with open(os.path.join(f"{output_dir}/config_history/", f"config_modifications.yml"), 'w') as out_file:
        yaml.dump(cfg_mod, out_file, default_flow_style=False)

    for iPtBin, pt_bin in enumerate(cfg_mod['ptbins']):
        
        ptmin = pt_bin['range'][0]
        ptmax = pt_bin['range'][1]
        pt_bin_index = cfg_flow['ptmins'].index(ptmin)
        ref_config_ptbin = get_reference_config_pt_bin(cfg_flow, pt_bin_index)
        
        os.makedirs(f"{output_dir}/pt_{int(ptmin*10)}_{int(ptmax*10)}/config_sys/", exist_ok=True)
        output_file = os.path.join(f"{output_dir}/pt_{int(ptmin*10)}_{int(ptmax*10)}/config_sys/", f"config_reference.yml")
        with open(output_file, 'w') as out_file:
            yaml.dump(ref_config_ptbin, out_file, default_flow_style=False)

        varied_configs = {}
        print(f"ptbin: {pt_bin}")
        for var in pt_bin:
            if var != "range" and var != "inv_mass_bins_steps" and var != "MaxChi2" and var != "MinSignificance" and var != "MaxSignificance" and var != "AnchorTemplsMode":
                print(f"var: {var}")
                varied_configs[var] = list(set(ref_config_ptbin[var] + pt_bin[var]))
            elif var == "inv_mass_bins_steps":
                varied_configs[var] = pt_bin[var]
            elif var == "AnchorTemplsMode":
                varied_configs[var] = pt_bin[var]
                

        # Perform itertools.product to get all combinations
        keys = list(varied_configs.keys())
        values = list(varied_configs.values())
        combinations = list(itertools.product(*values))
        config_variants = [dict(zip(keys, combination)) for combination in combinations] 

        # Print results
        for idx, variant in enumerate(config_variants):
            cfg_variant = copy.deepcopy(ref_config_ptbin)
            for varied_var in variant:
                if varied_var != "inv_mass_bins_steps" and varied_var != 'AnchorTemplsMode':
                    if isinstance(ref_config_ptbin[varied_var], list):
                        cfg_variant[varied_var] = [variant[varied_var]]
                    else:
                        cfg_variant[varied_var] = variant[varied_var]
                if varied_var == 'AnchorTemplsMode' and variant['AnchorTemplsMode'] != -1:
                    cfg_variant['AnchorTemplsMode'] = variant['AnchorTemplsMode']
                    cfg_variant['IncludeTempls'] = True
            cfg_variant["inv_mass_bins"] = [np.arange(variant['MassMin'], variant['MassMax']+variant['inv_mass_bins_steps'], variant['inv_mass_bins_steps']).tolist()]
            cfg_variant["out_dir"] = f"{output_dir}/pt_{int(ptmin*10)}_{int(ptmax*10)}/trails/"
            cfg_variant["suffix"] = f"{idx}"
            cfg_variant["nworkers"] = 1 # parallelization is in the bash script
            cfg_variant["MaxChi2"] = pt_bin['MaxChi2']
            cfg_variant["MinSignificance"] = pt_bin['MinSignificance']
            cfg_variant["MaxSignificance"] = pt_bin['MaxSignificance']
            if cfg_variant["MassMin"][0] > 1.75 or cfg_variant["MassMax"][0] < 1.95:
                # else the region used for the prefit of the bkg 
                # function would be empty, leading to a crash
                cfg_variant["NSigma4SB"] = [2]

            output_file = os.path.join(f"{output_dir}/pt_{int(ptmin*10)}_{int(ptmax*10)}/config_sys/", f"config_var_{idx}.yml")
            with open(output_file, 'w') as out_file:
                yaml.dump(cfg_variant, out_file, default_flow_style=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Arguments')
    parser.add_argument('input_config', metavar='text', default='config_Ds_Fit.yml')
    parser.add_argument('--modifications_config', "-m", metavar='text', default='')
    parser.add_argument("--outputdir", "-o", metavar="text", default=".", help="output directory")
    parser.add_argument("--multitrial_bdt", "-mb", action="store_true", default=False,
                        help="multitrial systematics for BDT")
    args = parser.parse_args()

    modify_yaml_bdt(args.input_config,
                    args.modifications_config,
                    args.outputdir)