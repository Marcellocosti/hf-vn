import argparse
import os
import re
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
import yaml
import ROOT
ROOT.gErrorIgnoreLevel = ROOT.kWarning
import array
import time
os.environ["CUDA_VISIBLE_DEVICES"] = "" # pylint: disable=wrong-import-position
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
from ROOT import TFile, TH1F, TGraphAsymmErrors, kBlack, kFullCircle
script_dir = os.path.dirname(os.path.realpath(__file__))
os.sys.path.append(os.path.join(script_dir, '..', 'utils'))
from utils import logger, make_dir_root_file
from load_utils import load_aod_file
from StyleFormatter import SetGlobalStyle, SetObjectStyle
from matplotlib import gridspec
ROOT.gROOT.SetBatch(True)
import multiprocessing as mp
import sys
sys.path.append("./flareflyfitter/")
from raw_yield_fitter import RawYieldFitter
mp.set_start_method("spawn", force=True)
msg_service = ROOT.RooMsgService.instance()


def fit_pt_bin(cfg_flow, cfg_cutset, proj_file_path, i_pt, pt_min, pt_max, outdir, is_multitrial):

    pt_label = f"pt_{int(pt_min*10)}_{int(pt_max*10)}"
    proj_file = TFile.Open(proj_file_path, "READ")
    hist_mass = proj_file.Get(f"{pt_label}/hMassData")
    hist_vn_vs_mass = proj_file.Get(f"{pt_label}/hVnVsMassData")

    stats = {}

    # Monitor time for reading main config
    t0 = time.perf_counter()
    # Initialize the fitter for sp-integrated yield extraction
    fit_cfg = cfg_flow['v2extraction']
    fitter = RawYieldFitter(cfg_flow['Dmeson'], pt_min, pt_max, f"{pt_label}_fit",
                            fit_cfg['Minimizer'], fit_vn_vs_mass=True, verbose = not is_multitrial)

    t1 = time.perf_counter()
    logger(f"Time to read main config: {t1 - t0} s", "INFO")

    fitter.set_fit_range(fit_cfg['MassFitRanges'][i_pt][0], fit_cfg['MassFitRanges'][i_pt][1])
    fitter.set_rebin(fit_cfg['Rebin'][i_pt] if isinstance(fit_cfg['Rebin'], list) else fit_cfg['Rebin'])
    t2 = time.perf_counter()

    # Add model components
    fitter.add_bkg_func(fit_cfg['BkgFunc'][i_pt] if isinstance(fit_cfg['BkgFunc'], list) else fit_cfg['BkgFunc'], "Comb. bkg",
                        vn_func=fit_cfg['BkgFuncVn'][i_pt] if isinstance(fit_cfg['BkgFuncVn'], list) else fit_cfg['BkgFuncVn'])
    sgn_funcs = {} # More info for signal functions, a dictionary is better
    sgn_funcs[fit_cfg['SgnFuncLabel']] = {
        'mass_func': fit_cfg['SgnFunc'][i_pt] if isinstance(fit_cfg['SgnFunc'], list) else fit_cfg['SgnFunc'],
        'vn_func': 'kConst',
        'part': cfg_flow['Dmeson']
    }
    if fit_cfg.get('InclSecPeak'):
        include_sec_peak = fit_cfg['InclSecPeak'][i_pt] if isinstance(fit_cfg['InclSecPeak'], list) else fit_cfg['InclSecPeak']
        if include_sec_peak:
            sgn_funcs[fit_cfg['SgnFuncSecPeakLabel']] = {
                'mass_func': fit_cfg['SgnFuncSecPeak'][i_pt] if isinstance(fit_cfg['SgnFuncSecPeak'], list) else fit_cfg['SgnFuncSecPeak'],
                'vn_func': 'kConst',
                'part': 'Dplus' if cfg_flow['Dmeson'] == 'Ds' else 'Dstar',
            }

    for i_sgn, (label, sgn_func) in enumerate(sgn_funcs.items()):
        fitter.add_sgn_func(sgn_func['mass_func'], label, sgn_func['part'], sgn_func['vn_func'])
    sgn_func_label = fit_cfg['SgnFuncLabel']

    t3 = time.perf_counter()
    logger(f"Time for adding model components: {t3 - t2} s", "INFO")

    # quit()

    # Create mask and select data
    t4 = time.perf_counter()
    logger(f"Time to query data in pt bin {pt_label}: {t4:.3f} s", "INFO")
    fitter.set_mass_data_to_fit_hist(hist_mass)
    fitter.set_vn_vs_mass_data_to_fit_hist(hist_vn_vs_mass)
    t5 = time.perf_counter()
    logger(f"Time to set data to fit in pt bin {pt_label}: {t5 - t4:.3f} s", "INFO")

    # Add correlated background if specified
    if cfg_flow.get('corr_bkgs'):
        sel_string_cutset = (
            f"fMlScore0 < {cfg_cutset['ScoreBkg']['max'][i_pt]} && "
            f"fMlScore0 >= {cfg_cutset['ScoreBkg']['min'][i_pt]} && "
            f"fMlScore1 < {cfg_cutset['ScoreFD']['max'][i_pt]} && "
            f"fMlScore1 >= {cfg_cutset['ScoreFD']['min'][i_pt]} && "
            f"fM >= {cfg_flow['v2extraction']['MassFitRanges'][i_pt][0]} && "
            f"fM <= {cfg_flow['v2extraction']['MassFitRanges'][i_pt][1]}"
        )
        fitter.add_corr_bkgs(cfg_flow['corr_bkgs'], sel_string_cutset, pt_min, pt_max)

    fitter.setup()
    if fit_cfg.get('InitPars'):
        fitter.set_fit_pars(fit_cfg['InitPars'], pt_min, pt_max)
    t6 = time.perf_counter()
    logger(f"Time for setting up fitter in pt bin {pt_label}: {t6 - t5} s", "INFO")
    # Prefit the MC prompt enhanced cut to fix the tails, binned fit
    if fit_cfg.get('FixSgnFromMC'):
        fitter.set_fix_sgn_to_mc_prefit(True)
        fitter.prefit_mc(f"{cfg_flow['outdir']}/corrbkgs/templs_{pt_label}.root")
        fitter.plot_mc_prefit(False, True, loc=["lower left", "upper left"], path=outdir)
        fitter.plot_raw_residuals_mc_prefit(path=f"{outdir}/fM_mc_prefit_residuals_{pt_label}.pdf")
    t7 = time.perf_counter()
    logger(f"Time for prefit MC in pt bin {pt_label}: {t7 - t6} s", "INFO")

    t8 = time.perf_counter()
    logger(f"Time for fixing signal parameters to sp-integrated fit in pt bin {pt_label}: {t8 - t7} s", "INFO")

    try:
        status_mass, converged_mass = fitter.fit()
        fig_mass_fit_path = f"{outdir}/fit_mass{pt_label}.pdf"
        fitter.plot_mass_fit(False, True, loc=["lower left", "upper left"],  # (log, show_extra_info)
                            path=fig_mass_fit_path)
        print(f"Mass fit saved to {fig_mass_fit_path}")
    except Exception as e:
        logger(f"Fit in pt bin {pt_label} failed with exception: {e}", "ERROR")
        raise e

    try:
        status_vn, converged_vn = fitter.perform_vn_vs_mass_fit_roofit()
        fig_vn_fit_path = f"{outdir}/fit_vn_vs_mass_{pt_label}.pdf"
        fitter.plot_vn_vs_mass_fit(False, True, loc=["lower left", "upper left"],  # (log, show_extra_info)
                                path=fig_vn_fit_path)
        print(f"Vn fit saved to {fig_vn_fit_path}")
    except Exception as e:
        logger(f"Fit in pt bin {pt_label} failed with exception: {e}", "ERROR")
        raise e

    fit_info, sgn_pars, sgn_pars_uncs, bkg_pars, bkg_pars_uncs = fitter.get_fit_info()

    t9 = time.perf_counter()
    logger(f"Time for performing sp-integrated fit in pt bin {pt_label}: {t9 - t8} s", "INFO")
    t10 = time.perf_counter()
    logger(f"Time for sp scan in pt bin {pt_label}: {t10 - t9} s", "INFO")
    stats['RawYieldsSimFit'] = fit_info[sgn_func_label]["ry"]
    stats['RawYieldsSimFitUnc'] = fit_info[sgn_func_label]["ry_unc"]
    stats['MeanSimFit'] = sgn_pars[f"mu_{sgn_func_label}"]
    stats['MeanSimFitUnc'] = sgn_pars_uncs[f"mu_{sgn_func_label}"]
    stats['SigmaSimFit'] = sgn_pars[f"sigma_{sgn_func_label}"]
    stats['SigmaSimFitUnc'] = sgn_pars_uncs[f"sigma_{sgn_func_label}"]
    # Clear data_cutset to save memory
    t11 = time.perf_counter()
    logger(f"Time after sp scan in pt bin {pt_label}: {t11:.3f} s\n", "INFO")

    del fitter

    logger(f"Finished fits in total time: {time.perf_counter() - t0} s\n\n", "INFO")
    return pt_label, i_pt, stats

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Arguments')
    parser.add_argument('config_flow', metavar='text', default='config_Ds_Fit.yml')
    parser.add_argument('config_cutset', metavar='text', default='cutset_XX.yml')
    parser.add_argument('proj_file_path', metavar='text', default='')
    parser.add_argument('--batch', '-b', help='suppress video output', action='store_true')
    parser.add_argument('--multitrial', help='suppress redundant prints', action='store_true')
    args = parser.parse_args()

    print(f"Running get_vn_by_sequential_fit with config {args.config_flow}, cutset config {args.config_cutset}, multitrial={args.multitrial}")
    ROOT.gROOT.SetBatch(True)
    with open(args.config_flow, 'r') as CfgFlow:
        cfg_flow = yaml.safe_load(CfgFlow)
    with open(args.config_cutset, 'r') as CfgCutset:
        cfg_cutset = yaml.safe_load(CfgCutset)
    # print(f"Loaded flow config: {cfg_flow} and cutset config: {cfg_cutset}")

    outdir = f"{cfg_flow['outdir']}/cutvar_{cfg_flow['suffix']}_combined/raw_yields/" \
             f"fits_{os.path.basename(args.proj_file_path).replace('.root', '').split('_')[-1]}"
    os.makedirs(outdir, exist_ok=True)

    pt_bins = cfg_flow['ptbins']
    tasks = []
    vals_stats = {}
    for i_pt, (pt_min, pt_max) in enumerate(zip(pt_bins[:-1], pt_bins[1:])):
        tasks.append((cfg_flow, cfg_cutset, args.proj_file_path, i_pt, pt_min, pt_max, outdir, args.multitrial))
    with ProcessPoolExecutor(max_workers=1) as executor:
    # with ProcessPoolExecutor(max_workers=cfg_flow['v2extraction'].get('PtWorkers', 1)) as executor:
        results = [executor.submit(fit_pt_bin, *task) for task in tasks]
        for task in as_completed(results):
            pt_label, i_pt, stats = task.result()
            vals_stats[pt_label] = (i_pt, stats)

    # Check for exceptions in workers
    for task in results:
        if task.exception() is not None:
            logger(f"Worker generated an exception: {task.exception()}", "ERROR")
            raise task.exception()
    print("All workers completed")
    quit()

    # Sort vals_stats by i_pt
    vals_stats = dict(sorted(vals_stats.items(), key=lambda item: item[1][0]))

    os.makedirs(os.path.dirname(cutset).replace('cutset', 'raw_yield'), exist_ok=True)
    summary = {
        'hRawYieldsSimFit': TH1F("hRawYieldsSimFit", "hRawYieldsSimFit", len(pt_bins) - 1, np.array(pt_bins)),
        'hSummedSpYields': TH1F("hSummedSpYields", "hSummedSpYields", len(pt_bins) - 1, np.array(pt_bins)),
        'hMeanSimFit': TH1F("hMeanSimFit", "hMeanSimFit", len(pt_bins) - 1, np.array(pt_bins)),
        'hSigmaSimFit': TH1F("hSigmaSimFit", "hSigmaSimFit", len(pt_bins) - 1, np.array(pt_bins)),
        'hVnSimFit': TH1F("hVnSimFit", "hVnSimFit", len(pt_bins) - 1, np.array(pt_bins)),
        'hVnSimFitUnc': TH1F("hVnSimFitUnc", "hVnSimFitUnc", len(pt_bins) - 1, np.array(pt_bins)),
        'hWeightedSum': TH1F("hWeightedSum", "hWeightedSum", len(pt_bins) - 1, np.array(pt_bins)),
        'gVnSimFit': TGraphAsymmErrors(1),
        'gVnUnc': TGraphAsymmErrors(1),
    }
    summary['gVnSimFit'].SetName("gVnSimFit")
    summary['gVnUnc'].SetName("gVnUnc")

    for pt_label, (i_pt, pt_stats) in vals_stats.items():
        pt_min = pt_bins[i_pt]
        pt_max = pt_bins[i_pt + 1]

        # Hardcode all results
        summary['gVnSimFit'].SetPoint(i_pt, (pt_min+pt_max)/2, vals)
        summary['gVnSimFit'].SetPointError(i_pt, (pt_max-pt_min)/2, (pt_max-pt_min)/2, cutset_vals[f"{var}Unc"], cutset_vals[f"{var}Unc"])
        summary['gVnUnc'].SetPoint(i_pt, (pt_min+pt_max)/2, cutset_vals[f"{var}Unc"])
        summary['gVnUnc'].SetPointError(i_pt, (pt_max-pt_min)/2, (pt_max-pt_min)/2, 1.e-20, 1.e-20)

    logger("\n\n")
    logger("Saving results to root files ... ", "INFO")
    out_file_name = cutset.replace('cutsets', 'raw_yields').replace('cutset', 'raw_yields').replace('.yml', '.root')
    outfile = TFile.Open(out_file_name, 'RECREATE')
    for pt_label in vals_stats.keys():
        make_dir_root_file(pt_label, outfile, verbose=False)
    for hist_name, hist in summary[main_cfg][i_cutset].items():
        SetObjectStyle(hist, color=kBlack, markerstyle=kFullCircle)
        if "/" in hist_name:
            pt_dir, hist_name = hist_name.split("/")
            if "SpRyBkgMass" in hist_name:
                make_dir_root_file(f"{pt_dir}/MassBinsBkg", outfile, verbose=False)
                outfile.cd(f"{pt_dir}/MassBinsBkg")
            else:
                outfile.cd(pt_dir)
            hist.SetName(hist_name)
            hist.Write(hist_name)
        else:
            outfile.cd()
            hist.Write(hist_name)
    outfile.Close()

    logger(f"Processed config file {cutset} and saved results to {out_file_name}", "INFO")
