import argparse
import os
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from flarefly.data_handler import DataHandler
from flarefly.fitter import F2MassFitter
import yaml
import ROOT
os.environ["CUDA_VISIBLE_DEVICES"] = "" # pylint: disable=wrong-import-position
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
from ROOT import TFile
from matplotlib import gridspec
ROOT.gROOT.SetBatch(True)
import multiprocessing as mp
mp.set_start_method("spawn", force=True)
msg_service = ROOT.RooMsgService.instance()
msg_service.setGlobalKillBelow(ROOT.RooFit.FATAL)  # Only show FATAL errors (you can also use ROOT.RooFit.ERROR or INFO)

def fit_histo_flarefly(histo, bkgfuncs, sgnfuncs, massmin, massmax, outfilename, mean_int=None, sigma_int=None):
    data_handler = DataHandler(histo, limits=[massmin, massmax])
    fitter_name = f"sp__diff_fit"
    fitter = F2MassFitter(data_handler, sgnfuncs, bkgfuncs, verbosity=0, name=fitter_name)
    fitter.set_background_initpar(0, "c0", 200.0)
    fitter.set_background_initpar(0, "c1", 10.0)
    if mean_int is not None and sigma_int is not None:
        print(f"Using mean_int = {mean_int}")
        fitter.set_signal_initpar(0, "mu",  mean_int,  fix = True)
        fitter.set_signal_initpar(0, "sigma", sigma_int, fix = True)
    else:
        print("Using default mean = 1.86")
        fitter.set_signal_initpar(0, "mu", 1.86, fix = False)
        fitter.set_signal_initpar(0, "sigma", 0.01, fix = False)

    fitter.mass_zfit()

    loc = ["lower left", "upper left"]
    ax_title = r"$M(K\mathrm{\pi\pi})$ GeV$/c^2$"
    
    fig, _ = fitter.plot_mass_fit(
        style="ATLAS",
        show_extra_info = True,
        figsize=(8, 8), extra_info_loc=loc,
        axis_title=ax_title,
    )

    os.makedirs(os.path.dirname(outfilename), exist_ok=True)
    fig.savefig(
        outfilename,
        dpi=300, bbox_inches="tight"
    )

    return fitter.get_raw_yield(), fitter.get_mass(), fitter.get_sigma()

def fit_histo_roofit(histo, bkgfuncs, sgnfuncs, massmin, massmax, outfilename, mean_int=None, sigma_int=None):

    mass = ROOT.RooRealVar("mass", "Invariant Mass", massmin, massmax)
    mass.setRange("fitRange", massmin, massmax)
    vnvsmass = ROOT.RooRealVar("vnvsmass", "Vn Vs Mass", massmin, massmax)
    vnvsmass.setRange("fitRange", massmin, massmax)
    data_hist = ROOT.RooDataHist("data_hist", "Dataset from histogram", ROOT.RooArgList(mass), histo)

    # Define Double-Sided Crystal Ball (DSCB) components for the signal
    if mean_int is not None and sigma_int is not None:
        sgn_mean = ROOT.RooRealVar("sgn_mean", "Signal Mean", mean_int, 1.85, 1.88)
        sgn_width = ROOT.RooRealVar("sgn_width", "Signal Width", sigma_int, 0.01, 0.02)
        print(f"Using mean_int = {mean_int}, sigma_int = {sigma_int}")
        sgn_mean.setConstant(True)
        sgn_width.setConstant(True)
    else:
        sgn_mean = ROOT.RooRealVar("sgn_mean", "Signal Mean", 1.86, 1.85, 1.88)
        sgn_width = ROOT.RooRealVar("sgn_width", "Signal Width", 0.01, 0.005, 0.02)
        print("Using default mean = 1.86, sigma = 0.01")

    gaussian = ROOT.RooGaussian("gaussian", "gaussian", mass, sgn_mean, sgn_width)

    # Define Background Polynomial component (pol is implememted as 1 + ...)
    c1_bkg = ROOT.RooRealVar("c1_bkg", "c1_bkg", -1, -3, 3)
    c2_bkg = ROOT.RooRealVar("c2_bkg", "c2_bkg", 0, -1, 1)
    c3_bkg = ROOT.RooRealVar("c3_bkg", "c3_bkg", 0, -1, 1)
    background = ROOT.RooPolynomial("background", "Polynomial Background", mass, ROOT.RooArgList(c1_bkg, c2_bkg, c3_bkg))

    # Create the RooAddPdf with components and their fractions
    n_signal = ROOT.RooRealVar("n_signal", "Number of signal events", 5000, 0, 1e6)
    n_background = ROOT.RooRealVar("n_background", "Number of background events", 10000, 0, 1e6)

    # Construct the extended model using RooAddPdf
    total_pdf_mass = ROOT.RooAddPdf("total_pdf_mass", "Signal + Background",
                                    ROOT.RooArgList(gaussian, background),
                                    ROOT.RooArgList(n_signal, n_background))

    # Perform the extended maximum likelihood fit
    fit_result = total_pdf_mass.fitTo(data_hist, ROOT.RooFit.Save(),
                                      ROOT.RooFit.Extended(True),
                                      ROOT.RooFit.Range("fitRange"),
                                      ROOT.RooFit.PrintLevel(-1),       # Suppresses messages
                                      ROOT.RooFit.Verbose(False),       # Turns off verbose mode
                                      ROOT.RooFit.Warnings(False),      # Suppresses warnings
                                      ROOT.RooFit.Timer(False)          # Disables timing info
                                    )
    fit_result.printMultiline(ROOT.std.cout, 3, True)

    # Create frame for plotting
    frame = mass.frame()
    data_hist.plotOn(frame, ROOT.RooFit.Range("fitRange"))  # Plot within fit range
    total_pdf_mass.plotOn(frame, ROOT.RooFit.Range("fitRange"))  # Plot the full model within fit range
    total_pdf_mass.plotOn(frame, ROOT.RooFit.Components("signal"), ROOT.RooFit.LineColor(ROOT.kRed), ROOT.RooFit.Range("fitRange"))  # Gaussian
    total_pdf_mass.plotOn(frame, ROOT.RooFit.Components("background"), ROOT.RooFit.LineColor(ROOT.kGreen), ROOT.RooFit.LineStyle(ROOT.kDashed), ROOT.RooFit.Range("fitRange"))  # Background

    # Draw the plot
    canvas = ROOT.TCanvas("canvas", "Fit Result", 800, 600)
    frame.Draw()
    os.makedirs(os.path.dirname(outfilename), exist_ok=True)
    canvas.SaveAs(outfilename)

    return (n_signal.getVal(), n_signal.getError()), (sgn_mean.getVal(), sgn_mean.getError()), (sgn_width.getVal(), sgn_width.getError())

def process_sp_bin(fitter, proj_file_path, pt_label, mass_min, mass_max, rebin_factor, sp_left_bin, sp_right_bin, outdir, int_ry, mean=None, sigma=None):
    proj_file = TFile.Open(proj_file_path, "READ")
    hist_mass_sp_int = proj_file.Get(f"{pt_label}/hMassSpData")

    sp_min = hist_mass_sp_int.GetYaxis().GetBinLowEdge(sp_left_bin)
    sp_max = hist_mass_sp_int.GetYaxis().GetBinUpEdge(sp_right_bin)

    sp_label = f"sp_{sp_min:.2f}_{sp_max:.2f}"
    fig_sp_pt_path = f"{outdir}/{pt_label}/{sp_label}.png"
    print(f"Processing sp bin {sp_label}: sp_min: {sp_min}, sp_max: {sp_max}, left bin = {sp_left_bin}, right bin = {sp_right_bin}")
    histo_sp = hist_mass_sp_int.ProjectionX(f"hMass_{sp_label}", sp_left_bin, sp_right_bin)
    histo_sp.SetDirectory(0)
    histo_sp.Rebin(rebin_factor)

    sp_center = (sp_min + sp_max) / 2
    try:
        if fitter == 'roofit':
            if mean is not None and sigma is not None:
                (raw_yield, raw_yield_unc), (mean_sp, mean_sp_unc), (sigma_sp, sigma_sp_unc) = fit_histo_roofit(histo_sp, ['chebpol2'], ['gaussian'], mass_min, mass_max, fig_sp_pt_path, mean, sigma)
            else:
                (raw_yield, raw_yield_unc), (mean_sp, mean_sp_unc), (sigma_sp, sigma_sp_unc) = fit_histo_roofit(histo_sp, ['chebpol2'], ['gaussian'], mass_min, mass_max, fig_sp_pt_path)
            return histo_sp, sp_center, raw_yield, raw_yield_unc, mean_sp, mean_sp_unc, sigma_sp, sigma_sp_unc
        else:
            if mean is not None and sigma is not None:
                (raw_yield, raw_yield_unc), (mean_sp, mean_sp_unc), (sigma_sp, sigma_sp_unc) = fit_histo_flarefly(histo_sp, ['chebpol2'], ['gaussian'], mass_min, mass_max, fig_sp_pt_path, mean, sigma)
            else:
                (raw_yield, raw_yield_unc), (mean_sp, mean_sp_unc), (sigma_sp, sigma_sp_unc) = fit_histo_flarefly(histo_sp, ['chebpol2'], ['gaussian'], mass_min, mass_max, fig_sp_pt_path)

        # if (raw_yield/int_ry) >= (0.5/100):  # Keep only bins with at least 1% of the integrated yield
        #     return histo_sp, sp_center, raw_yield, raw_yield_unc, mean_sp, mean_sp_unc, sigma_sp, sigma_sp_unc
        # else:
        #     histo_sp.SetName(f"hMass_{sp_label}_not_used")
        #     return histo_sp, sp_center, 0, 0, 0, 0, 0, 0
    except Exception as e:
        print(f"Error fitting sp {sp_min:.2f} - {sp_max:.2f}: {e}")
        return histo_sp, sp_center, 0, 0, 0, 0, 0, 0

def process_pt_bin(iPt, config, histos_summary, mass_min, mass_max, rebin_factor, sp_window_nbins, pt_label, proj_file_path):
    ROOT.gROOT.SetBatch(True)
    proj_file = TFile.Open(proj_file_path, "READ")
    hist_mass_int = proj_file.Get(f"{pt_label}/hMassData")
    hist_mass_sp_int = proj_file.Get(f"{pt_label}/hMassSpData")

    fig_int_fit_path = f"{outdir}/fit_int_{pt_label}.png"
    if config['v2Extraction']["UseFlareFly"]:
        (raw_yield, raw_yield_unc), (mean_int, _), (sigma_int, _) = fit_histo_flarefly(hist_mass_int, ['chebpol2'], ['gaussian'], mass_min, mass_max, fig_int_fit_path)
    else:
        (raw_yield, raw_yield_unc), (mean_int, _), (sigma_int, _) = fit_histo_roofit(hist_mass_int, ['chebpol2'], ['gaussian'], mass_min, mass_max, fig_int_fit_path)
        
    histos_summary['integrated_yields'].SetBinContent(iPt + 1, raw_yield)

    # Project inv mass as a function of scalar product bins
    axis_sp = hist_mass_sp_int.GetYaxis()
    sp_bins = axis_sp.GetNbins()
    sp_left_bins = [i for i in range(1, sp_bins + 2, sp_window_nbins+1) if abs(axis_sp.GetBinLowEdge(i)) < 2.0]
    sp_right_bins = [i - 1 for i in sp_left_bins[1:]] + [sp_left_bins[-1] + sp_window_nbins]

    sp_left_edges = [axis_sp.GetBinLowEdge(i) for i in sp_left_bins]
    sp_right_edges = [axis_sp.GetBinLowEdge(i) + axis_sp.GetBinWidth(i) for i in sp_right_bins]
    sp_edges = sp_left_edges + [sp_right_edges[-1]]
    # sp_edges = [-2.16, -1.84, -1.52, -1.36, -1.20, -1.04, -0.88, -0.72, -0.56, -0.40, -0.32, -0.24, -0.16, -0.08, 0.0, 0.08, 0.16, 0.24, 0.32, 0.40, 0.56, 0.72, 0.88, 1.04, 1.20, 1.36, 1.52, 1.84, 2.16]

    histos = {
        'means': ROOT.TH1F("hist_means", "hist_means", len(sp_edges) - 1, np.array(sp_edges)),
        'sigmas': ROOT.TH1F("hist_sigmas", "hist_sigmas", len(sp_edges) - 1, np.array(sp_edges)),
        'sp_ry': ROOT.TH1F("hist_sp_ry", "hist_sp_ry", len(sp_edges) - 1, np.array(sp_edges)),
        'sp_ry_unc': ROOT.TH1F("hist_sp_ry_unc", "hist_sp_ry_unc", len(sp_edges) - 1, np.array(sp_edges)),
        'weight_av_unc': ROOT.TH1F("hist_weight_av_unc", "hist_weight_av_unc", len(sp_edges) - 1, np.array(sp_edges)),
        'weight_av_unc_int': ROOT.TH1F("hist_weight_av_unc_int", "hist_weight_av_unc_int", len(sp_edges) - 1, np.array(sp_edges)),
        'weight_av_unc_int_first_term': ROOT.TH1F("hist_weight_av_unc_int_first_term", "hist_weight_av_unc_int_first_term", len(sp_edges) - 1, np.array(sp_edges)),
        'weight_av_unc_int_second_term': ROOT.TH1F("hist_weight_av_unc_int_second_term", "hist_weight_av_unc_int_second_term", len(sp_edges) - 1, np.array(sp_edges)),
    }
    for hist in histos.values():
        hist.SetDirectory(0)

    histos_sp, raw_yields, raw_yields_uncs, sp_centers, sp_yield_estimates = [], [], [], [], []
    ### PARALLELIZED
    sp_yield_estimates = []
    with ProcessPoolExecutor(max_workers=1) as executor:
        for isp, (sp_left_bin, sp_right_bin) in enumerate(zip(sp_left_bins, sp_right_bins)):
            if config['v2Extraction']["UseRooFit"]:
                if config['v2Extraction'].get('FixMeanToInt') and config['v2Extraction'].get('FixSigmaToInt'):
                    sp_yield_estimates.append(executor.submit(
                        process_sp_bin,
                        'roofit', proj_file_path, pt_label, mass_min, mass_max, rebin_factor, sp_left_bin, sp_right_bin, outdir, raw_yield, mean_int, sigma_int
                    ))
                else:
                    sp_yield_estimates.append(executor.submit(
                        process_sp_bin,
                        'roofit', proj_file_path, pt_label, mass_min, mass_max, rebin_factor, sp_left_bin, sp_right_bin, outdir, raw_yield
                    ))
            else:
                if config['v2Extraction'].get('FixMeanToInt') and config['v2Extraction'].get('FixSigmaToInt'):
                    sp_yield_estimates.append(executor.submit(
                        process_sp_bin,
                        'flarefly', proj_file_path, pt_label, mass_min, mass_max, rebin_factor, sp_left_bin, sp_right_bin, outdir, raw_yield, mean_int, sigma_int
                    ))
                else:
                    sp_yield_estimates.append(executor.submit(
                        process_sp_bin,
                        'flarefly', proj_file_path, pt_label, mass_min, mass_max, rebin_factor, sp_left_bin, sp_right_bin, outdir, raw_yield
                    ))

    for isp, sp_yield_estimate in enumerate(sp_yield_estimates):
        histo_sp, sp_center, raw_yield, raw_yield_unc, mean_sp, mean_sp_unc, sigma_sp, sigma_sp_unc = sp_yield_estimate.result()
        histos_sp.append(histo_sp)
        raw_yields.append(raw_yield)
        raw_yields_uncs.append(raw_yield_unc)
        sp_centers.append(sp_center)
        histos['sp_ry'].SetBinContent(isp + 1, raw_yield)
        histos['sp_ry_unc'].SetBinContent(isp + 1, raw_yield_unc)
        histos['means'].SetBinContent(isp + 1, mean_sp)
        histos['means'].SetBinError(isp + 1, mean_sp_unc)
        histos['sigmas'].SetBinContent(isp + 1, sigma_sp)
        histos['sigmas'].SetBinError(isp + 1, sigma_sp_unc)

    raw_yields = np.array(raw_yields)
    raw_yields_uncs = np.array(raw_yields_uncs)
    sp_centers = np.array(sp_centers)

    # Compute weighted average
    weighted_avg = np.sum(raw_yields * (sp_centers / reso)) / np.sum(raw_yields)

    # Uncertainty on weighted average
    weighted_avg_unc = 0
    for iSp, (iV2, iRy_unc) in enumerate(zip(sp_centers, raw_yields_uncs)):
        weighted_avg_unc = weighted_avg_unc +  ( ( ((iV2/reso) / np.sum(raw_yields)) - ( (np.sum(raw_yields * (sp_centers / reso))) / np.sum(raw_yields)**2 ) )**2)*iRy_unc**2
        histos['weight_av_unc'].SetBinContent(iSp+1, ( ( ((iV2/reso) / np.sum(raw_yields)) - ( (np.sum(raw_yields * (sp_centers / reso))) / np.sum(raw_yields)**2 ) )**2)*iRy_unc**2)
        histos['weight_av_unc_int'].SetBinContent(iSp+1, np.sqrt(weighted_avg_unc))
        histos['weight_av_unc_int_first_term'].SetBinContent(iSp+1, np.sqrt( ((iV2/reso) / np.sum(raw_yields))**2*iRy_unc**2))
        histos['weight_av_unc_int_second_term'].SetBinContent(iSp+1, np.sqrt(( (np.sum(raw_yields * (sp_centers / reso))) / np.sum(raw_yields)**2 )**2*iRy_unc**2))
    weighted_avg_unc = np.sqrt(weighted_avg_unc)
    histos_summary["summed_sp_yields"].SetBinContent(iPt + 1, np.sum(raw_yields))
    # proj_file.Close()
    # weighted_avg = 0
    # weighted_avg_unc = 0
    # return
    # return weighted_avg, weighted_avg_unc
    return histos_sp, histos, weighted_avg, weighted_avg_unc

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Arguments')
    parser.add_argument('input_config', metavar='text', default='config_Ds_Fit.yml')
    args = parser.parse_args()

    with open(args.input_config, 'r') as CfgFlow:
        cfg_flow = yaml.safe_load(CfgFlow)

    target_dir = f"{cfg_flow['outdir']}/cutvar_{cfg_flow['suffix']}_combined/"
    config_files = [f"{target_dir}/cutsets/{f}" for f in os.listdir(f"{target_dir}/cutsets") if f.endswith('.yml') and os.path.isfile(os.path.join(f"{target_dir}/cutsets", f))]
    config_files = sorted(
        config_files,
        key=lambda f: int(f.split('_')[-1].split('.')[0])
    )
    proj_files   = [f"{target_dir}/proj/{f}" for f in os.listdir(f"{target_dir}/proj") if f.endswith('.root') and os.path.isfile(os.path.join(f"{target_dir}/proj", f))]
    proj_files = sorted(
        proj_files,
        key=lambda f: int(f.split('_')[-1].split('.')[0])
    )

    print(f"Found {len(config_files)} config files and {len(proj_files)} project files.")
    print(f"\nConfig files: {config_files}")
    print(f"\nProject files: {proj_files}")
    pt_bins = cfg_flow['ptbins']
    reso = 0.746
    for proj_file_path, config, rebin_factors, sp_windows_nbins in zip(proj_files, config_files, cfg_flow['projections']['rebin_cutsets'], cfg_flow['projections']['sp_windows_nbins']):
        with open(config, 'r') as CfgCutsets:
            cfg_cutset = yaml.safe_load(CfgCutsets)

        outdir = f"{os.path.splitext(proj_file_path)[0]}_bincount/"
        os.makedirs(os.path.dirname(outdir), exist_ok=True)

        # if proj_file_path != "/home/mdicosta/alice/hf-vn/test_output/cutvar_test_cms_combined//proj/proj_01.root":
        #     continue

        histos_summary_yields = {
            'integrated_yields': ROOT.TH1F("hist_integrated_yields", "hist_integrated_yields", len(pt_bins) - 1, np.array(pt_bins)),
            'summed_sp_yields': ROOT.TH1F("hist_summed_sp_yields", "hist_summed_sp_yields", len(pt_bins) - 1, np.array(pt_bins))
        }
        histos_sp, histos_summary, weighted_avgs, weighted_avgs_uncs = [], [], [], []
        for iPt, (pt_min, pt_max, mass_min, mass_max, rebin_factor, sp_window_nbins) in enumerate(zip(pt_bins[:-1], pt_bins[1:], cfg_cutset['fitrangemin'], cfg_cutset['fitrangemax'], rebin_factors, sp_windows_nbins)):
            print(f"\nProcessing pt bin {iPt+1}/{len(pt_bins)-1}: {pt_min} - {pt_max} GeV/c")
            # Process each pt bin
            # process_pt_bin(iPt, cfg_flow, histos_summary_yields, mass_min, mass_max, rebin_factor, sp_window_nbins-1, f"pt_{int(pt_min*10)}_{int(pt_max*10)}", proj_file_path)
            # weighted_avg, weighted_avg_unc = process_pt_bin(iPt, cfg_flow, histos_summary_yields, mass_min, mass_max, rebin_factor, sp_window_nbins-1, f"pt_{int(pt_min*10)}_{int(pt_max*10)}", proj_file_path)
            histo_sp, histo_summary, weighted_avg, weighted_avg_unc = process_pt_bin(iPt, cfg_flow, histos_summary_yields, mass_min, mass_max, rebin_factor, sp_window_nbins-1, f"pt_{int(pt_min*10)}_{int(pt_max*10)}", proj_file_path)

            histos_sp.append(histo_sp)
            histos_summary.append(histo_summary)
            weighted_avgs.append(weighted_avg)
            weighted_avgs_uncs.append(weighted_avg_unc)

        hist_v2_values = ROOT.TH1F("hist_v2_values", "hist_v2_values", len(pt_bins) - 1, np.array(pt_bins))
        outfile = TFile.Open(f"{outdir}/summary.root", 'RECREATE')
        for ipt, (pt_min, pt_max, histos_sp_pt, histos_summary_pt, weight_avg_pt, weight_avg_unc_pt) in enumerate(zip(pt_bins[:-1], pt_bins[1:], histos_sp, histos_summary, weighted_avgs, weighted_avgs_uncs)):
            pt_label = f"pt_{int(pt_min*10)}_{int(pt_max*10)}"
            outfile.mkdir(pt_label)
            outfile.cd(pt_label)

            outfile.mkdir(f"{pt_label}/sp_bins")
            outfile.cd(f"{pt_label}/sp_bins")
            for hist in histos_sp_pt:
                hist.Write(hist.GetName())
            outfile.cd(pt_label)
            for hist_name, hist in histos_summary_pt.items():
                hist.Write(hist_name)

            # Write the weighted average values
            hist_v2_values.SetBinContent(ipt + 1, weight_avg_pt)
            hist_v2_values.SetBinError(ipt + 1, weight_avg_unc_pt)

        outfile.cd()
        for hist_name, hist in histos_summary_yields.items():
            hist.Write(hist_name)
        hist_v2_values.Write("hist_v2_values")
        outfile.Close()
        print(f"\n\nProcessed {proj_file_path} and config file {config} and saved results to {outdir}/summary.root")





    # ### NOT PARALLELIZED
    # for isp, (sp_min, sp_max) in enumerate(zip(sp_edges[:-1], sp_edges[1:])):
    #     print(f"sp_min = {sp_min}, sp_max = {sp_max}")
    #     sp_center, raw_yield, raw_yield_unc, mean_sp, mean_sp_unc, sigma_sp, sigma_sp_unc = process_sp_bin(
    #         proj_file_path, pt_label, mass_min, mass_max, rebin_factor, sp_min, sp_max
    #     )

    #     for sp_yield_estimate in sp_yield_estimates:
    #         sp_center, raw_yield, raw_yield_unc, mean_sp, mean_sp_unc, sigma_sp, sigma_sp_unc = sp_yield_estimate.result()
    #         raw_yields.append(raw_yield)
    #         raw_yields_uncs.append(raw_yield_unc)
    #         sp_centers.append(sp_center)
    #         histos['sp_ry'].SetBinContent(isp + 1, raw_yield)
    #         histos['sp_ry_unc'].SetBinContent(isp + 1, raw_yield_unc)
    #         histos['means'].SetBinContent(isp + 1, mean_sp)
    #         histos['means'].SetBinError(isp + 1, mean_sp_unc)
    #         histos['sigmas'].SetBinContent(isp + 1, sigma_sp)
    #         histos['sigmas'].SetBinError(isp + 1, sigma_sp_unc)