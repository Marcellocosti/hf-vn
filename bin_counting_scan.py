import argparse
import sys
import os
sys.path.append(f"{os.path.dirname(os.path.abspath(__file__))}/utils")
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # pylint: disable=wrong-import-position
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor
import itertools
from flarefly.data_handler import DataHandler
from flarefly.fitter import F2MassFitter
import yaml
import array
import ROOT
from ROOT import TFile
from matplotlib import gridspec
from sparse_dicts import get_pt_preprocessed_sparses
ROOT.gROOT.SetBatch(True)

def fit_integrated_spectrum(infile, config, outfile, outdir):
    """
    Fit the integrated spectrum for the given pT range and cuts.
    """

    means, sigmas, rys = [], [], []
    for iBin, (pt_min, pt_max, mass_min, mass_max) in enumerate(zip(config['Pt']['min'], config['Pt']['max'], 
                                                                    config['fitrangemin'], config['fitrangemax'])):
        print(f"Processing pT bin {iBin}: {pt_min}-{pt_max}, Mass range: {mass_min}-{mass_max}")
        pt_dir = f"pt_{int(pt_min*10)}_{int(pt_max*10)}"
        outfile.mkdir(pt_dir)
        outfile.cd(pt_dir)
        print(f"Processing integrated histogram {pt_dir}/hMassData of file {infile}")
        data_handler = DataHandler(infile, limits=[mass_min, mass_max], histoname=f"{pt_dir}/hMassData")
        sgn_func = ["gaussian"]
        bkg_func = ["chebpol2"]

        fitter_name = f"fd_{pt_min:.2f}_{pt_max:.2f}_diff_fit"
        fitter = F2MassFitter(data_handler, sgn_func, bkg_func, verbosity=5, name=fitter_name)
        fitter.set_background_initpar(0, "c0", 200.0)
        fitter.set_background_initpar(0, "c1", 10.0)
        fitter.mass_zfit()
                
        # Save the fit results
        # fitter.save_fit_results(outfile)
        means.append(fitter.get_mass()[0])
        sigmas.append(fitter.get_sigma()[0])
        rys.append(fitter.get_raw_yield()[0])
    
        loc = ["lower left", "upper left"]
        ax_title = r"$M(K\mathrm{\pi\pi})$ GeV$/c^2$"
        
        fig, _ = fitter.plot_mass_fit(
            style="ATLAS",
            show_extra_info = True,
            figsize=(8, 8), extra_info_loc=loc,
            axis_title=ax_title,
        )

        fig.savefig(
            os.path.join(
                outdir,
                f'pt_{pt_min:.2f}_{pt_max:.2f}_integrated_fit.png'
            ),
            dpi=300, bbox_inches="tight"
        )

    return means, sigmas, rys

def make_sideband_polynomial(exclude_min, exclude_max, integrate=False):
    def func(x, p):
        if not integrate:
            if exclude_min < x[0] < exclude_max:
                ROOT.TF1.RejectPoint()
                # return 0
        return p[0] + p[1]*x[0] + p[2]*x[0]*x[0]
    return func

def get_bin_counting(histo, func, mass_min, mass_max):
    """
    Perform a bin counting fit for the given histogram and function.
    """
    bin_counting = 0
    first_bin = histo.FindBin(mass_min)
    last_bin = histo.FindBin(mass_max)
    for iBin in range(first_bin, last_bin + 1):
        bin_content = histo.GetBinContent(iBin)
        if bin_content > 0:
            bin_counting += bin_content - func.Eval(histo.GetBinCenter(iBin))
    return bin_counting

def bin_counting_scan(infile, config, config_cutset, outfile, outdir, int_means, int_sigmas, int_rys):
    """
    Perform a bin counting scan for the given pT range and cuts.
    """
    
    proj_file = TFile.Open(infile, 'UPDATE')
    for iPt, (pt_min, pt_max, mass_min, mass_max, mean, sigma, ry) in enumerate(zip(config_cutset['Pt']['min'], config_cutset['Pt']['max'], 
                                                                                 config_cutset['fitrangemin'], config_cutset['fitrangemax'],
                                                                                 int_means, int_sigmas, int_rys)):
        print(f"\n\nProcessing pT bin {iPt}: {pt_min}-{pt_max}, Mass range: {mass_min}-{mass_max}")
        pt_dir = f"pt_{int(pt_min*10)}_{int(pt_max*10)}"
        outfile.mkdir(pt_dir)
        outfile.cd(pt_dir)

        MassSpTH2 = proj_file.Get(f"{pt_dir}/hMassSp")

        ry_bin_counting, ry_bin_counting_unc, left_edges, right_edges, fit_canvases = [], [], [], [], []

        # Build the SP intervals to be fitted
        # Find first and last non empty bins of SP 
        hSp = MassSpTH2.ProjectionY("hSp", 1, MassSpTH2.GetNbinsX())
        for iBin in range(1, hSp.GetNbinsX() + 1):
            if abs(hSp.GetBinCenter(iBin)) > config["projections"]["sp_ranges"][iPt]:
                print(f"Skipping bin {iBin} with center {abs(hSp.GetBinCenter(iBin))} outside SP range {config['projections']['sp_ranges'][iPt]}")
                continue
            if hSp.GetBinContent(iBin) <= 0:
                print(f"Skipping bin {iBin} with content {hSp.GetBinContent(iBin)}")
                continue
            if hSp.GetBinContent(iBin) > 0 and hSp.Integral(0, iBin-1) <= 0:
                print(f"Found first non-empty bin: {iBin} with content {hSp.GetBinContent(iBin)}")
                sp_first_bin = iBin
            if hSp.GetBinContent(iBin) > 0 and hSp.Integral(iBin+1, hSp.GetNbinsX()) <= 0:
                print(f"Found last non-empty bin: {iBin} with content {hSp.GetBinContent(iBin)}")
                sp_last_bin = iBin

        # iterate over the SP bins to define the windows
        window_bin_step = config["projections"]["sp_windows_nbins"][iPt]
        spWindowEdge = sp_first_bin + window_bin_step
        window_left_edge = sp_first_bin
        window_right_edge = spWindowEdge
        
        
        # print(f"\n\nSP first bin: {sp_first_bin}, SP last bin: {sp_last_bin}, SP first bin edge: {sp_first_bin_edge}, SP last bin edge: {sp_last_bin_edge}")
        # print(f"Window bin step: {window_bin_step}, SP window edge: {spWindowEdge}")
        # print(f"Window left edge: {window_left_edge}, Window right edge: {window_right_edge}")
        while window_right_edge < sp_last_bin:
            
            # print("\n\n\nITERATION")
            
            # Retrieve the histogram
            # print(f"    window_left_edge: {window_left_edge}, window_right_edge: {window_right_edge}, sp_last_bin_edge: {sp_last_bin_edge}")
            hMassProj = MassSpTH2.ProjectionX(f"hMass_{spWindowEdge}", window_left_edge, window_right_edge)

            # Perform a sideband fit for the InvMass of the current SP window and estimate the signal yield
            fPoly = ROOT.TF1("fPoly", make_sideband_polynomial(mean - 4*sigma, mean + 4*sigma), mass_min, mass_max, 3)
            fPoly.SetParameters(200, 10, 1)  # Initial guesses
            hMassProj.Fit(fPoly, "RQ")
            window_yield = get_bin_counting(hMassProj, fPoly, mean - 3*sigma, mean + 3*sigma)
            if window_yield > (ry / 100) * 5: # ask for at minimum 5% of the total yield
                # print(f"    Window yield {window_yield} is greater than 5% of total yield {ry}, saving window.")
                ry_bin_counting.append(window_yield)
                ry_bin_counting_unc.append(np.sqrt(window_yield))  # Assuming Poisson statistics for the uncertainty
                left_edges.append(hSp.GetBinLowEdge(window_left_edge))
                right_edges.append(hSp.GetBinLowEdge(window_right_edge) + hSp.GetBinWidth(window_right_edge))
                # print(f"    Appending left edge: {hSp.GetBinLowEdge(window_left_edge)}, right edge: {hSp.GetBinLowEdge(window_right_edge) + hSp.GetBinWidth(window_right_edge)}")
                window_left_edge = window_right_edge + 1
                window_right_edge += window_bin_step
                
                canvas = ROOT.TCanvas(f"c_{left_edges[-1]:.2f}_{right_edges[-1]:.2f}", f"c_{left_edges[-1]:.2f}_{right_edges[-1]:.2f}", 800, 600)
                hMassProj.SetTitle(f"SP Window {spWindowEdge} (left: {left_edges[-1]:.2f}, right: {right_edges[-1]:.2f})")
                hMassProj.GetXaxis().SetTitle("Invariant Mass (GeV/c^2)")
                hMassProj.GetYaxis().SetTitle("Counts")
                hMassProj.SetLineColor(ROOT.kBlue)
                hMassProj.Draw()
                fPoly.SetLineColor(ROOT.kRed)
                fPoly.Draw("same")
                leftEdge = ROOT.TLine(mean - 3*sigma, 0, mean - 3*sigma, hMassProj.GetMaximum())
                leftEdge.SetLineColor(ROOT.kGreen)
                leftEdge.SetLineWidth(2)
                leftEdge.Draw("same")
                rightEdge = ROOT.TLine(mean + 3*sigma, 0, mean + 3*sigma, hMassProj.GetMaximum())
                rightEdge.SetLineColor(ROOT.kGreen)
                rightEdge.SetLineWidth(2)
                rightEdge.Draw("same")
                os.makedirs(f"{outdir}/{pt_dir}", exist_ok=True)
                canvas.SaveAs(f"{outdir}/{pt_dir}/fit_window_{left_edges[-1]:.2f}_{right_edges[-1]:.2f}.png")
                fit_canvases.append(canvas)
            else:
                window_right_edge += window_bin_step

        print(f"\nleft_edges: {left_edges}")
        print(f"right_edges: {right_edges}\n")
        edges_array = array.array('d', left_edges + [right_edges[-1]])
        sgn_yield_bincounting = ROOT.TH1F(f"sgn_yield_bincounting", f"sgn_yield_bincounting",
                                          len(edges_array)-1, edges_array)
        for iBin in range(sgn_yield_bincounting.GetNbinsX()):
            sgn_yield_bincounting.SetBinContent(iBin + 1, ry_bin_counting[iBin])
            sgn_yield_bincounting.SetBinError(iBin + 1, ry_bin_counting_unc[iBin])
        sgn_yield_bincounting.Write()
        rel_sgn_yield_bincounting = ROOT.TH1F(f"rel_sgn_yield_bincounting", f"rel_sgn_yield_bincounting",
                                          len(edges_array)-1, edges_array)
        for iBin in range(rel_sgn_yield_bincounting.GetNbinsX()):
            rel_sgn_yield_bincounting.SetBinContent(iBin + 1, ry_bin_counting[iBin] / ry)
        rel_sgn_yield_bincounting.Write()
        # for canvas in fit_canvases:
        #     canvas.Write()
        # quit()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Arguments')
    parser.add_argument('input_config', metavar='text', default='config_Ds_Fit.yml')
    args = parser.parse_args()

    with open(args.input_config, 'r') as CfgFlow:
        cfg_flow = yaml.safe_load(CfgFlow)

    print(f"{cfg_flow["outdir"]}/cutvar_{cfg_flow['suffix']}_combined/")
    target_dir = f"{cfg_flow['outdir']}/cutvar_{cfg_flow['suffix']}_combined/"
    config_files = [f"{target_dir}/cutsets/{f}" for f in os.listdir(f"{target_dir}/cutsets") if f.endswith('.yml') and os.path.isfile(os.path.join(f"{target_dir}/cutsets", f))]
    proj_files   = [f"{target_dir}/proj/{f}" for f in os.listdir(f"{target_dir}/proj") if f.endswith('.root') and os.path.isfile(os.path.join(f"{target_dir}/proj", f))]
    
    print(f"Found {len(config_files)} config files and {len(proj_files)} project files.")
    
    for proj, config in zip(proj_files, config_files):
        with open(config, 'r') as CfgCutsets:
            cfg_cutset = yaml.safe_load(CfgCutsets)

        print(f"os.path.basename(proj): {os.path.basename(proj)}")
        # basename = os.path.splitext(os.path.basename(proj))[0]  # strip .root extension
        outdir = f"{os.path.splitext(proj)[0]}_bincount/"
        os.makedirs(os.path.dirname(outdir), exist_ok=True)
        outfile = TFile.Open(f"{outdir}/summary.root", 'RECREATE')
        # outfile = TFile.Open(f"{os.path.basename(proj)}_bincounting.root", 'RECREATE')
        
        # print("\n\n")
        # print(f"cfg_cutset: {cfg_cutset}")
        # print("\n\n")
        int_means, int_sigmas, int_rys = fit_integrated_spectrum(proj, cfg_cutset, outfile, outdir)
        # int_means = [1.861] * len(cfg_cutset['Pt']['min'])
        # int_sigmas = [0.015] * len(cfg_cutset['Pt']['min'])
        # int_rys = [10000] * len(cfg_cutset['Pt']['min'])
        bin_counting_scan(proj, cfg_flow, cfg_cutset, outfile, outdir, int_means, int_sigmas, int_rys)
        outfile.Close()
        # quit()

