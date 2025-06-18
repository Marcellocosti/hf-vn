import argparse
import sys
import os
sys.path.append(f"{os.path.dirname(os.path.abspath(__file__))}/utils")
# os.environ["CUDA_VISIBLE_DEVICES"] = ""  # pylint: disable=wrong-import-position
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor
from flarefly.data_handler import DataHandler
from flarefly.fitter import F2MassFitter
import yaml
import array
import ROOT
from ROOT import TFile
from matplotlib import gridspec
sys.path.append(f"{os.path.dirname(os.path.abspath(__file__))}/../utils")
from sparse_dicts import get_pt_preprocessed_sparses
ROOT.gROOT.SetBatch(True)
ROOT.gErrorIgnoreLevel = ROOT.kWarning  # Suppress Info messages and below

def fit_sp_bins(histofile, outdir, pt_min, pt_max, sp_min, sp_max, mass_min, mass_max, mean, sigma):
    """
    Fit the integrated spectrum for the given pT range and cuts.
    """

    print(f"Processing pT bin {pt_min}-{pt_max}, SP bin {sp_min}-{sp_max}")
    data_handler = DataHandler(histofile, limits=[mass_min, mass_max], histoname=f"pt_{int(pt_min * 10)}_{int(pt_max * 10)}/sp_histos/hMass__sp_{sp_min:.2f}_{sp_max:.2f}")

    sgn_func = ["gaussian"]
    bkg_func = ["chebpol2"]

    fitter_name = f"spfit_{sp_min:.2f}_{sp_max:.2f}"
    fitter = F2MassFitter(data_handler, sgn_func, bkg_func, verbosity=5, name=fitter_name)
    fitter.set_background_initpar(0, "c0", 200.0)
    fitter.set_background_initpar(0, "c1", 10.0)
    fitter.set_signal_initpar(0, "mu", mean) #, fix = True)
    fitter.set_signal_initpar(0, "sigma", sigma, fix = True)
    fitter.mass_zfit()

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
            f"{outdir}/pt_{int(pt_min*10)}_{int(pt_max*10)}",
            f'sp_{sp_min:.2f}_{sp_max:.2f}_fit.png'
        ),
        dpi=300, bbox_inches="tight"
    )

    return fitter.get_raw_yield()[0], fitter.get_raw_yield()[1]

def fit_integrated_spectrum(infile, config, outfile, outdir, histopath=""):
    """
    Fit the integrated spectrum for the given pT range and cuts.
    """

    means, means_uncs, sigmas, sigmas_uncs, rys, rys_uncs = [], [], [], [], [], []
    for iBin, (pt_min, pt_max, mass_min, mass_max) in enumerate(zip(config['Pt']['min'], config['Pt']['max'], 
                                                                    config['fitrangemin'], config['fitrangemax'])):
        print(f"Processing pT bin {iBin}: {pt_min}-{pt_max}, Mass range: {mass_min}-{mass_max}")
        pt_dir = f"pt_{int(pt_min*10)}_{int(pt_max*10)}"
        outfile.mkdir(pt_dir)
        outfile.cd(pt_dir)
        print(f"Processing integrated histogram {pt_dir}/hMassData of file {infile}")
        if histopath == "":
            data_handler = DataHandler(infile, limits=[mass_min, mass_max], histoname=f"{pt_dir}/hMassData")
        else:
            data_handler = DataHandler(infile, limits=[mass_min, mass_max], histoname=histopath)
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
        means_uncs.append(fitter.get_mass()[1])
        sigmas.append(fitter.get_sigma()[0])
        sigmas_uncs.append(fitter.get_sigma()[1])
        rys.append(fitter.get_raw_yield()[0])
        rys_uncs.append(fitter.get_raw_yield()[1])
    
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

    return means, means_uncs, sigmas, sigmas_uncs, rys, rys_uncs


def estimate_v2(cfg_flow, cfg_cutset, outdir, histofile, sp_bins, mean, sigma, reso=0.746):

    pt_bins = cfg_cutset['Pt']['min'] + [cfg_cutset['Pt']['max'][-1]]
    v2_vs_pt = ROOT.TH1F("v2_vs_pt", "v2_vs_pt", len(pt_bins) - 1, np.array(pt_bins))
    v2_vs_pt.SetName("v2_vs_pt")
    v2_vs_pt.SetTitle("v2 vs pT")
    v2_vs_pt.GetXaxis().SetTitle("pT (GeV/c)")
    v2_vs_pt.GetYaxis().SetTitle("v2")

    for iPt, (pt_min, pt_max, mass_min, mass_max, sp_pt_bins) in enumerate(zip(cfg_cutset['Pt']['min'], cfg_cutset['Pt']['max'], 
                                                                               cfg_cutset['fitrangemin'], cfg_cutset['fitrangemax'], sp_bins)):
        sp_centers = []
        raw_yields = []
        raw_yields_uncs = []
        sp_ry = ROOT.TH1F("sp_ry", "sp_ry", len(sp_pt_bins) - 1, np.array(sp_pt_bins))
        sp_ry_unc = ROOT.TH1F("sp_ry_unc", "sp_ry_unc", len(sp_pt_bins) - 1, np.array(sp_pt_bins))
        weight_av_unc = ROOT.TH1F("weight_av_unc", "weight_av_unc", len(sp_pt_bins) - 1, np.array(sp_pt_bins))
        weight_av_unc_int = ROOT.TH1F("weight_av_unc_int", "weight_av_unc_int", len(sp_pt_bins) - 1, np.array(sp_pt_bins))
        for isp, (sp_min, sp_max) in enumerate(zip(sp_pt_bins[:-1], sp_pt_bins[1:])):

            sp_centers.append((sp_min + sp_max) / 2)
            try:
                raw_yield, raw_yield_unc = fit_sp_bins(histofile, outdir, pt_min, pt_max, sp_min, sp_max, mass_min, mass_max, mean[iPt], sigma[iPt])
                raw_yields.append(raw_yield)
                sp_ry.SetBinContent(isp + 1, raw_yield)
                raw_yields_uncs.append(raw_yield_unc)
                sp_ry_unc.SetBinContent(isp + 1, raw_yield_unc)
                print(f"Raw yield for sp {sp_min:.2f} - {sp_max:.2f}: {raw_yield:.3f} ± {raw_yield_unc:.3f}")
            except Exception as e:
                print(f"Error fitting sp {sp_min:.2f} - {sp_max:.2f}: {e}")
                raw_yields.append(0)
                raw_yields_uncs.append(0)
                sp_ry.SetBinContent(isp + 1, 0)
                sp_ry_unc.SetBinContent(isp + 1, 0)

        raw_yields = np.array(raw_yields)
        raw_yields_uncs = np.array(raw_yields_uncs)
        sp_centers = np.array(sp_centers)

        # Compute weighted average
        weighted_avg = np.sum(raw_yields * (sp_centers / reso)) / np.sum(raw_yields)

        # Uncertainty on weighted average
        weighted_avg_unc = 0
        for iSp, (iV2, iRy, iRy_unc) in enumerate(zip(sp_centers, raw_yields, raw_yields_uncs)):
            weighted_avg_unc = weighted_avg_unc +  ( ( ((iV2/reso) / np.sum(raw_yields)) - ( (iRy * (iV2/reso)) / np.sum(raw_yields)**2 ) )**2)*iRy_unc**2
            weight_av_unc.SetBinContent(iSp+1, ( ( ((iV2/reso) / np.sum(raw_yields)) - ( (iRy * (iV2/reso)) / np.sum(raw_yields)**2 ) )**2)*iRy_unc**2)
            weight_av_unc_int.SetBinContent(iSp+1, np.sqrt(weighted_avg_unc))
        weighted_avg_unc = np.sqrt(weighted_avg_unc)

        v2_vs_pt.SetBinContent(iPt + 1, weighted_avg)
        v2_vs_pt.SetBinError(iPt + 1, weighted_avg_unc)

        outFile = TFile.Open(f"{outdir}/summary.root", 'UPDATE')
        outFile.cd(f"pt_{int(pt_min*10)}_{int(pt_max*10)}/sp_histos/")
        sp_ry.Write(f"sp_ry_{int(pt_min*10)}_{int(pt_max*10)}")
        sp_ry_unc.Write(f"sp_ry_unc_{int(pt_min*10)}_{int(pt_max*10)}")
        weight_av_unc.Write(f"weight_av_unc_{int(pt_min*10)}_{int(pt_max*10)}")
        weight_av_unc_int.Write(f"weight_av_unc_int_{int(pt_min*10)}_{int(pt_max*10)}")
        outFile.Close()

    outFile = TFile.Open(f"{outdir}/summary.root", 'UPDATE')
    outFile.cd()
    v2_vs_pt.Write()
    outFile.Close()
    print(f"Weighted average yield: {weighted_avg:.3f} ± {weighted_avg_unc:.3f}")

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

def adjust_edges(side, rebin, left_edges, right_edges, MassSpTH2, total_ry, mean, sigma, mass_min, mass_max, outdir, pt_dir):

    print(f"Adjusting left edges: {left_edges}")
    hSp = MassSpTH2.ProjectionY("hSp")

    if side == 'left':
        left_edge_first_window = left_edges[0]
        right_edge_first_window = right_edges[0]
    elif side == 'right':
        left_edge_first_window = left_edges[-1]
        right_edge_first_window = right_edges[-1]

    print(f"Left edge first window: {left_edge_first_window}, Right edge first window: {right_edge_first_window}")

    left_bin = hSp.FindBin(left_edge_first_window+0.001)
    right_bin = hSp.FindBin(right_edge_first_window-0.001)
    window_yield = 10000

    canva = ROOT.TCanvas("DummyCanva", "DummyCanva", 800, 600)
    # Use as criterion the relative variation of the yield

    delta_yield = 0.
    while ( (delta_yield / total_ry) < 0.05) and (left_bin < right_bin): # ask for at minimum 5% of the total yield

        hMassProj = MassSpTH2.ProjectionX("hMassFirstWindow", left_bin, right_bin)
        hMassProj.Rebin(rebin)
        fPoly = ROOT.TF1("fPoly", make_sideband_polynomial(mean - 4*sigma, mean + 4*sigma), mass_min, mass_max, 3)
        fPoly.SetParameters(200, 10, 1)
        hMassProj.Fit(fPoly, "RQ")
        window_yield = get_bin_counting(hMassProj, fPoly, mean - 3*sigma, mean + 3*sigma)
        delta_yield = abs(window_yield - total_ry)
        if ( (delta_yield / total_ry) < 0.05) and (left_bin < right_bin):
            final_histo = hMassProj.Clone(f"hMass_sp_{hSp.GetBinLowEdge(left_bin)}_{(hSp.GetBinLowEdge(right_bin) + hSp.GetBinWidth(right_bin))}")
        print(f"    Left bin edge: {left_bin}, Right bin edge: {right_bin}")
        print(f"    Current window yield: {window_yield}, Total yield: {total_ry}, Delta yield: {delta_yield}")
        if side == 'left':
            left_bin = left_bin + 1
        elif side == 'right':
            right_bin = right_bin - 1

        sp_window_string = f"{hSp.GetBinLowEdge(left_bin):.2f}_{(hSp.GetBinLowEdge(right_bin) + hSp.GetBinWidth(right_bin)):.2f}"
        canva = ROOT.TCanvas(sp_window_string, sp_window_string, 800, 600)
        hMassProj.SetTitle(sp_window_string)
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

    if side == 'left':
        updated_edge = hSp.GetBinLowEdge(left_bin-1)
        canva.SaveAs(f"{outdir}/{pt_dir}/fit_window_{updated_edge:.2f}_{right_edge_first_window:.2f}_updated_{side}_edge.png")
        final_histo.SetName(f"hMass__sp_{updated_edge:.2f}_{right_edge_first_window:.2f}")
    else:
        updated_edge = hSp.GetBinLowEdge(right_bin+1) + hSp.GetBinWidth(right_bin+1)
        canva.SaveAs(f"{outdir}/{pt_dir}/fit_window_{left_edge_first_window:.2f}_{updated_edge:.2f}_updated_{side}_edge.png")
        final_histo.SetName(f"hMass__sp_{left_edge_first_window:.2f}_{updated_edge:.2f}")
    print(f"    Adjusted {side} edge: {updated_edge}")

    return canva, updated_edge, final_histo

def bin_counting_scan(infile, config, config_cutset, outfile, outdir, int_means, int_sigmas, int_rys, rebin_factor):
    """
    Perform a bin counting scan for the given pT range and cuts.
    """
    
    proj_file = TFile.Open(infile, 'UPDATE')
    sp_edges = []
    for iPt, (pt_min, pt_max, mass_min, mass_max, mean, sigma, ry) in enumerate(zip(config_cutset['Pt']['min'], config_cutset['Pt']['max'],
                                                                                 config_cutset['fitrangemin'], config_cutset['fitrangemax'],
                                                                                 int_means, int_sigmas, int_rys)):
        print(f"\n\nProcessing pT bin {iPt}: {pt_min}-{pt_max}, Mass range: {mass_min}-{mass_max}")
        pt_dir = f"pt_{int(pt_min*10)}_{int(pt_max*10)}"
        outfile.mkdir(f"{pt_dir}/cSpHistos/")
        outfile.cd(pt_dir)

        MassSpTH2 = proj_file.Get(f"{pt_dir}/hMassSpData")
        ry_bin_counting, ry_bin_counting_unc, left_edges, right_edges, fit_canvases = [], [], [], [], []

        hSp = MassSpTH2.ProjectionY("hSp", 1, MassSpTH2.GetNbinsX())
        sp_first_bin = 1
        window_bin_step = config["projections"]["sp_windows_nbins"][iPt]
        spWindowEdge = sp_first_bin + window_bin_step
        window_left_edge = sp_first_bin
        window_right_edge = spWindowEdge

        fit_canvases = []
        histos = []
        while window_right_edge < hSp.GetNbinsX():

            print(f"Processing SP window: {window_left_edge} - {window_right_edge}")

            hMassProj = MassSpTH2.ProjectionX(f"hMass_{spWindowEdge}", window_left_edge, window_right_edge)
            hMassProj.Rebin(rebin_factor)
            hMassProj.SetTitle(f"SP Window {spWindowEdge} (left: {hSp.GetBinLowEdge(window_left_edge):.2f}, right: {hSp.GetBinLowEdge(window_right_edge) + hSp.GetBinWidth(window_right_edge):.2f})")
            hMassProj.SetName(f"hMass__sp_{hSp.GetBinLowEdge(window_left_edge):.2f}_{hSp.GetBinLowEdge(window_right_edge) + hSp.GetBinWidth(window_right_edge):.2f}")
            histos.append(hMassProj.Clone())
            fPoly = ROOT.TF1("fPoly", make_sideband_polynomial(mean - 4*sigma, mean + 4*sigma), mass_min, mass_max, 3)
            fPoly.SetParameters(200, 10, 1)
            hMassProj.Fit(fPoly, "RQ")
            window_yield = get_bin_counting(hMassProj, fPoly, mean - 3*sigma, mean + 3*sigma)
            print(f"    Window yield: {window_yield}, Total yield: {ry}")
            if window_yield > (ry * (5/100)): # ask for at minimum 5% of the total yield
                ry_bin_counting.append(window_yield)
                ry_bin_counting_unc.append(np.sqrt(window_yield))

                left_edges.append(hSp.GetBinLowEdge(window_left_edge))
                right_edges.append(hSp.GetBinLowEdge(window_right_edge) + hSp.GetBinWidth(window_right_edge))
                window_left_edge = window_right_edge + 1
                window_right_edge += window_bin_step

                canvas = ROOT.TCanvas(f"cSp_{left_edges[-1]:.2f}_{right_edges[-1]:.2f}", f"c_{left_edges[-1]:.2f}_{right_edges[-1]:.2f}", 800, 600)
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
                outfile.cd(f"{pt_dir}/cSpHistos/")
                canvas.Write()
            else:
                histos.pop()  # Remove the last histogram if it doesn't meet the yield criteria
                window_right_edge += window_bin_step

        canva_left_edge, left_bin_edge, histo_left_edge = adjust_edges('left', rebin_factor, left_edges, right_edges, MassSpTH2, ry_bin_counting[0], mean, sigma, mass_min, mass_max, outdir, pt_dir)
        left_edges[0] = left_bin_edge
        canva_right_edge, right_bin_edge, histo_right_edge = adjust_edges('right', rebin_factor, left_edges, right_edges, MassSpTH2, ry_bin_counting[-1], mean, sigma, mass_min, mass_max, outdir, pt_dir)
        right_edges[-1] = right_bin_edge
        sp_edges.append(left_edges + [right_edges[-1]])

        outfile.cd(pt_dir)
        canva_left_edge.Write()
        canva_right_edge.Write()

        outfile.mkdir(f"{pt_dir}/sp_histos/")
        outfile.cd(f"{pt_dir}/sp_histos/")
        histo_left_edge.Write()
        histo_right_edge.Write()
        for histo in histos:
            histo.Write()

        outfile.cd(pt_dir)
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

        ### Check sum of single yields with respect to the integrated yield
        total_single_yield = sum(ry_bin_counting)
        if total_single_yield > 0:
            print(f"Total single yield: {total_single_yield}, Integrated yield: {ry}")
            if abs(total_single_yield - ry)/ry > 0.03:  # 3% tolerance
                print(f"Warning: Sum of single yields differs by more than 3% wrt the integrated yield!")

    return sp_edges

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
    
    for proj, config, rebin_factors in zip(proj_files, config_files, cfg_flow['projections']['rebin_cutsets']):
        with open(config, 'r') as CfgCutsets:
            cfg_cutset = yaml.safe_load(CfgCutsets)

        outdir = f"{os.path.splitext(proj)[0]}_bincount/"
        os.makedirs(os.path.dirname(outdir), exist_ok=True)
        outfile = TFile.Open(f"{outdir}/summary.root", 'RECREATE')

        int_means, int_means_uncs, int_sigmas, int_sigmas_uncs, int_rys, int_rys_uncs = fit_integrated_spectrum(proj, cfg_cutset, outfile, outdir)
        # int_means = [1.861] * len(cfg_cutset['Pt']['min'])
        # int_sigmas = [0.015] * len(cfg_cutset['Pt']['min'])
        # int_rys = [10000] * len(cfg_cutset['Pt']['min'])
        
        sp_edges = bin_counting_scan(proj, cfg_flow, cfg_cutset, outfile, outdir, int_means, int_sigmas, int_rys, rebin_factors)
        outfile.Close()
        estimate_v2(cfg_flow, cfg_cutset, outdir, f"{outdir}/summary.root", sp_edges, int_means, int_sigmas, rebin_factors)
        print(f"Output written to {outdir}/summary.root")
