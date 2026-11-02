"""
python script for the computation of the prompt or non-prompt v2 via extrapolation
run: python ComputeV2vsFDFrac.py config.yaml --inputdir path/to/input --outputdir path/to/output --suffix text

"""
import argparse
import os
import yaml
import sys
import numpy as np
import ROOT
from ROOT import TFile, TH1, TCanvas, TLegend, TLatex, TGraphErrors, TF1, TH1D, TVirtualFitter, gROOT, TLine
from ROOT import kBlack, kAzure, kOrange, kFullCircle
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(script_dir, '..', 'utils'))
from utils import get_particle_info, logger
from load_utils import load_root_files
from StyleFormatter import SetObjectStyle, GetROOTColor

TH1.AddDirectory(False)

def set_frame_style(canv, Title, particleTit):
    canv.SetLeftMargin(0.15)
    canv.SetRightMargin(0.05)
    canv.SetBottomMargin(0.15)
    canv.SetTopMargin(0.05)
    hFrame = canv.DrawFrame(0.0, -0.5, 1, 0.5, f";Non-prompt {particleTit} fraction; #it{{v}}_{{2}}^{{#it{{obs}}}}")
    hFrame.GetYaxis().SetDecimals()
    hFrame.GetYaxis().SetNoExponent()
    hFrame.GetXaxis().SetMoreLogLabels()
    hFrame.GetYaxis().SetTitleSize(0.04)
    hFrame.GetYaxis().SetTitleOffset(1.2)
    hFrame.GetYaxis().SetLabelSize(0.04)
    hFrame.GetXaxis().SetTitleSize(0.04)
    hFrame.GetXaxis().SetLabelSize(0.04)
    hFrame.GetXaxis().SetTitleOffset(1.2)
    hFrame.GetYaxis().SetNdivisions(505)

def set_frame_margin(canv):
    canv.SetLeftMargin(0.15)
    canv.SetRightMargin(0.05)
    canv.SetBottomMargin(0.15)
    canv.SetTopMargin(0.05)

def v2_vs_frac(config, ptMins, ptMaxs, CutSets, rawYieldFiles, fracFiles, multitrial, outputDir):

    gROOT.SetBatch(True)

    nPtBins = len(ptMins)
    particleTit, _, decay, _, _, _ = get_particle_info(config['Dmeson'])

    hV2, gV2, hFracFD = [], [], []

    lineat0 = TLine(0, 0, 1, 0)
    lineat0.SetLineStyle(2)
    lineat0.SetLineColor(kBlack)

    for fracFile, v2File in zip(fracFiles, rawYieldFiles):
        inV2File = TFile.Open(v2File)
        hV2.append(inV2File.Get('hVnSimFit'))
        gV2.append(inV2File.Get('gVnSimFit'))

        inFracFile = TFile.Open(fracFile)
        hFracFD.append(inFracFile.Get('hFDFrac'))

    gFracVsV2, hV2VsFrac = [], [] # gFracVsV2 used for fitting, hV2VsFrac used for plotting
    hV2VsPtFD = hV2[0].Clone("hV2VsPtFD")
    hV2VsPtFDUnc = hV2[0].Clone("hV2VsPtFDUnc")
    hV2VsPtPrompt = hV2[0].Clone("hV2VsPtPrompt")
    hV2VsPtPromptUnc = hV2[0].Clone("hV2VsPtPromptUnc")

    cFrac, ptStrings, chi2Strings, promptV2Strings = [], [], [], []

    for iPt, (ptMin, ptMax) in enumerate(zip(ptMins, ptMaxs)):
        ptCent = (ptMin + ptMax) / 2
        nSets = CutSets[iPt]

        ptBinFrac = iPt + 1
        # Multitrial is evaluated per pt-bin, so the
        # pt-bins of ry and frac files are not equal
        if multitrial and hV2[0].GetNbinsX() != hFracFD[0].GetNbinsX():
            logger(f"Number of pt bins in raw yield and fraction files do not match: {hV2[0].GetNbinsX()} != {hFracFD[0].GetNbinsX()}", level="WARNING")
            for iFracBin in range(hFracFD[0].GetNbinsX() + 1):
                binCenter = hFracFD[0].GetBinCenter(iFracBin + 1)
                if binCenter > ptMin and binCenter < ptMax:
                    ptBinFrac = iFracBin + 1
                    logger(f"----> Matching pt bin found at index {ptBinFrac} with center {binCenter}", level="INFO")
                    break

        gFracVsV2.append(TGraphErrors(-1))
        hV2VsFrac.append(TH1D(f"hV2VsFrac_{iPt}", "", 1000, 0.0, 1.0))
        SetObjectStyle(hV2VsFrac[-1], markerstyle=kFullCircle, markersize=0)
        SetObjectStyle(gFracVsV2[-1], linecolor=kBlack, linewidth=2, markerstyle=kFullCircle, markersize=1, markercolor=kBlack)

        if not multitrial:
            logger(f"Processing pt bin {iPt+1}/{nPtBins}: {ptMin:.2f} < pT < {ptMax:.2f} GeV/c, nSets: {nSets}", level="INFO")

        skip_cuts_iPt = []
        if config.get('v2VsFrac') is not None:
            skip_cuts = config['v2VsFrac'].get('skip_cuts', [])
            skip_cuts_iPt = skip_cuts[iPt] if iPt < len(skip_cuts) else []
            if skip_cuts_iPt != []:
                logger(f"Skipping cutsets at indices: {skip_cuts_iPt}", level="WARNING")

        v2Values = [hist.GetBinContent(iPt + 1) for i, hist in enumerate(hV2) if i not in skip_cuts_iPt]
        v2Unc = [hist.GetBinError(iPt + 1) for i, hist in enumerate(hV2) if i not in skip_cuts_iPt]
        fracFDValues = [hist.GetBinContent(ptBinFrac) for i, hist in enumerate(hFracFD) if i not in skip_cuts_iPt]
        fracFDUnc = [hist.GetBinError(ptBinFrac) for i, hist in enumerate(hFracFD) if i not in skip_cuts_iPt]

        for iSet, (v2, fracFD, v2Unc, fracFDUnc) in enumerate(zip(v2Values, fracFDValues, v2Unc, fracFDUnc)):
            if not multitrial:
                logger(f"pt: {ptCent:.4f}, v2: {v2:.4f}, fracFD: {fracFD:.4f}", level="INFO")
            gFracVsV2[iPt].SetPoint(iSet, fracFD, v2)
            gFracVsV2[iPt].SetPointError(iSet, fracFDUnc, v2Unc)
        
        linFunc = TF1("linear", "pol1", 0, 1)
        SetObjectStyle(linFunc, color=kOrange+1, linestyle=2, linewidth=2)
        fitOption = ""
        if multitrial:
            fitOption += "Q"
        gFracVsV2[-1].Fit("linear", fitOption, "", 0, 1)
        chi2 = linFunc.GetChisquare()
        ndf = linFunc.GetNDF()

        # get the confidence intervals 0.683
        fitter = TVirtualFitter.GetFitter()
        fitter.GetConfidenceIntervals(hV2VsFrac[-1], 0.683)
        hV2VsFrac[-1].SetLineColorAlpha(kAzure+5, 0.15)

        # get the v2 value at the FD fraction = 1
        hV2VsPtFD.SetBinContent(iPt + 1, hV2VsFrac[-1].GetBinContent(hV2VsFrac[-1].GetNbinsX()))
        hV2VsPtFD.SetBinError(iPt + 1, hV2VsFrac[-1].GetBinError(hV2VsFrac[-1].GetNbinsX()))
        hV2VsPtFDUnc.SetBinContent(iPt + 1, hV2VsFrac[-1].GetBinError(hV2VsFrac[-1].GetNbinsX()))
        
        # get the v2 value at the FD fraction = 0
        hV2VsPtPrompt.SetBinContent(iPt + 1, hV2VsFrac[-1].GetBinContent(1))
        hV2VsPtPrompt.SetBinError(iPt + 1, hV2VsFrac[-1].GetBinError(1))
        hV2VsPtPromptUnc.SetBinContent(iPt + 1, hV2VsFrac[-1].GetBinError(1))
        
        #TODO: plot the v2 vs pt, and the center of the pt bin is calculate by the average of pT

        ptStrings.append(f"{ptMin:.1f} < #it{{p}}_{{T}} < {ptMax:.1f} GeV/#it{{c}}")
        chi2Strings.append(f"#chi^{{2}}/n.d.f = {chi2:.2f}/{ndf:.2f}")
        promptV2Strings.append(f"Prompt #it{{v}}_{{2}} = {hV2VsFrac[-1].GetBinContent(1):.3f} #pm {hV2VsFrac[-1].GetBinError(1):.3f}")

    # save the results
    os.makedirs(outputDir, exist_ok=True)
    outFile = TFile(os.path.join(outputDir, 'v2VsFrac.root'), 'recreate')
    
    t = TLatex(8, 8, "")
    t.SetNDC()
    t.SetTextFont(42)
    t.SetTextColor(kBlack)

    # Create canvas to contain all plots
    canvV2VsFrac = TCanvas("cV2VsFrac", "v2 versus fraction", 1800, 900) 
    canvV2VsFrac.Divide(int(np.ceil((nPtBins + 1) / 2)), 2)

    for iPt, (ptMin, ptMax) in enumerate(zip(ptMins, ptMaxs)):
        if iPt == 0:
            suffix_pdf = '('
        elif iPt == nPtBins-1:
            suffix_pdf = ')'
        else:
            suffix_pdf = ''
        if nPtBins == 1:
            suffix_pdf = ''

        cFrac.append(TCanvas(f"cFrac_{ptMin}_{ptMax}", "", 1200, 1200))
        set_frame_style(cFrac[-1], ptStrings[iPt], particleTit)

        t.SetTextSize(0.04)
        t.DrawLatex(0.25, 0.85, decay)
        t.DrawLatex(0.25, 0.78, f"{ptStrings[iPt]}")
        t.SetTextSize(0.035)
        t.DrawLatex(0.250, 0.28, f'{promptV2Strings[iPt]}')
        t.DrawLatex(0.250, 0.23, f'{chi2Strings[iPt]}')

        hV2VsFrac[iPt].Draw("same pZ")
        gFracVsV2[iPt].Draw("same pZ")

        cFrac[-1].Update()

        cFrac[iPt].SaveAs(f"{outputDir}/v2VsFrac.pdf{suffix_pdf}")
        cFrac[iPt].SaveAs(f"{outputDir}/v2VsFrac_pt{ptMin}_{ptMax}.png")

        canvV2VsFrac.cd(iPt + 1)
        hV2VsFrac[iPt].GetXaxis().SetTitle('#it{f}_{non-prompt}')
        hV2VsFrac[iPt].GetYaxis().SetTitle('#it{v}_{2}^{obs}')
        hV2VsFrac[iPt].GetYaxis().SetTitleOffset(1.5)
        hV2VsFrac[iPt].GetYaxis().SetDecimals()

        hV2VsFrac[iPt].SetTitle(f"{ptStrings[iPt]}")
        hV2VsFrac[iPt].SetStats(0)
        hV2VsFrac[iPt].Draw("EZ")
        gFracVsV2[iPt].Draw("same pZ")
        lineat0.Draw("same")

        outFile.mkdir(f"pt_{int(ptMin*10)}_{int(ptMax*10)}")
        outFile.cd(f"pt_{int(ptMin*10)}_{int(ptMax*10)}")
        cFrac[-1].Write()
        gFracVsV2[iPt].Write('gV2VsFrac')
        hV2VsFrac[iPt].Write('hV2VsFrac')

    outFile.cd()
    PtTit = "#it{p}_{T} GeV/#it{c}"
    leg = TLegend(0.55, 0.75, 0.88, 0.89)
    leg.SetTextSize(0.045)
    leg.SetBorderSize(0)
    leg.SetFillStyle(0)
    SetObjectStyle(hV2VsPtFD, color=GetROOTColor("kAzure+4"), fillstyle=1, markerstyle=20, markersize=1.2, linestyle=2)
    SetObjectStyle(hV2VsPtPrompt, color=GetROOTColor("kRed+1"), fillstyle=1, markerstyle=20, markersize=1.2)

    lineat0 = TLine(hV2VsPtFD.GetXaxis().GetXmin(), 0, hV2VsPtFD.GetXaxis().GetXmax(), 0) 
    lineat0.SetLineStyle(2)
    lineat0.SetLineColor(kBlack)

    cV2VsPtFD = TCanvas("cV2VsPtFD", "non-prompt v2 versus pt", 800, 800)
    set_frame_margin(cV2VsPtFD)
    cV2VsPtFD.cd()
    hV2VsPtFD.Draw("")
    lineat0.Draw("same")
    hV2VsPtFD.GetXaxis().SetTitle(PtTit)
    hV2VsPtFD.GetYaxis().SetTitle("Non-prompt #it{v_{2}}")
    hV2VsPtFD.GetYaxis().SetRangeUser(-0.05, 0.35)
    hV2VsPtFD.SetMarkerStyle(20)
    hV2VsPtFD.SetMarkerSize(2)
    hV2VsPtFD.GetYaxis().SetNoExponent()
    hV2VsPtFD.GetYaxis().SetDecimals()

    cV2VsPtPrompt = TCanvas("cV2VsPtPrompt", "prompt v2 versus pt", 800, 800)
    set_frame_margin(cV2VsPtPrompt)
    cV2VsPtPrompt.cd()
    hV2VsPtPrompt.Draw("")
    lineat0.Draw("same")
    hV2VsPtPrompt.GetXaxis().SetTitle(PtTit)
    hV2VsPtPrompt.GetYaxis().SetTitle("Prompt #it{v}_{2}")
    hV2VsPtPrompt.GetYaxis().SetRangeUser(-0.10, 0.35)
    hV2VsPtPrompt.GetYaxis().SetNoExponent()
    hV2VsPtPrompt.GetYaxis().SetDecimals()

    cPromptAndFDV2 = TCanvas("cPromptAndFDV2", "prompt and non-prompt v2 versus pt", 800, 800)
    set_frame_margin(cPromptAndFDV2)
    cPromptAndFDV2.cd()
    hV2VsPtFD.GetYaxis().SetTitle("#it{v_{2}}")
    hV2VsPtFD.Draw("")
    hV2VsPtPrompt.Draw("same")
    lineat0.Draw("same")

    leg.AddEntry(hV2VsPtFD, "Non-prompt #it{v_{2}}", "lp")
    leg.AddEntry(hV2VsPtPrompt, "Prompt #it{v_{2}}", "lp")
    leg.Draw("same")

    canvV2VsFrac.cd(len(ptMins) + 1)
    hV2VsPtPrompt.GetYaxis().SetTitleSize(0.04)
    hV2VsPtPrompt.GetYaxis().SetTitle("#it{v_{2}}")
    hV2VsPtPrompt.GetYaxis().SetTitleOffset(1.3)
    hV2VsPtPrompt.GetYaxis().SetLabelSize(0.035)
    hV2VsPtPrompt.GetXaxis().SetTitleSize(0.04)
    hV2VsPtPrompt.Draw("")
    hV2VsPtFD.Draw("same")
    lineat0.Draw("same")
    leg.Draw("same")

    hV2VsPtFD.Write()
    hV2VsPtPrompt.Write()
    hV2VsPtFDUnc.Write()
    hV2VsPtPromptUnc.Write()
    cV2VsPtFD.SaveAs(f"{outputDir}/v2VsPtFD.pdf")
    cV2VsPtPrompt.SaveAs(f"{outputDir}/v2VsPtPrompt.pdf")
    cPromptAndFDV2.SaveAs(f"{outputDir}/v2VsPtPromptAndFD.pdf")
    cV2VsPtFD.SaveAs(f"{outputDir}/v2VsPtFD.png")
    cV2VsPtPrompt.SaveAs(f"{outputDir}/v2VsPtPrompt.png")
    cPromptAndFDV2.SaveAs(f"{outputDir}/v2VsPtPromptAndFD.png")
    canvV2VsFrac.SaveAs(f"{outputDir}/v2VsFrac_allPts.png")

def main_v2_vs_frac(flow_config, ry_input_dir, frac_input_dir, correlated=False, multitrial=False, batch=False, outputdir=''):
    
    if batch:
        gROOT.SetBatch(True)

    rawYieldFiles = load_root_files(ry_input_dir, prefix='raw_yields_')
    fracFiles = load_root_files(frac_input_dir, prefix='frac_')
    if len(rawYieldFiles) < 3:
        raise ValueError(f'[{ry_input_dir}] At least 3 raw yield files are required for v2 vs frac computation, found {len(rawYieldFiles)}')
    if multitrial and len(rawYieldFiles) != len(fracFiles):
        logger(f"Number of raw yield and fraction files do not match: {len(rawYieldFiles)} != {len(fracFiles)}, "
               f"Attempting to match fraction files to raw yield files based on cutset suffixes... ", level="WARNING")
        # Retrieve cutset suffixes from raw yield files
        ry_suffixes = []
        for f in rawYieldFiles:
            _, _, cutset_suffix = os.path.basename(f).split('_')
            ry_suffixes.append(cutset_suffix.replace('.root', ''))
        # Retrieve cutset suffixes from fraction files
        frac_suffixes = []
        for f in fracFiles:
            _, cutset_suffix = os.path.basename(f).split('_')
            frac_suffixes.append(cutset_suffix.replace('.root', ''))
        # Filter the frac files and match them to the raw yield files
        matched_frac_files = []
        for ry_suffix in ry_suffixes:
            if ry_suffix in frac_suffixes:
                matched_index = frac_suffixes.index(ry_suffix)
                matched_frac_files.append(fracFiles[matched_index])
            else:
                raise ValueError(f'No matching fraction file found for raw yield file with suffix: {ry_suffix}')
        fracFiles = matched_frac_files
        logger(f"Matched fraction files: {fracFiles} vs {rawYieldFiles}", level="WARNING")

    if len(fracFiles) != len(rawYieldFiles):
        raise ValueError(f'Number of eff and frac files do not match: {len(fracFiles)} != {len(rawYieldFiles)}')

    with open(flow_config, 'r') as f:
        config = yaml.safe_load(f)
        ptMins = config['ptbins'][:-1]
        ptMaxs = config['ptbins'][1:]
        nPtBins = len(ptMins)
        if correlated:
            sig = config['cut_variation']['corr_bdt_cut']['sig']
            CutSets = [len(list(np.arange(sig['min'][i], sig['max'][i], sig['step'][i]))) - 1 for i in range(nPtBins)]
        else:
            sig = config['cut_variation']['uncorr_bdt_cut']['sig']
            CutSets = [len(sig[i]) - 1 for i in range(nPtBins)]

    # Use ry_input_dir, not frac_input_dir, to define output dir
    # because in multitrial frac_input_dir is constant and the
    # reference results would be overwritten
    ry_input_dir = os.path.normpath(ry_input_dir)
    outputDir = ry_input_dir.replace('raw_yields', 'v2') if outputdir == '' else f"{outputdir}/v2"
    v2_vs_frac(
        config,
        ptMins,
        ptMaxs,
        CutSets,
        rawYieldFiles,
        fracFiles,
        multitrial,
        outputDir
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Arguments')
    parser.add_argument("flow_config", metavar="text",
                        default="config.yaml", help="flow configuration file")
    parser.add_argument('ry_input_dir', metavar='text',
                        default="input", help="input path to the raw yields")
    parser.add_argument('frac_input_dir', metavar='text',
                        default="input", help="input path to the data driven fractions")
    parser.add_argument('--batch', '-b', action='store_true',
                        help="run in batch mode")
    parser.add_argument('--correlated', '-corr', action='store_true',
                        help="perform correlated analysis")
    parser.add_argument('--multitrial', action='store_true',
                        help='suppress redundant prints')
    parser.add_argument('--outputdir', '-o', metavar='text', default='',
                        help='output directory (used for systematics)', required=False)
    args = parser.parse_args()

    if args.multitrial:
        ROOT.gErrorIgnoreLevel = ROOT.kError

    main_v2_vs_frac(
        args.flow_config,
        args.ry_input_dir,
        args.frac_input_dir,
        args.correlated,
        args.multitrial,
        args.batch,
        args.outputdir
    )
