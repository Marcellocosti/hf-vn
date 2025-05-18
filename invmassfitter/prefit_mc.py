from ROOT import gROOT, TFile, TCanvas, RooRealVar, RooDataHist, RooGaussian, RooAddPdf, RooCrystalBall, RooFit, RooArgList, RooArgSet, TH1D
import numpy as np
import sys
import argparse
import yaml
import os
script_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(f'{script_dir}/../')
print(f'{script_dir}/../')
from flow_analysis_utils import get_centrality_bins

# Set ROOT batch mode
gROOT.SetBatch(1)

parser = argparse.ArgumentParser(description='Arguments')
parser.add_argument('fitConfigFileName', metavar='text', default='config_Dplus_Fit.yml')
parser.add_argument('inFileName', metavar='text', default='')
parser.add_argument('cent', metavar='text', default='')
parser.add_argument("--outputdir", "-o", metavar="text",
                    default=".", help="output directory")
parser.add_argument("--suffix", "-s", metavar="text",
                    default="", help="suffix for output files")
args = parser.parse_args()

with open(args.fitConfigFileName, 'r', encoding='utf8') as ymlfitConfigFile:
    fitConfig = yaml.load(ymlfitConfigFile, yaml.FullLoader)

print(f"args.cent: {args.cent}")
cent, centMinMax = get_centrality_bins(args.cent)


# Define pT and mass bins
ptMins = fitConfig['ptmins']
ptMaxs = fitConfig['ptmaxs']
massMins = fitConfig['MassMin']
massMaxs = fitConfig['MassMax']

# Open input ROOT file
infile = TFile.Open(args.inFileName, 'r')
if not infile or not infile.IsOpen():
    print(f"ERROR: Cannot open file {args.inFileName}! Exiting.")
    sys.exit()

# Create output ROOT file
outFile = TFile(f"{args.outputdir}/Prefit_mc_prompt_enhanced.root", "RECREATE")

# Define histograms for fit parameters
hists = {}
models = ["kGaus", "k2Gaus", "kDoubleCBSymm", "kDoubleCBAsymm"]
params = {
        "kGaus": ["Mean", "Sigma"], 
        "k2Gaus": ["Mean", "Sigma", "Frac", "Sigma2"],
        "kDoubleCBSymm": ["Mean", "Sigma", "Alpha1", "N1"],
        "kDoubleCBAsymm": ["Mean", "Sigma", "Alpha1", "N1", "Alpha2", "N2"]
    }

for model in models:
    hists[model] = {param: TH1D(f"{param}", f"{param}", len(ptMins), 0, len(ptMins)) for param in params[model]}
    hists[model].update({"Chi2": TH1D(f"hChi2", "Chi2", len(ptMins), 0, len(ptMins))})

# Loop over pT bins
for iPt, (ptMin, ptMax) in enumerate(zip(ptMins, ptMaxs)):
    print(f"Fitting pT bin {ptMin} - {ptMax} GeV/c")

    print(f"infile.ls(): {infile.ls()}")
    print(f"Getting histo from: cent_bins{centMinMax[0]}_{centMinMax[1]}/pt_bins{ptMin}_{ptMax}/hPromptMass")
    hMass = infile.Get(f'cent_bins{centMinMax[0]}_{centMinMax[1]}/pt_bins{ptMin}_{ptMax}/hPromptMass')
    if not hMass:
        print(f"WARNING: Histogram not found for pT {ptMin}-{ptMax}. Skipping.")
        continue

    mass = RooRealVar("mass", "M(KKPi)", massMins[iPt], massMaxs[iPt], "GeV/c^2")
    data_hist = RooDataHist(f"data_hist_{iPt}", "Dataset from TH1", RooArgList(mass), hMass)

    # Define fit models
    mean = RooRealVar(f"mean_{iPt}", "Mean", 1.87, 1.80, 1.95)
    sigma = RooRealVar(f"sigma_{iPt}", "Sigma", 0.010, 0.005, 0.020)
    gaus = RooGaussian(f"gaus_{iPt}", "Gaussian", mass, mean, sigma)

    sigma2 = RooRealVar(f"sigma2_{iPt}", "Sigma2", 0.015, 0.005, 0.030)
    frac = RooRealVar(f"frac_{iPt}", "Fraction", 0.5, 0.0, 1.0)
    gaus2 = RooGaussian(f"gaus2_{iPt}", "Gaussian2", mass, mean, sigma2)
    double_gaus = RooAddPdf(f"double_gaus_{iPt}", "Double Gaussian", RooArgList(gaus, gaus2), RooArgList(frac))

    alpha1 = RooRealVar(f"alpha1_{iPt}", "Alpha Left", 1.5, 0.5, 3.0)
    n1 = RooRealVar(f"n1_{iPt}", "n Left", 5, 1, 10)
    symm_cb = RooCrystalBall(f"symm_cb_{iPt}", "Symmetric CB", mass, mean, sigma, alpha1, n1, alpha1, n1)

    alpha2 = RooRealVar(f"alpha2_{iPt}", "Alpha Right", 1.5, 0.5, 3.0)
    n2 = RooRealVar(f"n2_{iPt}", "n Right", 5, 1, 10)
    asymm_cb = RooCrystalBall(f"asymm_cb_{iPt}", "Asymmetric CB", mass, mean, sigma, alpha1, n1, alpha2, n2)

    # Store models in a dictionary
    models_dict = {
        "kGaus": gaus,
        "k2Gaus": double_gaus,
        "kDoubleCBSymm": symm_cb,
        "kDoubleCBAsymm": asymm_cb
    }

    # Fit and store results
    for model_name, model in models_dict.items():
        print(f"  Fitting with {model_name}...")

        fit_result = model.fitTo(data_hist, RooFit.Save(), RooFit.PrintLevel(-1))

        # Create a frame
        frame = mass.frame()
        data_hist.plotOn(frame, RooFit.Name("data"))
        model.plotOn(frame, RooFit.LineColor(2), RooFit.Name("fit"))

        # Special case: Draw second Gaussian for Double Gaussian
        if model_name == "DoubleGaussian":
            gaus2.plotOn(frame, RooFit.LineColor(4), RooFit.LineStyle(2), RooFit.Name("gaus2"))

        # Compute chi2/ndf
        chi2 = -1.0
        if frame.findObject("fit"):
            chi2 = frame.chiSquare("fit", "data")
        print(f"    chi2/ndf = {chi2}")

        # Save parameters to histograms
        hists[model_name]["Mean"].SetBinContent(iPt + 1, mean.getVal())
        hists[model_name]["Sigma"].SetBinContent(iPt + 1, sigma.getVal())
        hists[model_name]["Chi2"].SetBinContent(iPt + 1, chi2)

        if model_name == "k2Gaus":
            hists[model_name]["Frac"].SetBinContent(iPt + 1, frac.getVal())
            hists[model_name]["Sigma2"].SetBinContent(iPt + 1, sigma2.getVal())            
        if model_name in ["kDoubleCBSymm", "kDoubleCBAsymm"]:
            hists[model_name]["Alpha1"].SetBinContent(iPt + 1, alpha1.getVal())
            hists[model_name]["N1"].SetBinContent(iPt + 1, n1.getVal())
        if model_name == "kDoubleCBAsymm":
            hists[model_name]["Alpha2"].SetBinContent(iPt + 1, alpha2.getVal())
            hists[model_name]["N2"].SetBinContent(iPt + 1, n2.getVal())

        # Save the fit plot
        canvas = TCanvas(f"cMass_{model_name}_{iPt}", f"Mass Fit {model_name} {ptMin}-{ptMax} GeV/c", 800, 600)
        frame.Draw()

        # Store results in ROOT file
        outFile.mkdir(f"{model_name}/pt_bins{ptMin}_{ptMax}")
        outFile.cd(f"{model_name}/pt_bins{ptMin}_{ptMax}")
        model.Write(f"{model_name}_pt{ptMin*10:.0f}_{ptMax*10:.0f}")
        canvas.Write(f"canvas_{model_name}_pt{ptMin*10:.0f}_{ptMax*10:.0f}")

# Save histograms
outFile.cd()
for model in models:
    outFile.cd(f"{model}")
    print(f"hists[model]: {hists[model]}")
    for hist in hists[model].values():
        hist.Write()

outFile.Close()
print("All fits and parameter histograms saved successfully!")
