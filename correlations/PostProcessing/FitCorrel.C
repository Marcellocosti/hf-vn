// Copyright 2019-2020 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file DhCorrelationExtraction.cxx
/// \brief class for D-h correlation extraction
/// \author Samuele Cattaruzzi <samuele.cattaruzzi@cern.ch>
/// \author Swapnesh Santosh Khade <swapnesh.santosh.khade@cern.ch>

#include "DhCorrelationFitter.h"
#include "Riostream.h"

#include <TF1.h>
#include <TH1D.h>
#include <TMath.h>
#include <TPaveText.h>
#include <TROOT.h>
#include <TStyle.h>
#include <TSystem.h>

#include <rapidjson/document.h>
#include <rapidjson/filereadstream.h>

#include <cstdio>
#include <iostream>
#include <string>
#include <vector>

using namespace std;
using namespace rapidjson;

bool removeNSPeakLowPt = false;
bool doCorrelation = false;

template <typename ValueType>
void readArray(const Value& jsonArray, vector<ValueType>& output)
{
  for (auto it = jsonArray.Begin(); it != jsonArray.End(); it++) {
    auto value = it->template Get<ValueType>();
    output.emplace_back(value);
  }
}

void SetTH1HistoStyle(TH1D*& histo, TString hTitle, TString hXaxisTitle, TString hYaxisTitle,
                      Style_t markerStyle, Color_t markerColor, Double_t markerSize,
                      Color_t lineColor, Int_t lineWidth, Float_t hTitleXaxisOffset = 1.3, Float_t hTitleYaxisOffset = 1.3,
                      Float_t hTitleXaxisSize = 0.045, Float_t hTitleYaxisSize = 0.045, Float_t hLabelXaxisSize = 0.045, Float_t hLabelYaxisSize = 0.045,
                      Bool_t centerXaxisTitle = false, Bool_t centerYaxisTitle = false);
void SetTH1HistoStyle(TH1F*& histo, TString hTitle, TString hXaxisTitle, TString hYaxisTitle,
                      Style_t markerStyle, Color_t markerColor, Double_t markerSize,
                      Color_t lineColor, Int_t lineWidth, Float_t hTitleXaxisOffset = 1.3, Float_t hTitleYaxisOffset = 1.3,
                      Float_t hTitleXaxisSize = 0.045, Float_t hTitleYaxisSize = 0.045, Float_t hLabelXaxisSize = 0.045, Float_t hLabelYaxisSize = 0.045,
                      Bool_t centerXaxisTitle = false, Bool_t centerYaxisTitle = false);

void FitCorrel(const TString cfgFileName = "config_CorrAnalysis.json")
{
  std::cout << "=========================== " << std::endl;
  std::cout << "   D-h Correlation Fitting  " << std::endl;
  std::cout << "=========================== " << std::endl;
  gStyle->SetOptStat(0);
  gStyle->SetPadLeftMargin(0.2);
  gStyle->SetPadRightMargin(0.005);
  gStyle->SetPadBottomMargin(0.2);
  gStyle->SetFrameLineWidth(2);
  gStyle->SetLineWidth(2);
  gStyle->SetCanvasDefH(1126);
  gStyle->SetCanvasDefW(1840);

  std::cout << "Style set" << std::endl;
  // Load config
  std::cout << "Loading config file: " << cfgFileName.Data() << std::endl;
  FILE* configFile = fopen(cfgFileName.Data(), "r");
  Document config;
  char readBuffer[65536];
  FileReadStream is(configFile, readBuffer, sizeof(readBuffer));
  config.ParseStream(is);
  fclose(configFile);
  std::cout << "Config file loaded" << std::endl;

  string CodeNameAnalysis = config["CodeName"].GetString();
  gSystem->Exec(Form("rm -rf Output_CorrelationFitting_%s_Root/ Output_CorrelationFitting_%s_png/", CodeNameAnalysis.data(), CodeNameAnalysis.data()));
  gSystem->Exec(Form("mkdir Output_CorrelationFitting_%s_Root/ Output_CorrelationFitting_%s_png/", CodeNameAnalysis.data(), CodeNameAnalysis.data()));
  std::cout << "Output directories created" << std::endl;
  string inputFileNameFit = config["InputFileNameFitCorr"].GetString();
  const TString inFileName = Form("Output_CorrelationExtraction_%s_Root/%s", CodeNameAnalysis.data(), inputFileNameFit.data());
  std::cout << "Input file name for fitting: " << inFileName.Data() << std::endl;
  bool isReflected = config["IsRiflected"].GetBool();
  bool drawSystematicErrors = config["DrawSystematics"].GetBool();
  bool sameSystematics = config["SameSystematics"].GetBool();
  bool shiftBaseUp = config["ShiftBaseUp"].GetBool();
  bool shiftBaseDown = config["ShiftBaseDown"].GetBool();
  std::cout << "Input file name retrieved from config" << std::endl;

  std::vector<double> binsInvMassIntervals;
  std::vector<double> binsPtCandIntervalsVec;
  std::vector<double> binsPtHadIntervals;
  std::vector<int> fitFunc;
  std::cout << "Reading bin intervals and fit function from config" << std::endl;

  const Value& InvMassValue = config["binsInvMassIntervals"];
  readArray(InvMassValue, binsInvMassIntervals);
  std::cout << "binsInvMassIntervals size: " << binsInvMassIntervals.size() << std::endl;

  const Value& PtCandValue = config["binsPtCandIntervals"];
  readArray(PtCandValue, binsPtCandIntervalsVec);
  std::cout << "binsPtCandIntervals size: " << binsPtCandIntervalsVec.size() << std::endl;

  const Value& PtHadValue = config["binsPtHadIntervals"];
  readArray(PtHadValue, binsPtHadIntervals);
  std::cout << "binsPtHadIntervals size: " << binsPtHadIntervals.size() << std::endl;

  std::vector<double> parVals;
  std::vector<double> parLowBounds;
  std::vector<double> parUpperBounds;
  std::cout << "Reading fit parameters from config" << std::endl;

  const Value& ParVals = config["parVals"];
  readArray(ParVals, parVals);
  std::cout << "parVals size: " << parVals.size() << std::endl;

  const Value& ParLowBounds = config["parLowBounds"];
  readArray(ParLowBounds, parLowBounds);
  std::cout << "parLowBounds size: " << parLowBounds.size() << std::endl;

  const Value& ParUpperBounds = config["parUpperBounds"];
  readArray(ParUpperBounds, parUpperBounds);
  std::cout << "parUpperBounds size: " << parUpperBounds.size() << std::endl;

  const int nBinsInvMass = binsInvMassIntervals.size() - 1;
  const int nBinsPtCand = binsPtCandIntervalsVec.size() - 1;
  const int nBinsPtHad = binsPtHadIntervals.size() - 1;
  const int npars = parVals.size();
  std::cout << "nBinsInvMass: " << nBinsInvMass << std::endl;

  double binsPtCandIntervals[nBinsPtCand + 1];
  for (int i = 0; i < nBinsPtCand + 1; i++) {
    binsPtCandIntervals[i] = binsPtCandIntervalsVec[i];
  }
  std::cout << "Converted binsPtCandIntervals to C-style array" << std::endl;

  const Value& FitFuncValue = config["FitFunction"];
  readArray(FitFuncValue, fitFunc);
  std::cout << "fitFunc size: " << fitFunc.size() << std::endl;

  int fixBase = config["FixBaseline"].GetInt();
  int fixMean = config["FixMean"].GetInt();
  std::cout << "FixBaseline: " << fixBase << std::endl;

  int nBaselinePoints = config["nBaselinePoints"].GetInt();
  vector<int> pointsForBaselineVec;
  const Value& pointsForBaselineValue = config["binsForBaseline"];
  readArray(pointsForBaselineValue, pointsForBaselineVec);
  if (pointsForBaselineVec.size() != nBaselinePoints) {
    cout << "ERROR: size of the vector pointsForBaseline is different from the number of nBaselinePoints" << endl;
    return;
  }
  std::cout << "nBaselinePoints: " << nBaselinePoints << std::endl;
  int pointsForBaseline[nBaselinePoints];
  for (int i = 0; i < nBaselinePoints; i++) {
    pointsForBaseline[i] = pointsForBaselineVec[i];
  }
  std::cout << "Converted pointsForBaseline to C-style array" << std::endl;

  std::cout << "=========================== " << std::endl;
  std::cout << "Input variables from config" << std::endl;
  for (int iBinPtCand = 0; iBinPtCand < nBinsPtCand; iBinPtCand++) {
    std::cout << "iPt = " << iBinPtCand + 1 << "  FitFunction    = " << fitFunc[iBinPtCand] << std::endl;
  }
  std::cout << "FixBaseline    = " << fixBase << std::endl;
  std::cout << "FixMean    = " << fixMean << std::endl;
  std::cout << "=========================== " << std::endl;
  std::cout << " " << std::endl;

  // TODO: reflections
  bool refl = false;
  std::cout << "isReflected: " << isReflected << std::endl;

  // Input file
  std::cout << "Opening input file: " << inFileName.Data() << std::endl;
  TFile* inFile = new TFile(inFileName.Data());
  TFile* inFileSystematicErrors = new TFile("OutputSystematicUncertainties/SystematicUncertaintesAngCorrMerged.root", "recreate");
  TFile* inFileFitSystematicErrors = new TFile("OutputSystematicUncertainties/SystematicUncertaintesFitPhysObsMerged.root", "recreate");
  std::cout << "Input file opened" << std::endl;

  // Canvas
  TCanvas* CanvasCorrPhi[nBinsPtHad][nBinsInvMass];
  std::cout << "Canvas declared" << std::endl;

  // Histograms
  TH1D* hCorrPhi[nBinsPtCand][nBinsPtHad][nBinsInvMass];
  TH1F* hSystematicErrors[nBinsPtCand][nBinsPtHad][nBinsInvMass];
  TH1D* hSystematicErrorsPlot[nBinsPtCand][nBinsPtHad][nBinsInvMass];
  std::cout << "Histograms declared" << std::endl;

  // DhCorrelationFitter
  const double fMin{-0.5 * TMath::Pi()}, fMax{1.5 * TMath::Pi()}; // limits for the fitting function
  DhCorrelationFitter* corrFitter[nBinsPtHad][nBinsPtCand][nBinsInvMass];
  std::cout << "DhCorrelationFitter declared" << std::endl;

  // Output histograms
  TH1D* hBaselin[nBinsPtHad][nBinsInvMass];
  TH1D* hNSYield[nBinsPtHad][nBinsInvMass];
  TH1D* hNSSigma[nBinsPtHad][nBinsInvMass];
  TH1D* hASYield[nBinsPtHad][nBinsInvMass];
  TH1D* hASSigma[nBinsPtHad][nBinsInvMass];
  TH1D* hBeta[nBinsPtHad][nBinsInvMass];
  TH1D* hNSYieldBinCount[nBinsPtHad][nBinsInvMass];
  TH1D* hASYieldBinCount[nBinsPtHad][nBinsInvMass];
  TH1D* hv2Delta[nBinsPtHad][nBinsInvMass];

  // extract TH1D and prepare fit
  for (int iBinInvMass = 0; iBinInvMass < nBinsInvMass; iBinInvMass++) {
    for (int iBinPtCand = 0; iBinPtCand < nBinsPtCand; iBinPtCand++) {
      for (int iBinPtHad = 0; iBinPtHad < nBinsPtHad; iBinPtHad++) {
        if (isReflected) {
          hCorrPhi[iBinPtCand][iBinPtHad][iBinInvMass] = reinterpret_cast<TH1D*>(inFile->Get(Form("hCorrectedCorrReflected_PtCand%.0fto%.0f_PtAssoc%.0fto%.0f_InvMassBin%d", binsPtCandIntervals[iBinPtCand], binsPtCandIntervals[iBinPtCand + 1], binsPtHadIntervals[iBinPtHad], binsPtHadIntervals[iBinPtHad + 1], iBinInvMass+1)));
        } else {
          hCorrPhi[iBinPtCand][iBinPtHad][iBinInvMass] = reinterpret_cast<TH1D*>(inFile->Get(Form("hCorrectedCorr_PtCand%.0fto%.0f_PtAssoc%.0fto%.0f_InvMassBin%d", binsPtCandIntervals[iBinPtCand], binsPtCandIntervals[iBinPtCand + 1], binsPtHadIntervals[iBinPtHad], binsPtHadIntervals[iBinPtHad + 1], iBinInvMass+1)));
        }

        corrFitter[iBinPtHad][iBinPtCand][iBinInvMass] = new DhCorrelationFitter(reinterpret_cast<TH1F*>(hCorrPhi[iBinPtCand][iBinPtHad][iBinInvMass]), fMin, fMax);
        corrFitter[iBinPtHad][iBinPtCand][iBinInvMass]->SetHistoIsReflected(refl);
        corrFitter[iBinPtHad][iBinPtCand][iBinInvMass]->SetFixBaseline(fixBase);
        corrFitter[iBinPtHad][iBinPtCand][iBinInvMass]->SetBaselineUpOrDown(shiftBaseUp, shiftBaseDown);
        corrFitter[iBinPtHad][iBinPtCand][iBinInvMass]->SetPointsForBaseline(nBaselinePoints, pointsForBaseline);
        //corrFitter[iBinPtHad][iBinPtCand][iBinInvMass]->Setv2(v2AssocPart[iBinPtCand], v2Dmeson[iBinPtCand]);
        corrFitter[iBinPtHad][iBinPtCand][iBinInvMass]->SetReflectedCorrHisto(isReflected);

        corrFitter[iBinPtHad][iBinPtCand][iBinInvMass]->SetFixMean(fixMean);
        corrFitter[iBinPtHad][iBinPtCand][iBinInvMass]->SetPtRanges(binsPtCandIntervals[iBinPtCand], binsPtCandIntervals[iBinPtCand + 1], binsPtHadIntervals[iBinPtHad], binsPtHadIntervals[iBinPtHad + 1]);
        corrFitter[iBinPtHad][iBinPtCand][iBinInvMass]->SetExternalValsAndBounds(npars, parVals.data(), parLowBounds.data(), parUpperBounds.data()); // these are starting points and limits...
      }
    }
  }

  // Plots and fit
  for (int iBinInvMass = 0; iBinInvMass < nBinsInvMass; iBinInvMass++) {
    std::cout << "[INFO] InvMass: " << binsInvMassIntervals[iBinInvMass] << " - "<< binsInvMassIntervals[iBinInvMass+1] << std::endl;
    for (int iBinPtHad = 0; iBinPtHad < nBinsPtHad; iBinPtHad++) {
      std::cout << "[INFO] PtHad: " << binsPtHadIntervals[iBinPtHad] << " - "<< binsPtHadIntervals[iBinPtHad+1] << std::endl;
      CanvasCorrPhi[iBinPtHad][iBinInvMass] = new TCanvas(Form("CanvasCorrPhi_PtBinAssoc%d_InvMassBin%d", iBinPtHad + 1, iBinInvMass+1), Form("CorrPhiDs_PtBinAssoc%d_InvMassBin%d", iBinPtHad + 1, iBinInvMass+1));
      
      if (nBinsPtCand <= 4) {
          CanvasCorrPhi[iBinPtHad][iBinInvMass]->Divide(2, 2);
        }
        if (nBinsPtCand > 4 && nBinsPtCand <= 6) {
          CanvasCorrPhi[iBinPtHad][iBinInvMass]->Divide(3, 2);
        }

      // histograms with fir parameters
      hBaselin[iBinPtHad][iBinInvMass] = new TH1D(Form("hBaselin_PtBinAssoc%d_InvMassBin%d", iBinPtHad + 1, iBinInvMass+1), "", nBinsPtCand, binsPtCandIntervals);
      hNSYield[iBinPtHad][iBinInvMass] = new TH1D(Form("hNSYield_PtBinAssoc%d_InvMassBin%d", iBinPtHad + 1, iBinInvMass+1), "", nBinsPtCand, binsPtCandIntervals);
      hNSSigma[iBinPtHad][iBinInvMass] = new TH1D(Form("hNSSigma_PtBinAssoc%d_InvMassBin%d", iBinPtHad + 1, iBinInvMass+1), "", nBinsPtCand, binsPtCandIntervals);
      hASYield[iBinPtHad][iBinInvMass] = new TH1D(Form("hASYield_PtBinAssoc%d_InvMassBin%d", iBinPtHad + 1, iBinInvMass+1), "", nBinsPtCand, binsPtCandIntervals);
      hASSigma[iBinPtHad][iBinInvMass] = new TH1D(Form("hASSigma_PtBinAssoc%d_InvMassBin%d", iBinPtHad + 1, iBinInvMass+1), "", nBinsPtCand, binsPtCandIntervals);
      hBeta[iBinPtHad][iBinInvMass] = new TH1D(Form("hBeta_PtBinAssoc%d_InvMassBin%d", iBinPtHad + 1, iBinInvMass+1), "", nBinsPtCand, binsPtCandIntervals);
      hNSYieldBinCount[iBinPtHad][iBinInvMass] = new TH1D(Form("hNSYieldBinCount_PtBinAssoc%d_InvMassBin%d", iBinPtHad + 1, iBinInvMass+1), "", nBinsPtCand, binsPtCandIntervals);
      hASYieldBinCount[iBinPtHad][iBinInvMass] = new TH1D(Form("hASYieldBinCount_PtBinAssoc%d_InvMassBin%d", iBinPtHad + 1, iBinInvMass+1), "", nBinsPtCand, binsPtCandIntervals);
      hv2Delta[iBinPtHad][iBinInvMass] = new TH1D(Form("hv2Delta_PtBinAssoc%d_InvMassBin%d", iBinPtHad + 1, iBinInvMass+1), "", nBinsPtCand, binsPtCandIntervals);

      for (int iBinPtCand = 0; iBinPtCand < nBinsPtCand; iBinPtCand++) {
        std::cout << "[INFO] PtCand: " << binsPtCandIntervals[iBinPtCand] << " - "<< binsPtCandIntervals[iBinPtCand+1] << std::endl;
        SetTH1HistoStyle(hCorrPhi[iBinPtCand][iBinPtHad][iBinInvMass], "", "#Delta#phi [rad]", "#frac{dN^{assoc}}{d#Delta#phi} [rad^{-1}]", kFullCircle, kRed + 1, 1.4, kRed + 1, 3);

        CanvasCorrPhi[iBinPtHad][iBinInvMass]->cd(iBinPtCand + 1);
        CanvasCorrPhi[iBinPtHad][iBinInvMass]->SetTickx();
        CanvasCorrPhi[iBinPtHad][iBinInvMass]->SetTicky();
        hCorrPhi[iBinPtCand][iBinPtHad][iBinInvMass]->SetStats(0);
        hCorrPhi[iBinPtCand][iBinPtHad][iBinInvMass]->SetMinimum(0);

        // Fit
        corrFitter[iBinPtHad][iBinPtCand][iBinInvMass]->SetFuncType(static_cast<DhCorrelationFitter::FunctionType>(fitFunc[iBinPtCand]));
        corrFitter[iBinPtHad][iBinPtCand][iBinInvMass]->Fitting(kTRUE, kTRUE); // the first term is for drawing the fit functions, the second argument is useExternalParams

        TF1* fFit = corrFitter[iBinPtHad][iBinPtCand][iBinInvMass]->GetFitFunction();

        // Title of the histogram
        TPaveText* pttext = new TPaveText(0.15, 0.9, 0.85, 0.95, "NDC");
        pttext->SetFillStyle(0);
        pttext->SetBorderSize(0);
        TText* tpT = pttext->AddText(0., 0.8, Form("%.0f < p_{T}^{D_{s}} < %.0f GeV/c, p_{T}^{assoc} > %.1f GeV/c", binsPtCandIntervals[iBinPtCand], binsPtCandIntervals[iBinPtCand + 1], binsPtHadIntervals[iBinPtHad]));

        // Fill the histograms with the fit parameters
        if (doCorrelation) {
          hBaselin[iBinPtHad][iBinInvMass]->SetBinContent(iBinPtCand + 1, corrFitter[iBinPtHad][iBinPtCand][iBinInvMass]->GetPedestal());
          hBaselin[iBinPtHad][iBinInvMass]->SetBinError(iBinPtCand + 1, corrFitter[iBinPtHad][iBinPtCand][iBinInvMass]->GetPedestalError());
          if (iBinPtCand == 0 && removeNSPeakLowPt) {
            hNSYield[iBinPtHad][iBinInvMass]->SetBinContent(iBinPtCand + 1, -1);
            hNSYield[iBinPtHad][iBinInvMass]->SetBinError(iBinPtCand + 1, 0);

            hNSSigma[iBinPtHad][iBinInvMass]->SetBinContent(iBinPtCand + 1, -1);
            hNSSigma[iBinPtHad][iBinInvMass]->SetBinError(iBinPtCand + 1, 0);

            hBeta[iBinPtHad][iBinInvMass]->SetBinContent(iBinPtCand + 1, -1);
            hBeta[iBinPtHad][iBinInvMass]->SetBinError(iBinPtCand + 1, 0);
          } else {
            hNSYield[iBinPtHad][iBinInvMass]->SetBinContent(iBinPtCand + 1, corrFitter[iBinPtHad][iBinPtCand][iBinInvMass]->GetNSYield());
            hNSYield[iBinPtHad][iBinInvMass]->SetBinError(iBinPtCand + 1, corrFitter[iBinPtHad][iBinPtCand][iBinInvMass]->GetNSYieldError());

            if (fitFunc[iBinPtCand] != 5 && fitFunc[iBinPtCand] != 6) {
              hNSSigma[iBinPtHad][iBinInvMass]->SetBinContent(iBinPtCand + 1, corrFitter[iBinPtHad][iBinPtCand][iBinInvMass]->GetNSSigma());
              hNSSigma[iBinPtHad][iBinInvMass]->SetBinError(iBinPtCand + 1, corrFitter[iBinPtHad][iBinPtCand][iBinInvMass]->GetNSSigmaError());
            } else {
              hNSSigma[iBinPtHad][iBinInvMass]->SetBinContent(iBinPtCand + 1, TMath::Sqrt(1. / corrFitter[iBinPtHad][iBinPtCand][iBinInvMass]->GetNSSigma()));
              Double_t errrel = corrFitter[iBinPtHad][iBinPtCand][iBinInvMass]->GetNSSigmaError() / corrFitter[iBinPtHad][iBinPtCand][iBinInvMass]->GetNSSigma() / 2.;
              hNSSigma[iBinPtHad][iBinInvMass]->SetBinError(iBinPtCand + 1, errrel * TMath::Sqrt(1. / corrFitter[iBinPtHad][iBinPtCand][iBinInvMass]->GetNSSigma()));
            }
          }
          hNSYieldBinCount[iBinPtHad][iBinInvMass]->SetBinContent(iBinPtCand + 1, corrFitter[iBinPtHad][iBinPtCand][iBinInvMass]->GetBinCountingNSYield());
          hNSYieldBinCount[iBinPtHad][iBinInvMass]->SetBinError(iBinPtCand + 1, corrFitter[iBinPtHad][iBinPtCand][iBinInvMass]->GetBinCountingNSYieldErr());

          hASYield[iBinPtHad][iBinInvMass]->SetBinContent(iBinPtCand + 1, corrFitter[iBinPtHad][iBinPtCand][iBinInvMass]->GetASYield());
          hASYield[iBinPtHad][iBinInvMass]->SetBinError(iBinPtCand + 1, corrFitter[iBinPtHad][iBinPtCand][iBinInvMass]->GetASYieldError());

          hASYieldBinCount[iBinPtHad][iBinInvMass]->SetBinContent(iBinPtCand + 1, corrFitter[iBinPtHad][iBinPtCand][iBinInvMass]->GetBinCountingASYield());
          hASYieldBinCount[iBinPtHad][iBinInvMass]->SetBinError(iBinPtCand + 1, corrFitter[iBinPtHad][iBinPtCand][iBinInvMass]->GetBinCountingASYieldErr());
          if (fitFunc[iBinPtCand] != 5 && fitFunc[iBinPtCand] != 6) {
            hASSigma[iBinPtHad][iBinInvMass]->SetBinContent(iBinPtCand + 1, corrFitter[iBinPtHad][iBinPtCand][iBinInvMass]->GetASSigma());
            hASSigma[iBinPtHad][iBinInvMass]->SetBinError(iBinPtCand + 1, corrFitter[iBinPtHad][iBinPtCand][iBinInvMass]->GetASSigmaError());
          } else {
            hASSigma[iBinPtHad][iBinInvMass]->SetBinContent(iBinPtCand + 1, TMath::Sqrt(1. / corrFitter[iBinPtHad][iBinPtCand][iBinInvMass]->GetASSigma()));
            Double_t errrel = corrFitter[iBinPtHad][iBinPtCand][iBinInvMass]->GetASSigmaError() / corrFitter[iBinPtHad][iBinPtCand][iBinInvMass]->GetASSigma() / 2.;
            hASSigma[iBinPtHad][iBinInvMass]->SetBinError(iBinPtCand + 1, errrel * TMath::Sqrt(1. / corrFitter[iBinPtHad][iBinPtCand][iBinInvMass]->GetASSigma()));
          }
          if (fitFunc[iBinPtCand] == 4) { // param beta for gen. gauss
            hBeta[iBinPtHad][iBinInvMass]->SetBinContent(iBinPtCand + 1, corrFitter[iBinPtHad][iBinPtCand][iBinInvMass]->GetBeta());
            hBeta[iBinPtHad][iBinInvMass]->SetBinError(iBinPtCand + 1, corrFitter[iBinPtHad][iBinPtCand][iBinInvMass]->GetBetaError());
          }
        } else {
          hv2Delta[iBinPtHad][iBinInvMass]->SetBinContent(iBinPtCand + 1, corrFitter[iBinPtHad][iBinPtCand][iBinInvMass]->Getv2Delta());
          hv2Delta[iBinPtHad][iBinInvMass]->SetBinError(iBinPtCand + 1, corrFitter[iBinPtHad][iBinPtCand][iBinInvMass]->Getv2DeltaError()); 
        }

        // Draw
        hCorrPhi[iBinPtCand][iBinPtHad][iBinInvMass]->Draw("same");
        pttext->Draw("same");
      }
      CanvasCorrPhi[iBinPtHad][iBinInvMass]->SaveAs(Form("Output_CorrelationFitting_%s_png/CorrPhiDs_PtBinAssoc%d_InvMassBin%d.png", CodeNameAnalysis.data(), iBinPtHad + 1, iBinInvMass+1));
      CanvasCorrPhi[iBinPtHad][iBinInvMass]->SaveAs(Form("Output_CorrelationFitting_%s_Root/CorrPhiDs_PtBinAssoc%d_InvMassBin%d.root", CodeNameAnalysis.data(), iBinPtHad + 1, iBinInvMass+1));
    }
  }

  // histogram with fit parameter and errors
  TFile* outFile = new TFile(Form("Output_CorrelationFitting_%s_Root/CorrPhiDs_FinalPlots.root", CodeNameAnalysis.data()), "RECREATE");
  outFile->cd();
  for (int iBinInvMass = 0; iBinInvMass < nBinsInvMass; iBinInvMass++) {
    for (int iBinPtHad = 0; iBinPtHad < nBinsPtHad; iBinPtHad++) {
      if (doCorrelation) {
        hBaselin[iBinPtHad][iBinInvMass]->Write();
        hNSYield[iBinPtHad][iBinInvMass]->Write();
        hNSSigma[iBinPtHad][iBinInvMass]->Write();
        hASYield[iBinPtHad][iBinInvMass]->Write();
        hASSigma[iBinPtHad][iBinInvMass]->Write();
        hBeta[iBinPtHad][iBinInvMass]->Write();
        hNSYieldBinCount[iBinPtHad][iBinInvMass]->Write();
        hASYieldBinCount[iBinPtHad][iBinInvMass]->Write();
      } else {
        hv2Delta[iBinPtHad][iBinInvMass]->Write();
      }
    }
  }
  outFile->Close();

  return;
}

void SetTH1HistoStyle(TH1D*& histo, TString hTitle, TString hXaxisTitle, TString hYaxisTitle,
                      Style_t markerStyle, Color_t markerColor, Double_t markerSize,
                      Color_t lineColor, Int_t lineWidth, Float_t hTitleXaxisOffset = 1.3, Float_t hTitleYaxisOffset = 1.3,
                      Float_t hTitleXaxisSize = 0.045, Float_t hTitleYaxisSize = 0.045, Float_t hLabelXaxisSize = 0.045, Float_t hLabelYaxisSize = 0.045,
                      Bool_t centerXaxisTitle = false, Bool_t centerYaxisTitle = false)
{

  histo->SetTitle(hTitle.Data());
  histo->GetXaxis()->SetTitle(hXaxisTitle.Data());
  histo->GetYaxis()->SetTitle(hYaxisTitle.Data());
  histo->SetMarkerStyle(markerStyle);
  histo->SetMarkerColor(markerColor);
  histo->SetMarkerSize(markerSize);
  histo->SetLineColor(lineColor);
  histo->SetLineWidth(lineWidth);
  histo->GetXaxis()->SetTitleOffset(hTitleXaxisOffset);
  histo->GetYaxis()->SetTitleOffset(hTitleYaxisOffset);
  histo->GetXaxis()->SetTitleSize(hTitleXaxisSize);
  histo->GetYaxis()->SetTitleSize(hTitleYaxisSize);
  histo->GetXaxis()->SetLabelSize(hLabelXaxisSize);
  histo->GetYaxis()->SetLabelSize(hLabelYaxisSize);
  histo->GetXaxis()->CenterTitle(centerXaxisTitle);
  histo->GetYaxis()->CenterTitle(centerYaxisTitle);

  return;
}

void SetTH1HistoStyle(TH1F*& histo, TString hTitle, TString hXaxisTitle, TString hYaxisTitle,
                      Style_t markerStyle, Color_t markerColor, Double_t markerSize,
                      Color_t lineColor, Int_t lineWidth, Float_t hTitleXaxisOffset = 1.3, Float_t hTitleYaxisOffset = 1.3,
                      Float_t hTitleXaxisSize = 0.045, Float_t hTitleYaxisSize = 0.045, Float_t hLabelXaxisSize = 0.045, Float_t hLabelYaxisSize = 0.045,
                      Bool_t centerXaxisTitle = false, Bool_t centerYaxisTitle = false)
{

  histo->SetTitle(hTitle.Data());
  histo->GetXaxis()->SetTitle(hXaxisTitle.Data());
  histo->GetYaxis()->SetTitle(hYaxisTitle.Data());
  histo->SetMarkerStyle(markerStyle);
  histo->SetMarkerColor(markerColor);
  histo->SetMarkerSize(markerSize);
  histo->SetLineColor(lineColor);
  histo->SetLineWidth(lineWidth);
  histo->GetXaxis()->SetTitleOffset(hTitleXaxisOffset);
  histo->GetYaxis()->SetTitleOffset(hTitleYaxisOffset);
  histo->GetXaxis()->SetTitleSize(hTitleXaxisSize);
  histo->GetYaxis()->SetTitleSize(hTitleYaxisSize);
  histo->GetXaxis()->SetLabelSize(hLabelXaxisSize);
  histo->GetYaxis()->SetLabelSize(hLabelYaxisSize);
  histo->GetXaxis()->CenterTitle(centerXaxisTitle);
  histo->GetYaxis()->CenterTitle(centerYaxisTitle);

  return;
}
