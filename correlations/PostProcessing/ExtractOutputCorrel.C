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
/// \usage .L DhCorrelationExtraction.cxx+
/// \usage .x ExtractOutputCorrel.C("config-file-name")
/// \author Samuele Cattaruzzi <samuele.cattaruzzi@cern.ch>
/// \author Swapnesh Santosh Khade <swapnesh.santosh.khade@cern.ch>

#include "DhCorrelationExtraction.h"
#include "Riostream.h"

#include <TROOT.h>
#include <TStyle.h>

#include <rapidjson/document.h>
#include <rapidjson/filereadstream.h>

#include <cstdio>
#include <iostream>
#include <string>
#include <vector>

using namespace rapidjson;

template <typename ValueType>
void readArray(const Value& jsonArray, std::vector<ValueType>& output)
{
  for (auto it = jsonArray.Begin(); it != jsonArray.End(); it++) {
    auto value = it->template Get<ValueType>();
    output.emplace_back(value);
  }
}

void parseStringArray(const Value& jsonArray, std::vector<std::string>& output)
{
  size_t arrayLength = jsonArray.Size();
  for (size_t i = 0; i < arrayLength; i++) {
    if (jsonArray[i].IsString()) {
      output.emplace_back(jsonArray[i].GetString());
    }
  }
}

void SetInputCorrelNames(DhCorrelationExtraction* plotter, TString pathFileSE, TString pathFileME, TString dirSE, TString dirME, TString histoNameCorrSignal, TString histoNameCorrSideba, TString histoNameCorrSidebaLeft, TString histoNameCorrSidebaRight);
void SetInputHistoInvMassNames(DhCorrelationExtraction* plotter, TString pathFileMass, std::vector<std::string> inputMassNames);
void SetInputHistoFDSubtraction(DhCorrelationExtraction* plotter, TString pathFileFDTemplate, TString pathFileFDPromptFrac, TString histoNameFDTemplatePrompt, TString histoNameFDTemplateNonPrompt, TString histoNameRawFracPrompt);
void SetInputHistoSecPart(DhCorrelationExtraction* plotter, TString pathFileSecPart, TString dirSecPartName, TString histoNamePrimaryPart, TString histoNameAllPart);
void SetInputHistoBiasBtoD(DhCorrelationExtraction* plotter, TString pathfFilePromptMcRec, TString pathfFileNonPromptMcRec);
void SetInputHistoNames_v2(DhCorrelationExtraction* plotter, TString pathFileSE, TString pathFileME, TString dirSE, TString dirME, TString histoNameCorrSE, TString histoNameCorrME);

void ExtractOutputCorrel(const TString cfgFileName = "config_CorrAnalysis.json")
{
  std::cout << "==========================" << std::endl;
  std::cout << "D-h Correlation Extraction" << std::endl;
  // gStyle -> SetOptStat(0);
  gStyle->SetPadLeftMargin(0.15);
  gStyle->SetPadBottomMargin(0.15);
  gStyle->SetFrameLineWidth(2);
  gStyle->SetLineWidth(2);
  gStyle->SetCanvasDefH(1126);
  gStyle->SetCanvasDefW(1840);

  std::cout << "Style set" << std::endl;

  // Load config
  FILE* configFile = fopen(cfgFileName.Data(), "r");
  Document config;
  char readBuffer[65536];
  FileReadStream is(configFile, readBuffer, sizeof(readBuffer));
  config.ParseStream(is);
  fclose(configFile);

  std::cout << "Config file loaded" << std::endl;

  string CodeNameAnalysis = config["CodeName"].GetString();
  gSystem->Exec(Form("rm -rf Output_CorrelationExtraction_%s_Root/ Output_CorrelationExtraction_%s_png/", CodeNameAnalysis.data(), CodeNameAnalysis.data()));
  gSystem->Exec(Form("mkdir Output_CorrelationExtraction_%s_Root/ Output_CorrelationExtraction_%s_png/", CodeNameAnalysis.data(), CodeNameAnalysis.data()));

  gSystem->Exec("rm -rf InvMass/");
  gSystem->Exec("mkdir InvMass/");

  std::cout << "Output directories created" << std::endl;

  string pathFileSE = config["pathFileSE"].GetString();
  std::cout << "pathFileSE: " << pathFileSE << std::endl;
  string pathFileME = config["pathFileME"].GetString();
  std::cout << "pathFileME: " << pathFileME << std::endl;
  string pathFileMass = config["pathFileMass"].GetString();
  std::cout << "pathFileMass: " << pathFileMass << std::endl;
  string pathFileFDTemplate = config["pathFileFDTemplate"].GetString();
  std::cout << "pathFileFDTemplate: " << pathFileFDTemplate << std::endl;
  string pathFileFDPromptFrac = config["pathFileFDPromptFrac"].GetString();
  std::cout << "pathFileFDPromptFrac: " << pathFileFDPromptFrac << std::endl;
  string pathFileSecPart = config["pathFileSecPart"].GetString();
  std::cout << "pathFileSecPart: " << pathFileSecPart << std::endl;
  string pathfFilePromptMcRec = config["pathfFilePromptMcRec"].GetString();
  std::cout << "pathfFilePromptMcRec: " << pathfFilePromptMcRec << std::endl;
  string pathfFileNonPromptMcRec = config["pathfFileNonPromptMcRec"].GetString();
  std::cout << "pathfFileNonPromptMcRec: " << pathfFileNonPromptMcRec << std::endl;

  string dirSE = config["InputDirSE"].GetString();
  std::cout << "dirSE: " << dirSE << std::endl;
  string dirME = config["InputDirME"].GetString();
  std::cout << "dirME: " << dirME << std::endl;
  string dirSecPart = config["InputDirSecPart"].GetString();
  std::cout << "dirSecPart: " << dirSecPart << std::endl;
  string histoNameCorrSE = config["InputHistoCorrSE"].GetString();
  std::cout << "histoNameCorrSE: " << histoNameCorrSE << std::endl;
  string histoNameCorrME = config["InputHistoCorrME"].GetString();
  std::cout << "histoNameCorrME: " << histoNameCorrME << std::endl;
  string histoNameCorrSignal = config["InputHistoCorrSignalName"].GetString();
  std::cout << "histoNameCorrSignal: " << histoNameCorrSignal << std::endl;
  string histoNameCorrSideba = config["InputHistoCorrSidebaName"].GetString();
  std::cout << "histoNameCorrSideba: " << histoNameCorrSideba << std::endl;
  string histoNameCorrSidebaLeft = config["InputHistoCorrSidebaLeftName"].GetString();
  std::cout << "histoNameCorrSidebaLeft: " << histoNameCorrSidebaLeft << std::endl;
  string histoNameCorrSidebaRight = config["InputHistoCorrSidebaRightName"].GetString();
  std::cout << "histoNameCorrSidebaRight: " << histoNameCorrSidebaRight << std::endl;
  string histoNameFDTemplatePrompt = config["InputHistoFDTemplatePrompt"].GetString();
  std::cout << "histoNameFDTemplatePrompt: " << histoNameFDTemplatePrompt << std::endl;
  string histoNameFDTemplateNonPrompt = config["InputHistoFDTemplateNonPrompt"].GetString();
  std::cout << "histoNameFDTemplateNonPrompt: " << histoNameFDTemplateNonPrompt << std::endl;
  string histoNameRawFracPrompt = config["InputHistoFDPromptFrac"].GetString();
  std::cout << "histoNameRawFracPrompt: " << histoNameRawFracPrompt << std::endl;
  string histoNamePrimaryPart = config["InputHistoPrimaryPart"].GetString();
  std::cout << "histoNamePrimaryPart: " << histoNamePrimaryPart << std::endl;
  string histoNameAllPart = config["InputHistoAllPart"].GetString();
  std::cout << "histoNameAllPart: " << histoNameAllPart << std::endl;

  std::cout << "Input paths and histogram names retrieved from config" << std::endl;

  std::vector<std::string> InputHistoMassName;

  const Value& inputMassNames = config["InputHistoMassName"];
  parseStringArray(inputMassNames, InputHistoMassName);

  std::cout << InputHistoMassName[0].data() << std::endl;
  std::cout << InputHistoMassName[1].data() << std::endl;
  std::cout << InputHistoMassName[2].data() << std::endl;

  std::vector<double> binsInvMassIntervals;
  std::vector<double> binsPtCandIntervals;
  std::vector<double> binsPtHadIntervals;
  std::vector<double> deltaEtaInterval;

  const Value& InvMassValue = config["binsInvMassIntervals"];
  readArray(InvMassValue, binsInvMassIntervals);

  const Value& PtCandValue = config["binsPtCandIntervals"];
  readArray(PtCandValue, binsPtCandIntervals);

  const Value& PtHadValue = config["binsPtHadIntervals"];
  readArray(PtHadValue, binsPtHadIntervals);

  const Value& deltaEtaValue = config["deltaEtaInterval"];
  readArray(deltaEtaValue, deltaEtaInterval);
  double deltaEtaMin = deltaEtaInterval[0];
  double deltaEtaMax = deltaEtaInterval[1];

  int specie = config["DmesonSpecie"].GetInt();
  bool rebinAngCorr = config["RebinAngCorr"].GetBool();
  bool rebinFDCorr = config["RebinFDCorr"].GetBool();
  bool rebinSecPart = config["RebinSecPart"].GetBool();
  int rebinDeltaPhi = config["nRebinDeltaPhi"].GetInt();
  int rebinDeltaEta = config["nRebinDeltaEta"].GetInt();
  double valPhiMEnorm = config["ValPhiMEnorm"].GetDouble();
  double valEtaMEnorm = config["ValEtaMEnorm"].GetDouble();

  int npools = config["NumberOfPools"].GetInt();
  bool poolByPool = config["CorrectPoolsSeparately"].GetBool();
  bool applySecPartCorr = config["ApplySecPartCorr"].GetBool();
  bool applyBiasBtoDCorr = config["ApplyBiasBtoDCorr"].GetBool();
  bool applyFDCorr = config["ApplyFDCorr"].GetBool();
  bool isDividedSideb = config["IsDividedSideb"].GetBool();
  bool useSidebLeft = config["UseSidebLeft"].GetBool();
  bool useSidebRight = config["UseSidebRight"].GetBool();

  if (useSidebLeft && useSidebLeft) {
    std::cout << "Using left and right" << std::endl;
  }

  std::cout << "=========================== " << std::endl;
  std::cout << "Input variables from config" << std::endl;
  std::cout << "deltaEtaMin    = " << deltaEtaMin << std::endl;
  std::cout << "deltaEtaMax    = " << deltaEtaMax << std::endl;
  std::cout << "DmesonSpecie    = " << specie << std::endl;
  std::cout << "nPools    = " << npools << std::endl;
  std::cout << "poolByPool    = " << poolByPool << std::endl;
  std::cout << "=========================== " << std::endl;
  std::cout << " " << std::endl;

  const int nBinsInvMass = binsInvMassIntervals.size() - 1;
  const int nBinsPtCand = binsPtCandIntervals.size() - 1;
  const int nBinsPtHad = binsPtHadIntervals.size() - 1;

  std::vector<std::vector<std::vector<TH2D*>>> hCorrel_SE(nBinsPtCand, std::vector<std::vector<TH2D*>>(nBinsPtHad, std::vector<TH2D*>(nBinsInvMass, nullptr)));
  std::vector<std::vector<std::vector<TH2D*>>> hCorrel_ME(nBinsPtCand, std::vector<std::vector<TH2D*>>(nBinsPtHad, std::vector<TH2D*>(nBinsInvMass, nullptr)));
  std::vector<std::vector<std::vector<TH2D*>>> hCorrectedCorrel_2D(nBinsPtCand, std::vector<std::vector<TH2D*>>(nBinsPtHad, std::vector<TH2D*>(nBinsInvMass, nullptr)));
  std::vector<std::vector<std::vector<TH1D*>>> hCorrectedCorrel(nBinsPtCand, std::vector<std::vector<TH1D*>>(nBinsPtHad, std::vector<TH1D*>(nBinsInvMass, nullptr)));
  std::vector<std::vector<std::vector<TH1D*>>> hCorrectedCorrel_BaselineSubtr(nBinsPtCand, std::vector<std::vector<TH1D*>>(nBinsPtHad, std::vector<TH1D*>(nBinsInvMass, nullptr)));
  std::vector<std::vector<std::vector<TH1D*>>> hCorrectedCorrel_Reflected(nBinsPtCand, std::vector<std::vector<TH1D*>>(nBinsPtHad, std::vector<TH1D*>(nBinsInvMass, nullptr)));
  std::vector<std::vector<std::vector<TH1D*>>> hCorrectedCorrel_Reflected_BaselineSubtr(nBinsPtCand, std::vector<std::vector<TH1D*>>(nBinsPtHad, std::vector<TH1D*>(nBinsInvMass, nullptr)));

  TH2F* hMassVsPt;
  // TH2D* hCorrel_SE[nBinsPtCand][nBinsPtHad][nBinsInvMass];
  // TH2D* hCorrel_ME[nBinsPtCand][nBinsPtHad][nBinsInvMass];
  // TH2D* hCorrectedCorrel_2D[nBinsPtCand][nBinsPtHad][nBinsInvMass];
  // TH1D* hCorrectedCorrel[nBinsPtCand][nBinsPtHad][nBinsInvMass];
  // TH1D* hCorrectedCorrel_BaselineSubtr[nBinsPtCand][nBinsPtHad][nBinsInvMass];
  // TH1D* hCorrectedCorrel_Reflected[nBinsPtCand][nBinsPtHad][nBinsInvMass];
  // TH1D* hCorrectedCorrel_Reflected_BaselineSubtr[nBinsPtCand][nBinsPtHad][nBinsInvMass];

  // Create and set the correlation plotter class
  DhCorrelationExtraction* plotter = new DhCorrelationExtraction();

  Bool_t flagSpecie = plotter->SetDmesonSpecie(static_cast<DhCorrelationExtraction::DmesonSpecie>(specie));
  plotter->SetNpools(npools);
  plotter->SetCorrectPoolsSeparately(poolByPool); // kTRUE = pool.by-pool extraction and correction; kFALSE = merged ME pools
  plotter->SetFDSubtraction(applyFDCorr);
  plotter->SetSecPartContamination(applySecPartCorr);
  plotter->SetDeltaEtaRange(deltaEtaMin, deltaEtaMax);
  plotter->SetSubtractSoftPiInMEdistr(kFALSE);
  plotter->SetRebinOptions(rebinAngCorr, rebinFDCorr, rebinSecPart);
  plotter->SetRebin2DcorrelHisto(rebinDeltaEta, rebinDeltaPhi); // Xaxis: deltaEta, Yaxis: deltaPhi
  plotter->SetBinDeltaPhiEtaForMEnorm(valPhiMEnorm, valEtaMEnorm);
  plotter->SetCorrBiasBtoD(applyBiasBtoDCorr);
  plotter->SetDebugLevel(1);

  if (!flagSpecie)
    std::cout << "[ERROR] Wrong D meson flag" << std::endl;

  // Set the input file config
  SetInputCorrelNames(plotter, pathFileSE, pathFileME, dirSE, dirME, histoNameCorrSignal, histoNameCorrSideba, histoNameCorrSidebaLeft, histoNameCorrSidebaRight);
  SetInputHistoNames_v2(plotter, pathFileSE, pathFileME, dirSE, dirME, histoNameCorrSE, histoNameCorrME);
  SetInputHistoInvMassNames(plotter, pathFileMass, InputHistoMassName);
  if (applyFDCorr)
    SetInputHistoFDSubtraction(plotter, pathFileFDTemplate, pathFileFDPromptFrac, histoNameFDTemplatePrompt, histoNameFDTemplateNonPrompt, histoNameRawFracPrompt);
  if (applySecPartCorr)
    SetInputHistoSecPart(plotter, pathFileSecPart, dirSecPart, histoNamePrimaryPart, histoNameAllPart);
  if (applyBiasBtoDCorr)
    SetInputHistoBiasBtoD(plotter, pathfFilePromptMcRec, pathfFileNonPromptMcRec);
  Bool_t readSEandME = plotter->ReadInputSEandME();
  if (readSEandME)
    std::cout << "Files SE and ME read correctly" << std::endl;
  Bool_t readInvMass = plotter->ReadInputInvMass();
  if (readInvMass)
    std::cout << "Files inv. mass read correctly" << std::endl;
  if (applyFDCorr) {
    Bool_t readFDSubtr = plotter->ReadInputFDSubtr();
    if (readFDSubtr)
      std::cout << "Files for FD subtr. read correctly" << std::endl;
  }
  if (applySecPartCorr) {
    Bool_t readSecPart = plotter->ReadInputSecondaryPartContamination();
    if (readSecPart)
      std::cout << "Files for secondary part. contamination read correctly" << std::endl;
  }

  std::cout << "Start loop inv. mass" << std::endl;
  // Loop over candidate pt and assoc. particle pt
  for (int iBinInvMass = 0; iBinInvMass < nBinsInvMass; iBinInvMass++) {
    std::cout << "[INFO] InvMass: " << binsInvMassIntervals[iBinInvMass] << " - "<< binsInvMassIntervals[iBinInvMass+1] << std::endl;
    for (int iBinPtCand = 0; iBinPtCand < nBinsPtCand; iBinPtCand++) {
      plotter->SetDividedSidebands(isDividedSideb, useSidebLeft, useSidebRight);
      //plotter->GetSignalAndBackgroundForNorm(binsPtCandIntervals[iBinPtCand], binsPtCandIntervals[iBinPtCand + 1]);
      plotter->GetSignalAndBackgroundForNorm_v2(binsPtCandIntervals[iBinPtCand], binsPtCandIntervals[iBinPtCand + 1], binsInvMassIntervals[iBinInvMass], binsInvMassIntervals[iBinInvMass + 1]);
      for (int iBinPtHad = 0; iBinPtHad < nBinsPtHad; iBinPtHad++) {
        plotter->SetBinCandAndHad(iBinPtCand + 1, iBinPtHad + 1, iBinInvMass + 1);
        plotter->ExtractCorrelations(binsPtCandIntervals[iBinPtCand], binsPtCandIntervals[iBinPtCand + 1], binsPtHadIntervals[iBinPtHad], binsPtHadIntervals[iBinPtHad + 1], binsInvMassIntervals[iBinInvMass], binsInvMassIntervals[iBinInvMass + 1], CodeNameAnalysis);
        hCorrel_SE[iBinPtCand][iBinPtHad][iBinInvMass] = (TH2D*)plotter->GetCorrHisto2D_SE();
        hCorrel_ME[iBinPtCand][iBinPtHad][iBinInvMass] = (TH2D*)plotter->GetCorrHisto2D_ME();
        if (iBinInvMass == 0 && iBinPtCand == 0 && iBinPtHad == 0) hMassVsPt = (TH2F*)plotter->GetInvMassVsPtHisto();
        hCorrectedCorrel_2D[iBinPtCand][iBinPtHad][iBinInvMass] = (TH2D*)plotter->GetCorrectedCorrHisto2D();
        hCorrectedCorrel[iBinPtCand][iBinPtHad][iBinInvMass] = (TH1D*)plotter->GetCorrectedCorrHisto();
        hCorrectedCorrel_BaselineSubtr[iBinPtCand][iBinPtHad][iBinInvMass] = (TH1D*)plotter->GetCorrectedCorrHisto_BaselineSubtr();
        hCorrectedCorrel_Reflected[iBinPtCand][iBinPtHad][iBinInvMass] = (TH1D*)plotter->GetCorrectedCorrHisto_Reflected();
        hCorrectedCorrel_Reflected_BaselineSubtr[iBinPtCand][iBinPtHad][iBinInvMass] = (TH1D*)plotter->GetCorrectedCorrHisto_Reflected_BaselineSubtr();
      }
    }
  }

  // output file inv mass vs pt
  string shortName = CodeNameAnalysis.substr(0, 4); // first 3 characters from index 0 to index 3
  TFile* outFileMass = new TFile(Form("InvMass/InvMassVsPt_%s.root", shortName.data()), "RECREATE");
  outFileMass->cd();
  hMassVsPt->Write();
  outFileMass->Close();

  // output file original 2D distributions
  TFile* outFileOriginal = new TFile(Form("Output_CorrelationExtraction_%s_Root/ExtractCorrelationsResults_2D.root", CodeNameAnalysis.data()), "RECREATE");
  outFileOriginal->cd();
  for (int iBinInvMass = 0; iBinInvMass < nBinsInvMass; iBinInvMass++) {
    for (int iBinPtCand = 0; iBinPtCand < nBinsPtCand; iBinPtCand++) {
      for (int iBinPtHad = 0; iBinPtHad < nBinsPtHad; iBinPtHad++) {
        hCorrel_SE[iBinPtCand][iBinPtHad][iBinInvMass]->Write();
        hCorrel_ME[iBinPtCand][iBinPtHad][iBinInvMass]->Write();
        hCorrectedCorrel_2D[iBinPtCand][iBinPtHad][iBinInvMass]->Write();
      }
    }
  }
  outFileOriginal->Close();

  // output file
  TFile* outFile = new TFile(Form("Output_CorrelationExtraction_%s_Root/ExtractCorrelationsResults.root", CodeNameAnalysis.data()), "RECREATE");
  outFile->cd();
  for (int iBinInvMass = 0; iBinInvMass < nBinsInvMass; iBinInvMass++) {
    for (int iBinPtCand = 0; iBinPtCand < nBinsPtCand; iBinPtCand++) {
      for (int iBinPtHad = 0; iBinPtHad < nBinsPtHad; iBinPtHad++) {
        hCorrectedCorrel[iBinPtCand][iBinPtHad][iBinInvMass]->Write();
      }
    }
  }
  outFile->Close();

  // output file baseline subtr.
  TFile* outFile_BaselineSubtr = new TFile(Form("Output_CorrelationExtraction_%s_Root/ExtractCorrelationsResults_BaselineSubtr.root", CodeNameAnalysis.data()), "RECREATE");
  outFile_BaselineSubtr->cd();
  for (int iBinInvMass = 0; iBinInvMass < nBinsInvMass; iBinInvMass++) {
    for (int iBinPtCand = 0; iBinPtCand < nBinsPtCand; iBinPtCand++) {
      for (int iBinPtHad = 0; iBinPtHad < nBinsPtHad; iBinPtHad++) {
        hCorrectedCorrel_BaselineSubtr[iBinPtCand][iBinPtHad][iBinInvMass]->Write();
      }
    }
  }
  outFile_BaselineSubtr->Close();

  // output file reflected
  TFile* outFile_Reflected = new TFile(Form("Output_CorrelationExtraction_%s_Root/ExtractCorrelationsResults_Reflected.root", CodeNameAnalysis.data()), "RECREATE");
  outFile_Reflected->cd();
  for (int iBinInvMass = 0; iBinInvMass < nBinsInvMass; iBinInvMass++) {
    for (int iBinPtCand = 0; iBinPtCand < nBinsPtCand; iBinPtCand++) {
      for (int iBinPtHad = 0; iBinPtHad < nBinsPtHad; iBinPtHad++) {
        hCorrectedCorrel_Reflected[iBinPtCand][iBinPtHad][iBinInvMass]->Write();
      }
    }
  }
  outFile_Reflected->Close();

  // output file reflected baseline subtr.
  TFile* outFile_Reflected_BaselineSubtr = new TFile(Form("Output_CorrelationExtraction_%s_Root/ExtractCorrelationsResults_Reflected_BaselineSubtr.root", CodeNameAnalysis.data()), "RECREATE");
  outFile_Reflected_BaselineSubtr->cd();
  for (int iBinInvMass = 0; iBinInvMass < nBinsInvMass; iBinInvMass++) {
    for (int iBinPtCand = 0; iBinPtCand < nBinsPtCand; iBinPtCand++) {
      for (int iBinPtHad = 0; iBinPtHad < nBinsPtHad; iBinPtHad++) {
        hCorrectedCorrel_Reflected_BaselineSubtr[iBinPtCand][iBinPtHad][iBinInvMass]->Write();
      }
    }
  }
  outFile_Reflected_BaselineSubtr->Close();

  return;
}

void SetInputHistoNames_v2(DhCorrelationExtraction* plotter, TString pathFileSE, TString pathFileME, TString dirSE, TString dirME, TString histoNameCorrSE, TString histoNameCorrME)
{
  plotter->SetInputFilenameSE(pathFileSE.Data());
  plotter->SetInputFilenameME(pathFileME.Data());
  plotter->SetDirNameSE(dirSE.Data());
  plotter->SetDirNameME(dirME.Data());
  plotter->SetCorrelHistoSE(histoNameCorrSE.Data());
  plotter->SetCorrelHistoME(histoNameCorrME.Data());

  return;
}

void SetInputCorrelNames(DhCorrelationExtraction* plotter, TString pathFileSE, TString pathFileME, TString dirSE, TString dirME, TString histoNameCorrSignal, TString histoNameCorrSideba, TString histoNameCorrSidebaLeft, TString histoNameCorrSidebaRight)
{

  // Ds paths
  plotter->SetInputFilenameSE(pathFileSE.Data());
  plotter->SetInputFilenameME(pathFileME.Data());
  plotter->SetDirNameSE(dirSE.Data());
  plotter->SetDirNameME(dirME.Data());
  plotter->SetSECorrelHistoSignalName(histoNameCorrSignal.Data());
  plotter->SetSECorrelHistoSidebandName(histoNameCorrSideba.Data());
  plotter->SetMECorrelHistoSignalName(histoNameCorrSignal.Data());
  plotter->SetMECorrelHistoSidebandName(histoNameCorrSideba.Data());
  plotter->SetSECorrelHistoSidebandLeftName(histoNameCorrSidebaLeft.Data());
  plotter->SetMECorrelHistoSidebandLeftName(histoNameCorrSidebaLeft.Data());
  plotter->SetSECorrelHistoSidebandRightName(histoNameCorrSidebaRight.Data());
  plotter->SetMECorrelHistoSidebandRightName(histoNameCorrSidebaRight.Data());

  return;
}

void SetInputHistoInvMassNames(DhCorrelationExtraction* plotter, TString pathFileMass, std::vector<std::string> inputMassNames)
{ // to use if sgn and bkg extraction is done apart

  plotter->SetInputFilenameMass(pathFileMass.Data());
  plotter->SetMassHistoNameSgn(inputMassNames[0].data());
  plotter->SetMassHistoNameBkg(inputMassNames[1].data());
  plotter->SetMassHistoNameSBs(inputMassNames[2].data());

  return;
}

void SetInputHistoFDSubtraction(DhCorrelationExtraction* plotter, TString pathFileFDTemplate, TString pathFileFDPromptFrac, TString histoNameFDTemplatePrompt, TString histoNameFDTemplateNonPrompt, TString histoNameRawFracPrompt)
{

  plotter->SetInputFilenameFDTemplate(pathFileFDTemplate.Data());
  plotter->SetInputFilenameFDPromptFrac(pathFileFDPromptFrac.Data());
  plotter->SetInputHistoNameFDTemplatePrompt(histoNameFDTemplatePrompt.Data());
  plotter->SetInputHistoNameFDTemplateNonPrompt(histoNameFDTemplateNonPrompt.Data());
  plotter->SetInputHistoNameFDPromptFrac(histoNameRawFracPrompt.Data());

  return;
}

void SetInputHistoSecPart(DhCorrelationExtraction* plotter, TString pathFileSecPart, TString dirSecPartName, TString histoNamePrimaryPart, TString histoNameAllPart)
{

  plotter->SetInputFilenameSecPart(pathFileSecPart.Data());
  plotter->SetDirNameSecPart(dirSecPartName.Data());
  plotter->SetHistoSecPartName(histoNamePrimaryPart.Data(), histoNameAllPart.Data());

  return;
}

void SetInputHistoBiasBtoD(DhCorrelationExtraction* plotter, TString pathfFilePromptMcRec, TString pathfFileNonPromptMcRec)
{

  plotter->SetInputFilenameBiasBtoD(pathfFilePromptMcRec.Data(), pathfFileNonPromptMcRec.Data());

  return;
}
