#include "CCDB/CcdbApi.h"
#include "TDirectoryFile.h"
#include "TFile.h"
#include "TH2D.h"
#include "TH3F.h"
#include "TMath.h"
#include "runinfo_test.h"

enum Detectors {
  kFT0C = 0,
  kFT0A,
  kFT0M,
  kFV0A,
  kTPCPOS,
  kTPCNEG,
  kTPCALL,
  kNDetectors
};

struct DetectorConfig {
    std::string name;
    std::string file;
    std::string ref;
};

using DetectorMap = std::array<DetectorConfig, kNDetectors>;

void Recenter(TH2* h, std::vector<double>& corr){
    corr.push_back(h->GetMean(1));
    corr.push_back(h->GetMean(2));
}

double CalcB(double rho, double sigmax, double sigmay){
    return rho * sigmax * sigmay * TMath::Sqrt(2.0 * (sigmax * sigmax + sigmay * sigmay - 2.0 * sigmax * sigmay * TMath::Sqrt(1.0 - rho * rho)) / ((sigmax * sigmax - sigmay * sigmay) * (sigmax * sigmax - sigmay * sigmay) + 4.0 * (sigmax * sigmay * rho) * (sigmax * sigmay * rho)));
}

void Twist(TH2* h, std::vector<double>& corr){
    double aPlus, aMinus;
    double lambdaPlus, lambdaMinus;
    double b = CalcB(h->GetCorrelationFactor(), h->GetStdDev(1), h->GetStdDev(2));

    aPlus = TMath::Sqrt(2. * TMath::Power(h->GetStdDev(1), 2.) - TMath::Power(b, 2.));
    aMinus = TMath::Sqrt(2. * TMath::Power(h->GetStdDev(2), 2.) - TMath::Power(b, 2.));

    lambdaPlus = b / aPlus;
    lambdaMinus = b / aMinus;

    corr.push_back(lambdaPlus);
    corr.push_back(lambdaMinus);
}

void Rescale(TH2* h, std::vector<double>& corr){
    double aPlus, aMinus;
    double b = CalcB(h->GetCorrelationFactor(), h->GetStdDev(1), h->GetStdDev(2));
    aPlus = TMath::Sqrt(2. * TMath::Power(h->GetStdDev(1), 2.) - TMath::Power(b, 2.));
    aMinus = TMath::Sqrt(2. * TMath::Power(h->GetStdDev(2), 2.) - TMath::Power(b, 2.));

    corr.push_back(aPlus);
    corr.push_back(aMinus);
}

TH2F* ProjQxQyCentDiff(TH3F* hQxQyCentUncor, float zmin, float zmax, TFile* c){

    TH2F* hQxQyCentDiff = new TH2F(Form("hQxQyCent_%f_%f_Diff", zmin, zmax), "",
                                   hQxQyCentUncor->GetNbinsX(),
                                   hQxQyCentUncor->GetXaxis()->GetBinCenter(1) - hQxQyCentUncor->GetXaxis()->GetBinWidth(1)/2.,
                                   hQxQyCentUncor->GetXaxis()->GetBinCenter(hQxQyCentUncor->GetNbinsX()) +
                                                                               hQxQyCentUncor->GetXaxis()->GetBinWidth(hQxQyCentUncor->GetNbinsX())/2.,
                                   hQxQyCentUncor->GetNbinsY(),
                                   hQxQyCentUncor->GetYaxis()->GetBinCenter(1) - hQxQyCentUncor->GetYaxis()->GetBinWidth(1)/2.,
                                   hQxQyCentUncor->GetYaxis()->GetBinCenter(hQxQyCentUncor->GetNbinsY()) +
                                                                               hQxQyCentUncor->GetYaxis()->GetBinWidth(hQxQyCentUncor->GetNbinsY())/2.);

    // In principle not necessary
    for(int i=0;i<hQxQyCentUncor->GetNbinsX();i++){
        for(int j=0;j<hQxQyCentUncor->GetNbinsY();j++){
            hQxQyCentDiff->SetBinContent(i+1, j+1, 0.0);
        }
    }

    for(int k=hQxQyCentUncor->GetZaxis()->FindBin(zmin);k<hQxQyCentUncor->GetZaxis()->FindBin(zmax);k++){
        for(int i=0;i<hQxQyCentUncor->GetNbinsX();i++){
            for(int j=0;j<hQxQyCentUncor->GetNbinsY();j++){
                hQxQyCentDiff->SetBinContent(i+1, j+1, hQxQyCentDiff->GetBinContent(i+1, j+1) +
                                                hQxQyCentUncor->GetBinContent(i+1, j+1, k));
            }
        }
    }

    hQxQyCentDiff->Write();

    return hQxQyCentDiff;
}


std::vector<double> fillCorrections(string outdir, string detector, string fname, string dirname, string ref, int nmode){
    TFile* fin = new TFile(fname.data(), "read");
    TH3F* hQxQyCentUncor = (TH3F*)fin->Get(Form("%s/histQvec%sUncorV%d",dirname.c_str(),ref.c_str(),nmode));

    std::cout << "Creating file "
              << Form("%s/debugQvecs_%s.root", outdir.c_str(), detector.c_str())
              << " for debugging purposes..." << std::endl;
    TFile* c = new TFile(Form("%s/debugQvecs_%s.root", outdir.c_str(), detector.c_str()), "recreate");
    const int nCentBins = 100;      // 1% centrality differential
    TH2F* hQvecUncor[nCentBins];
    std::vector<double> CorUncor;
    for(int i=0;i<nCentBins;i++){
        hQvecUncor[i] = (TH2F*)ProjQxQyCentDiff(hQxQyCentUncor, i, i+1, c);
        Recenter(hQvecUncor[i], CorUncor);
        Twist(hQvecUncor[i], CorUncor);
        Rescale(hQvecUncor[i], CorUncor);
    }
    std::cout << "[ref = \"" << ref << "\", nmode = " << nmode << "] Recenter " << CorUncor[0] << ", " << CorUncor[1] << std::endl;
    std::cout << "[ref = \"" << ref << "\", nmode = " << nmode << "] Twist " << CorUncor[2] << ", " << CorUncor[3] << std::endl;
    std::cout << "[ref = \"" << ref << "\", nmode = " << nmode << "] Rescale " << CorUncor[4] << ", " << CorUncor[5] << std::endl;
    std::cout << std::endl;
    delete fin;
    c->Close();
    return CorUncor;
}


void makeHist(std::string outDir, const DetectorMap& detMap, int runId, string ext1Id, string ext2Id, ULong64_t sor, ULong64_t eor, int vn, TFile* corrFile){

    int nCentBins = 100;
    int nDetectorBins = 10;
    int nCorrectionParams = 6;


    TH3F* hCCDB = new TH3F("ccdb","",
                           nCentBins,0,nCentBins,                          // cent
                           nCorrectionParams,0-0.5,nCorrectionParams-0.5,  // const
                           nDetectorBins,0-0.5,nDetectorBins-0.5);         // det

    std::cout << "Filling corrections for FT0C ... " << std::endl;
    std::vector<double> QvecCorFT0C   = fillCorrections(outDir, detMap[kFT0C].name,   detMap[kFT0C].file,   "q-vectors-correction", detMap[kFT0C].ref,   vn);
    std::cout << "Filling corrections for FT0A ... " << std::endl;
    std::vector<double> QvecCorFT0A   = fillCorrections(outDir, detMap[kFT0A].name,   detMap[kFT0A].file,   "q-vectors-correction", detMap[kFT0A].ref,   vn);
    std::cout << "Filling corrections for FT0M ... " << std::endl;
    std::vector<double> QvecCorFT0M   = fillCorrections(outDir, detMap[kFT0M].name,   detMap[kFT0M].file,   "q-vectors-correction", detMap[kFT0M].ref,   vn);
    std::cout << "Filling corrections for FV0A ... " << std::endl;
    std::vector<double> QvecCorFV0A   = fillCorrections(outDir, detMap[kFV0A].name,   detMap[kFV0A].file,   "q-vectors-correction", detMap[kFV0A].ref,   vn);
    std::cout << "Filling corrections for TPCPOS ... " << std::endl;
    std::vector<double> QvecCorTPCPOS = fillCorrections(outDir, detMap[kTPCPOS].name, detMap[kTPCPOS].file, "q-vectors-correction", detMap[kTPCPOS].ref, vn);
    std::cout << "Filling corrections for TPCNEG ... " << std::endl;
    std::vector<double> QvecCorTPCNEG = fillCorrections(outDir, detMap[kTPCNEG].name, detMap[kTPCNEG].file, "q-vectors-correction", detMap[kTPCNEG].ref, vn);
    std::cout << "Filling corrections for TPCALL ... " << std::endl;
    std::vector<double> QvecCorTPCall = fillCorrections(outDir, detMap[kTPCALL].name, detMap[kTPCALL].file, "q-vectors-correction", detMap[kTPCALL].ref, vn);

    for(int i=0;i<nCentBins;i++){
        for(int j=0;j<nCorrectionParams;j++){
            hCCDB->SetBinContent(i+1, j+1, kFT0C+1, QvecCorFT0C.at(j + i*6));      // Recenter (2 pars), Twist (2 pars), Rescale (2 pars)
            hCCDB->SetBinContent(i+1, j+1, kFT0A+1, QvecCorFT0A.at(j + i*6));      // for each 1% centrality bin
            hCCDB->SetBinContent(i+1, j+1, kFT0M+1, QvecCorFT0M.at(j + i*6));
            hCCDB->SetBinContent(i+1, j+1, kFV0A+1, QvecCorFV0A.at(j + i*6));
            hCCDB->SetBinContent(i+1, j+1, kTPCPOS+1, QvecCorTPCPOS.at(j + i*6));
            hCCDB->SetBinContent(i+1, j+1, kTPCNEG+1, QvecCorTPCNEG.at(j + i*6));
            hCCDB->SetBinContent(i+1, j+1, kTPCALL+1, QvecCorTPCall.at(j + i*6));

            hCCDB->SetBinContent(i+1, j+1, 8, QvecCorTPCNEG.at(j + i*6));   // dummy
            hCCDB->SetBinContent(i+1, j+1, 9, QvecCorTPCNEG.at(j + i*6));   // dummy
            hCCDB->SetBinContent(i+1, j+1, 10, QvecCorTPCNEG.at(j + i*6));  // dummy
        }
    }

    corrFile->mkdir(Form("%d/v%d", runId, vn));
    corrFile->cd(Form("%d/v%d", runId, vn));
    hCCDB->Write();

    // const string ccdbPath = "http://alice-ccdb.cern.ch";
    // const string ccdbInternalPath = "Users/m/mdicosta/Qvector/Pass4/QvecCalib";
    // o2::ccdb::CcdbApi ccdb;
    // map<string, string> metadata;//, metadataRCT, header; // NOTE: Re-enable the other two if timing information is needed.
    // ccdb.init(Form("%s", ccdbPath.data()));
    // if(vn==2){
    //     string harmonics = "v2";
    //     metadata["runnum"] = std::to_string(runId);
    //     metadata["harmonics"] = harmonics;
    //     ccdb.storeAsTFileAny(hCCDB, Form("%s/%s", ccdbInternalPath.data(), harmonics.c_str()), metadata, sor, eor);
    // }

    // if(vn==3){
    // string harmonics = "v3";
    // metadata["runnum"] = std::to_string(runId);
    // metadata["harmonics"] = harmonics;
    // ccdb.storeAsTFileAny(hCCDB, Form("%s/%s", ccdbInternalPath.data(), harmonics.c_str()), metadata, sor, eor);
    // }

    // if(vn==4){
    //     string harmonics = "v4";
    //     metadata["runnum"] = std::to_string(runId);
    //     metadata["harmonics"] = harmonics;
    //     ccdb.storeAsTFileAny(hCCDB, Form("%s/%s", ccdbInternalPath.data(), harmonics.c_str()), metadata, sor, eor);
    // }
    // // metadata.clear();
}

void calibQVecs(std::string outDir, 
                std::string inFileFT0A,
                std::string whichDetFT0A,
                std::string inFileFT0C,
                std::string whichDetFT0C,
                std::string inFileFT0M,
                std::string whichDetFT0M,
                std::string inFileFV0A,
                std::string whichDetFV0A,
                std::string inFileTPCPOS,
                std::string whichDetTPCPOS,
                std::string inFileTPCNEG,
                std::string whichDetTPCNEG,
                std::string inFileTPCALL,
                std::string whichDetTPCALL) {

    gSystem->mkdir(Form("%s/%s/qvecCorrs/qvec_corrections.root", outDir.c_str(), runstr.data()), true);

    DetectorMap detMap = {{
        {"FT0C",   inFileFT0C,   whichDetFT0C},
        {"FT0A",   inFileFT0A,   whichDetFT0A},
        {"FT0M",   inFileFT0M,   whichDetFT0M},
        {"FV0A",   inFileFV0A,   whichDetFV0A},
        {"TPCPOS", inFileTPCPOS, whichDetTPCPOS},
        {"TPCNEG", inFileTPCNEG, whichDetTPCNEG},
        {"TPCALL", inFileTPCALL, whichDetTPCALL}
    }};

    TFile *fileCorrs = new TFile(Form("%s/%s/qvecCorrs/qvec_corrections.root", outDir.c_str(), runstr.data()), "recreate");

    std::array<int, 1> harmonics = {2}; // ,3,4};
    for (int vn : harmonics) {
        for(int i=0;i<nrun;i++){
            makeHist(Form("%s/%s/qvecCorrs/", outDir.c_str(), runstr.data()), detMap, runnums[i], Form("qvec_fit_%d",runnums[i]), Form("qvec_tpc_%d",runnums[i]), sors[i], eors[i], vn, fileCorrs);
        }
    }

    fileCorrs->Close();

}
