#include "CCDB/CcdbApi.h"
#include "TFile.h"
#include "TH3F.h"
#include "TROOT.h"

int loadCorrectionsToCCDB(std::string correctionsPath, int run, long long sor, long long eor, std::string ccdbDir, bool loadEse, bool loadSp, bool loadGainEq, int vn = 2) {

    gROOT->SetBatch(kTRUE);

    std::cout << "Loading corrections to CCDB for run " << run << " from file " << correctionsPath << ", SOR: " << sor << ", EOR: " << eor << std::endl;

    const string ccdbPath = "http://alice-ccdb.cern.ch";
    o2::ccdb::CcdbApi ccdb;
    map<string, string> metadata;//, metadataRCT, header; // NOTE: Re-enable the other two if timing information is needed.
    ccdb.init(Form("%s", ccdbPath.data()));

    if (loadSp) {
        std::cout << "Loading SP corrections from file " << Form("%s/runs/%d/SP/qvecCorrs/qvec_corrections.root", correctionsPath.data(), run) << std::endl;
        TFile* inFile = new TFile(Form("%s/runs/%d/SP/qvecCorrs/qvec_corrections.root", correctionsPath.data(), run), "read");
        TH3F* hCCDB = (TH3F*)inFile->Get("ccdb");
        if(vn==2){
            metadata["runnum"] = std::to_string(run);
            metadata["harmonics"] = "v2";
            std::cout << "\nWill load histogram to ccdb path " << Form("%s/v2", ccdbDir.data()) << ", sor: " << sor << ", eor: " << eor;
            ccdb.storeAsTFileAny(hCCDB, Form("%s/v2", ccdbDir.data()), metadata, sor, eor);
        }

        if(vn==3){
            metadata["runnum"] = std::to_string(run);
            metadata["harmonics"] = "v3";
            std::cout << "\nWill load histogram to ccdb path " << Form("%s/v3", ccdbDir.data()) << ", sor: " << sor << ", eor: " << eor;
            ccdb.storeAsTFileAny(hCCDB, Form("%s/v3", ccdbDir.data()), metadata, sor, eor);
        }

        if(vn==4){
            metadata["runnum"] = std::to_string(run);
            metadata["harmonics"] = "v4";
            std::cout << "\nWill load histogram to ccdb path " << Form("%s/v4", ccdbDir.data()) << ", sor: " << sor << ", eor: " << eor;
            ccdb.storeAsTFileAny(hCCDB, Form("%s/v4", ccdbDir.data()), metadata, sor, eor);
        }
        metadata.clear();
        inFile->Close();
    }

    if (loadEse) {
        TFile* inFile = new TFile(Form("%s/runs/%d/EsE/qvecCorrs/qvec_corrections.root", correctionsPath.data(), run), "read");
        TH3F* hCCDB = (TH3F*)inFile->Get("ccdb");
        if(vn == 2){
            metadata["runnum"] = std::to_string(run);
            metadata["harmonics"] = "eseq2";
            std::cout << "\nWill load histogram to ccdb path " << Form("%s/eseq2", ccdbDir.data()) << ", sor: " << sor << ", eor: " << eor;
            ccdb.storeAsTFileAny(hCCDB, Form("%s/eseq2", ccdbDir.data()), metadata, sor, eor);
        }

        if(vn == 3){
            metadata["runnum"] = std::to_string(run);
            metadata["harmonics"] = "eseq3";
            std::cout << "\nWill load histogram to ccdb path " << Form("%s/eseq3", ccdbDir.data()) << ", sor: " << sor << ", eor: " << eor;
            ccdb.storeAsTFileAny(hCCDB, Form("%s/eseq3", ccdbDir.data()), metadata, sor, eor);
        }

        if(vn == 4){
            metadata["runnum"] = std::to_string(run);
            metadata["harmonics"] = "eseq4";
            std::cout << "\nWill load histogram to ccdb path " << Form("%s/eseq4", ccdbDir.data()) << ", sor: " << sor << ", eor: " << eor;
            ccdb.storeAsTFileAny(hCCDB, Form("%s/eseq4", ccdbDir.data()), metadata, sor, eor);
        }
        metadata.clear();
        inFile->Close();
    }

    if (loadGainEq) {
        std::cout << "Loading gain equalization constants not yet implemented!" << std::endl;
        // GAIN EQUALIZATION -- TODO
        // for (int i=0; i<nrun; i++) {
        //     std::vector<double> corrConst;
        //     std::string inFile = Form("%s/AnalysisResults_%d.root", inFileDir.data(), runnums[i]);
        //     calibGainRun("", inFileDir, 0, corrConst, runnums[i], false, "", dataset);
    
        //     metadata["runnum"] = std::to_string(runnums[i]);
        //     metadata["detector"] = "FT0";
    
        //     ULong64_t sor = sors[i];
        //     ULong64_t eor = eors[i];
            // ccdb.storeAsTFileAny(&corrConst, Form("%s/%s", ccdbDir.data(), "FT0"), metadata, sor, eor);
    
        //     metadata.clear();
        // }
    }

    return 0;
}
