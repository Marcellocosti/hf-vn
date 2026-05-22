#include "CCDB/CcdbApi.h"
#include "TDirectoryFile.h"
#include "TFile.h"
#include "TH2D.h"
#include "TMath.h"
#include "TColor.h"
#include "TCanvas.h"
#include "TLegend.h"
#include "TSystem.h"
#include "TGraph.h"
#include "TPad.h"
#include "TF1.h"
#include "TLine.h"
#include "TStyle.h"
#include "runinfo_test.h"

using std::map;
using std::string;

const int ndraw = 8;
int RainbowColor[ndraw];
const int NRGBs = 5;
double stops[NRGBs] = {0.00, 0.34, 0.61, 0.84, 1.00};
double red[NRGBs] = {0.00, 0.00, 0.87, 0.9 * 1.00, 0.51};
double green[NRGBs] = {0.00, 0.81, 0.9 * 1.00, 0.20, 0.00};
double blue[NRGBs] = {0.51, 0.9 * 1.00, 0.12, 0.00, 0.00};
void initColors() {
    int sysColorPallet = TColor::CreateGradientColorTable(NRGBs, stops, red, green, blue, ndraw);
    for (int i=0; i<ndraw; i++) {
        RainbowColor[i] = sysColorPallet + ndraw - i - 1;
    }
}

void calcRelGainFT0(std::vector<double>& RelGain, std::vector<double>& mean, std::vector<TH1D*>& hAmpProj, TH2F* hFIT, int nCh, TGraph* gMean, TGraph* gMeanScaled) {
    TH1D* hTotalAmpFT0C;
    TH1D* hTotalAmpFT0A;

    for (int i=0; i<nCh; i++) {
        hAmpProj[i] = (TH1D*)hFIT->ProjectionX(Form("hAmpProj_%d",i), i+1, i+1);
        if (i==0) {
            hTotalAmpFT0C = (TH1D*)hAmpProj[i]->Clone("hTotalAmpFT0C");
            hTotalAmpFT0A = (TH1D*)hAmpProj[i]->Clone("hTotalAmpFT0A");
            hTotalAmpFT0C->Reset();
            hTotalAmpFT0A->Reset();
        }
        if (i < 96) {
            hTotalAmpFT0A->Add(hAmpProj[i], 1.0);
        } else{
            hTotalAmpFT0C->Add(hAmpProj[i], 1.0);
        }

        mean[i] = hAmpProj[i]->GetMean();
        gMean->SetPoint(i, mean[i], (double)i + 0.5);
        hAmpProj[i]->SetLineColor(RainbowColor[i % ndraw]);

        hAmpProj[i]->GetXaxis()->SetRangeUser(0, 5000);
        hAmpProj[i]->GetYaxis()->SetTitle("Counts");
        hAmpProj[i]->SetLineWidth(2);
    }

    for (int i=0; i<nCh; i++) {
        RelGain[i] = mean[i];
        if (i < 96) {
            RelGain[i] /= hTotalAmpFT0A->GetMean();
        } else{
            RelGain[i] /= hTotalAmpFT0C->GetMean();
        }
    }
}

void calcRelGainFV0 (std::vector<double>& RelGain, std::vector<double>& mean, std::vector<TH1D*>& hAmpProj, TH2F* hFIT, int nCh, TGraph* gMean, TGraph* gMeanScaled) {
    const int nFV0Rings = 5;
    TH1D* hTotalAmp[nFV0Rings];

    for (int i=0;i<nCh;i++) {
        hAmpProj[i] = (TH1D*)hFIT->ProjectionX(Form("hAmpProj_%d",i), i+1, i+1);
        if (i==0) {
            for (int j=0;j<nFV0Rings;j++) {
                hTotalAmp[j] = (TH1D*)hFIT->ProjectionX(Form("hAmpProj_%d_%d",j,i),i+2,i+2);
                hTotalAmp[j]->Reset();
            }
        }

        if (i<40) {
            hTotalAmp[i/8]->Add(hAmpProj[i], 1.0);
        } else {
            hTotalAmp[4]->Add(hAmpProj[i], 1.0);
        }
        mean[i] = hAmpProj[i]->GetMean();
        gMean->SetPoint(i, mean[i], (double)i + 0.5);
        hAmpProj[i]->SetLineColor(RainbowColor[i % ndraw]);
        hAmpProj[i]->GetXaxis()->SetRangeUser(0,3000);
        hAmpProj[i]->GetYaxis()->SetTitle("Counts");
        hAmpProj[i]->SetLineWidth(2);
    }

    for (int i=0;i<nCh;i++) {
        RelGain[i] = mean[i];
        std::cout << "Set relative gain for channel " << i << " to " << RelGain[i] << " before normalization." << std::endl;
        if (i<40) {
            RelGain[i] /= hTotalAmp[i/8]->GetMean();
        } else {
            RelGain[i] /= hTotalAmp[4]->GetMean();
        }
    }
}

void calibGainRun(std::string outDir, std::string inFile, int nCh, int runId, std::vector<double>& corr, int arunId, bool draw, std::string detName, std::string dataset) {
    gSystem->mkdir(Form("%s/%s/gainCor/%s/fig_gain_%d", outDir.data(), dataset.data(), detName.data(), arunId), true);

    gStyle->SetTitleFont(43,"X");
    gStyle->SetTitleFont(43,"Y");
    gStyle->SetLabelFont(43,"X");
    gStyle->SetLabelFont(43,"Y");

    gStyle->SetTitleSize(32,"X");
    gStyle->SetTitleSize(32,"Y");
    gStyle->SetLabelSize(28,"X");
    gStyle->SetLabelSize(28,"Y");

    gStyle->SetOptStat(0);

    TCanvas* c = new TCanvas("c","c",800,700);
    gPad->SetLeftMargin(0.15);
    gPad->SetBottomMargin(0.15);
    gPad->SetRightMargin(0.03);
    gPad->SetTopMargin(0.03);
    gPad->SetTicks();
    gPad->SetLogy();

    TLegend* leg = new TLegend(0.4,0.6,0.9,0.9);
    leg->SetTextFont(43);
    leg->SetTextSize(32);
    leg->SetLineWidth(0);
    leg->SetFillStyle(0);
    leg->SetNColumns(2);
    TFile* fin = new TFile(inFile.data(), "read");
    if (!fin || fin->IsZombie()) {
        std::cout << "Cannot open file" << std::endl;
        return;
    }

    // TFile* fin = new TFile(Form("data/AnalysisResults_qvec_tpc_%d.root",runId),"read");
    // if (!gSystem->IsFileInIncludePath(Form("data/AnalysisResults_qvec_tpc_%d.root",runId))) return;

    TH2F* hFIT = (TH2F*)fin->Get(Form("q-vectors-table/%sAmp", detName.data()));
    hFIT->GetYaxis()->SetTitle("Channel ID");
    hFIT->GetXaxis()->SetTitle(Form("%s Amplitude", detName.data()));

    cout << "histogram loaded" << endl;
    std::vector<TH1D*> hAmpProj(nCh);
    std::vector<double> mean(nCh);
    std::vector<double> RelGain(nCh);
    TGraph* gMean = new TGraph();
    TGraph* gMeanScaled = new TGraph();

    std::cout << "Calculating relative gain ... " << std::endl;
    // Calculate relative gain
    if (detName == "FT0") {
        calcRelGainFT0(RelGain, mean, hAmpProj, hFIT, nCh, gMean, gMeanScaled);
    } else if (detName == "FV0") {
        calcRelGainFV0(RelGain, mean, hAmpProj, hFIT, nCh, gMean, gMeanScaled);
    }
    std::cout << "Relative gain calculated" << std::endl;

    std::vector<TH1D*> hAmpProjScaled(nCh);
    TH2F* hFITScaled = (TH2F*)hFIT->Clone("hFITScaled");
    hFITScaled->Reset();

    for (int i=0; i<nCh; i++) {
        if (RelGain[i] > 1e-4) {
            hAmpProjScaled[i] = new TH1D(Form("hAmpProjScaled_%d",i),"",hAmpProj[i]->GetNbinsX(),
                                        (hAmpProj[i]->GetBinCenter(1) - hAmpProj[i]->GetBinWidth(1)/2.) / RelGain[i],
                                        (hAmpProj[i]->GetBinCenter(hAmpProj[i]->GetNbinsX()) + hAmpProj[i]->GetBinWidth(hAmpProj[i]->GetNbinsX())/2.) / RelGain[i]);
        } else {
            hAmpProjScaled[i] = new TH1D(Form("hAmpProjScaled_%d",i),"",1,0,1);
        }
        for (int j=0; j<hAmpProjScaled[ i]->GetNbinsX();j++) {
            hAmpProjScaled[i]->SetBinContent(j+1, hAmpProj[i]->GetBinContent(j+1));
            hFITScaled->SetBinContent(hFITScaled->GetXaxis()->FindBin(hAmpProjScaled[i]->GetBinCenter(j+1)),
                                      i+1, hAmpProjScaled[i]->GetBinContent(j+1));
        }
        std::cout << "[" << detName << "] Mean of scaled amplitude for channel " << i << ": " << hAmpProjScaled[i]->GetMean() << std::endl;
        gMeanScaled->SetPoint(i, hAmpProjScaled[i]->GetMean(), (double)i + 0.5);
        hAmpProjScaled[i]->SetLineColor(RainbowColor[i % ndraw]);
        hAmpProjScaled[i]->SetTitle(Form(";Scaled %s amp;Counts", detName.data()));
    }
    gMeanScaled->SetMarkerStyle(20);

    for (int indraw=0; indraw<nCh/ndraw; indraw++) {
        hAmpProj[indraw]->SetMaximum(1e7); //!
        hAmpProj[indraw]->Draw();
        leg->Clear();
        // leg->AddEntry((TObject*)0, Form("run ID: %d",arunId), "");
        if (detName == "FT0" && indraw < 12) {
            leg->SetHeader(Form("FT0A, run ID: %d",arunId));
        } else if (detName == "FT0" && indraw >= 12) {
            leg->SetHeader(Form("FT0C, run ID: %d",arunId));
        } else {
            leg->SetHeader(Form("FV0, run ID: %d",arunId));
        }
        for (int j=0; j<ndraw; j++) {
            hAmpProj[indraw + j]->Draw("same");
            leg->AddEntry(hAmpProj[indraw + j], Form("Ch Id: %d",indraw*ndraw + j), "l");
        }
        leg->Draw();
        if (draw) c->SaveAs(Form("%s/%s/gainCor/%s/fig_gain_%d/amp_%dId.pdf", outDir.data(), dataset.data(), detName.data(), arunId, indraw));
    }

    gPad->SetLogy(0);
    gPad->SetLogz(1);

    gPad->SetRightMargin(0.1);
    if (detName == "FV0") {
        hFIT->GetXaxis()->SetRangeUser(0, 3000);
        hFIT->GetYaxis()->SetRangeUser(-0.5, 50.0);
    }

    hFIT->Draw("colz");
    gMean->SetMarkerStyle(20);
    gMean->Draw("P");

    TLine* lB = new TLine(0,96,5000,96);
    lB->SetLineStyle(1);
    lB->SetLineWidth(5);
    lB->SetLineColor(kBlue+3);
    lB->Draw("same");

    // Config detector
    if (draw) c->SaveAs(Form("%s/%s/gainCor/%s/summary/raw_amps/amp_id_2D_%d.pdf", outDir.data(), dataset.data(), detName.data(), arunId));

    hFITScaled->GetXaxis()->SetRangeUser(0, 3000);
    if (detName == "FV0") {
        hFITScaled->GetYaxis()->SetRangeUser(-0.5, 50.0);
        hFITScaled->GetXaxis()->SetRangeUser(0, 3000.0);
    }
    hFITScaled->Draw("colz");
    gMeanScaled->Draw("P");

    if (draw) c->SaveAs(Form("%s/%s/gainCor/%s/summary/scaled_amps/amp_id_2D_scaled_%d.pdf", outDir.data(), dataset.data(), detName.data(), arunId));

    gPad->SetLogy();
    gPad->SetRightMargin(0.03);
    for (int i=0; i<nCh/ndraw; i++) {

        if (hAmpProjScaled[i*ndraw]) hAmpProjScaled[i*ndraw]->Draw();
        leg->Clear();
        // leg->AddEntry((TObject*)0, Form("runID: %d",arunId), "");
        if (detName == "FT0" && i < 12) {
            leg->SetHeader(Form("FT0A, run ID: %d",arunId));
        } else if (detName == "FT0" && i >= 12) {
            leg->SetHeader(Form("FT0C, run ID: %d",arunId));
        } else {
            leg->SetHeader(Form("FV0, run ID: %d",arunId));
        }
        for (int j=0; j<ndraw; j++) {
            if (hAmpProjScaled[i*ndraw + j]) hAmpProjScaled[i*ndraw + j]->Draw("same");
            leg->AddEntry(hAmpProjScaled[i*ndraw + j], Form("Ch Id: %d",i*ndraw + j), "l");
        }
        leg->Draw();
        std::cout << "saving ... " << std::endl;
        if (draw) c->SaveAs(Form("%s/%s/gainCor/%s/fig_gain_%d/ScaledAmp_%dId.pdf", outDir.data(), dataset.data(), detName.data(), arunId, i));
    }

    for (int i=0; i<nCh; i++) {
        if (RelGain[i] == 0) RelGain[i] = 1;
        corr.push_back(RelGain[i]);
    }

    delete c;
}

void calibGainDetector(std::string outDir, std::string inFile, int nCh, bool draw, std::string detName, std::string dataset) {

    std::vector<double> corrConst;

    std::vector<TGraph*> gMeanVal(nCh);
    for (int i=0; i<nCh; i++) {
        gMeanVal[i] = new TGraph();
        gMeanVal[i]->SetMarkerStyle(20);
        gMeanVal[i]->GetXaxis()->SetNoExponent(1);

        gMeanVal[i]->GetXaxis()->SetTitleFont(43);
        gMeanVal[i]->GetYaxis()->SetTitleFont(43);
        gMeanVal[i]->GetXaxis()->SetLabelFont(43);
        gMeanVal[i]->GetYaxis()->SetLabelFont(43);

        gMeanVal[i]->GetXaxis()->SetTitleSize(32);
        gMeanVal[i]->GetYaxis()->SetTitleSize(32);
        gMeanVal[i]->GetXaxis()->SetLabelSize(28);
        gMeanVal[i]->GetYaxis()->SetLabelSize(28);

        gMeanVal[i]->SetTitle(";run ID;Relative gain");

    }
    for (int i=0; i<nrun; i++) {
        cout << "run: " << runnums[i] << endl;
        calibGainRun(outDir, inFile, nCh, runnums[i], corrConst, runnums[i], draw, detName, dataset);
        cout << "constructed" << endl;
        for (int j=0; j<nCh; j++) {
            gMeanVal[j]->SetPoint(i, runnums[i], corrConst.at(j));
        }

        // if (saveCCDB) {
        //     metadata["runnum"] = std::to_string(runnums[i]);
        //     metadata["detector"] = "FT0";

        //     ULong64_t sor = sors[i];
        //     ULong64_t eor = eors[i];
        //     ccdb.storeAsTFileAny(&corrConst, Form("%s/%s", ccdbInternalPath.data(), "FT0"), metadata, sor, eor);

        //     metadata.clear();
        // }
        cout << "finished " << endl;
        corrConst.clear();
    }
    if (draw) {
        TCanvas* c2 = new TCanvas("c2","c2",800,700);
        gPad->SetLeftMargin(0.15);
        gPad->SetBottomMargin(0.15);
        gPad->SetRightMargin(0.03);
        gPad->SetTopMargin(0.03);
        gPad->SetTicks();

        TLegend* leg2 = new TLegend(0.4,0.6,0.9,0.9);
        leg2->SetLineWidth(0.0);
        leg2->SetFillStyle(0);
        leg2->SetTextFont(43);
        leg2->SetTextSize(22);

        std::vector<TF1*> fpol0(nCh);
        std::vector<double> var(nCh);
        for (int i=0; i<nCh; i++) {
            gMeanVal[i]->Draw("AP");

            fpol0[i] = new TF1("f1","[0]",runnums[0],runnums[nrun-1]);
            gMeanVal[i]->Fit(fpol0[i], "Q");

            var[i] = 0.0;
            for (int j=0; j<gMeanVal[ i]->GetN();j++) {
                var[i] += pow(gMeanVal[i]->GetY()[j] - fpol0[i]->GetParameter(0) ,2);
            }
            var[i] = sqrt(var[i]);
            var[i] /= fpol0[i]->GetParameter(0);
            leg2->Clear();
            leg2->AddEntry((TObject*)0, Form("channel ID: %d",i), "");
            leg2->AddEntry((TObject*)0, Form("STD from fit; %.3lf", var[i]), "");
            leg2->Draw();
            c2->SaveAs(Form("%s/%s/gainCor/%s/figs_rundep/gain_vs_runID_%dch.pdf", outDir.data(), dataset.data(), detName.data(), i));
        }

        TGraph* gvar = new TGraph();
        for (int i=0; i<nCh; i++) {
            gvar->SetPoint(i, (double)i, var[i]);
        }
        gvar->SetTitle(";Channel ID;Standard Deviation");
        gvar->SetMarkerStyle(20);
        gvar->Draw("AP");
        c2->SaveAs(Form("%s/%s/gainCor/%s/figs_rundep/gainvar.pdf", outDir.data(), dataset.data(), detName.data()));
    }
}

int calibGain(std::string outDir, std::string inFile) {

    std::cout << "start" << std::endl;
    bool saveCCDB = false;
    bool draw = true;
    initColors();

    // const string ccdbPath = "http://alice-ccdb.cern.ch";
    // const string ccdbInternalPath = "Users/m/mdicosta/Qvector/Pass4/GainEq";
    // o2::ccdb::CcdbApi ccdb;
    // map<string, string> metadata;//, metadataRCT, header; // NOTE: Re-enable the other two if timing information is needed.
    // ccdb.init(Form("%s", ccdbPath.data()));

    cout << "init " << endl;

    std::cout << "Running gain calibration for FT0" << std::endl;
    const int nChFT0 = 208;
    gSystem->mkdir(Form("%s/%s/gainCor/FT0/summary/raw_amps", outDir.data(), runstr.data()), true);
    gSystem->mkdir(Form("%s/%s/gainCor/FT0/summary/scaled_amps", outDir.data(), runstr.data()), true);
    gSystem->mkdir(Form("%s/%s/gainCor/FT0/figs_rundep", outDir.data(), runstr.data()), true);
    calibGainDetector(outDir, inFile, nChFT0, draw, "FT0", runstr);

    std::cout << "Ended FT0 gain calibration\n\n\n" << std::endl;
    
    const int nChFV0 = 48;
    gSystem->mkdir(Form("%s/%s/gainCor/FV0/summary/raw_amps", outDir.data(), runstr.data()), true);
    gSystem->mkdir(Form("%s/%s/gainCor/FV0/summary/scaled_amps", outDir.data(), runstr.data()), true);
    gSystem->mkdir(Form("%s/%s/gainCor/FV0/figs_rundep", outDir.data(), runstr.data()), true);
    calibGainDetector(outDir, inFile, nChFV0, draw, "FV0", runstr);
    std::cout << "Ended FV0 gain calibration\n\n\n" << std::endl;

    return 0;
}
