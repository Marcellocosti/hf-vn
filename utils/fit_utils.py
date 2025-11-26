'''
Module with function definitions and fit utils
'''

from ROOT import TMath, TF1, TH1F, kBlue, kGreen, TDatabasePDG, TH1D # pylint: disable=import-error,no-name-in-module

def SingleGaus(x, par):
    '''
    Gaussian function

    Parameters
    ----------
    - x: function variable
    - par: function parameters
        par[0]: normalisation
        par[1]: mean
        par[2]: sigma
    '''
    return par[0]*TMath.Gaus(x[0], par[1], par[2], True)


def DoubleGaus(x, par):
    '''
    Sum of two Gaussian functions with same mean and different sigma

    Parameters
    ----------
    - x: function variable
    - par: function parameters
        par[0]: normalisation
        par[1]: mean
        par[2]: first sigma
        par[3]: second sigma
        par[4]: fraction of integral in second Gaussian
    '''
    firstGaus = TMath.Gaus(x[0], par[1], par[2], True)
    secondGaus = TMath.Gaus(x[0], par[1], par[3], True)
    return par[0] * ((1-par[4])*firstGaus + par[4]*secondGaus)


def DoublePeakSingleGaus(x, par):
    '''
    Sum of two Gaussian functions with different mean and sigma

    Parameters
    ----------
    - x: function variable
    - par: function parameters
        par[0]: normalisation first peak
        par[1]: mean first peak
        par[2]: sigma first peak
        par[3]: normalisation second peak
        par[4]: mean second peak
        par[5]: sigma second peak
    '''
    firstGaus = par[0]*TMath.Gaus(x[0], par[1], par[2], True)
    secondGaus = par[3]*TMath.Gaus(x[0], par[4], par[5], True)
    return firstGaus + secondGaus


def DoublePeakDoubleGaus(x, par):
    '''
    Sum of a double Gaussian function and a single Gaussian function

    Parameters
    ----------
    - x: function variable
    - par: function parameters
        par[0]: normalisation first peak
        par[1]: mean first peak
        par[2]: first sigma first peak
        par[3]: second sigma first peak
        par[4]: fraction of integral in second Gaussian first peak
        par[5]: normalisation second peak
        par[6]: mean second peak
        par[7]: sigma second peak
    '''
    firstGaus = TMath.Gaus(x[0], par[1], par[2], True)
    secondGaus = TMath.Gaus(x[0], par[1], par[3], True)
    thirdGaus = par[5]*TMath.Gaus(x[0], par[6], par[7], True)
    return par[0] * ((1-par[4])*firstGaus + par[4]*secondGaus) + thirdGaus


def VoigtFunc(x, par):
    '''
    Voigtian function

    Parameters
    ----------
    - x: function variable
    - par: function parameters
        par[0]: normalisation
        par[1]: mean
        par[2]: sigma
        par[3]: gamma
    '''

    return par[0] * TMath.Voigt(x[0]-par[1], par[2], par[3])


def ExpoPowLaw(x, par):
    '''
    Exponential times power law function

    Parameters
    ----------
    - x: function variable
    - par: function parameters
        par[0]: normalisation
        par[1]: mass (lowest possible value)
        par[2]: expo slope
    '''

    return par[0] * TMath.Sqrt(x[0] - par[1]) * TMath.Exp(-1. * par[2] * (x[0] - par[1]))


def PeakPowLaw(x, par):
    '''
    Power law function with peak at low x, used to fit pT-differential spectra

    Parameters
    ----------
    - x: function variable
    - par: function parameters
        par[0]: normalisation
        par[1]: alpha
        par[2]: beta
        par[3]: gamma
    '''

    return par[0] * x[0] / TMath.Power((1 + TMath.Power(x[0] / par[1], par[3])), par[2])


# pylint: disable=too-many-instance-attributes
class BkgFitFuncCreator:
    '''
    Class to handle custom background functions as done by AliHFInvMassFitter. Mainly designed
    to provide functions for sidebands fitting

    Parameters
    -------------------------------------------------
    - funcName: function to use. Currently implemented: 'expo', 'pol0', 'pol1', 'pol2', 'pol3'
    - minMass:  lower extreme of fitting interval
    - maxMass:  higher extreme of fitting interval
    - numSigmaSideBands: number of widths excluded around the peak
    - peakMass: peak mass (if not defined the signal region will not be excluded from the function)
    - peakSigma: peak width
    - secPeakMass: second peak mass (if not defined the second-peak region will not be excluded from the function)
    - secPeakSigma: second peak width
    '''
    __implFunc = {'expo': '_ExpoIntegralNorm',
                  'pol0': '_Pol0IntegralNorm',
                  'pol1': '_Pol1IntegralNorm',
                  'pol2': '_Pol2IntegralNorm',
                  'pol3': '_Pol3IntegralNorm',
                  'expopow': '_ExpoPowIntegralNorm'
                  }

    __numPar = {'expo': 2,
                'pol0': 1,
                'pol1': 2,
                'pol2': 3,
                'pol3': 4,
                'expopow': 2
                }

    def __init__(self, funcName, minMass, maxMass, numSigmaSideBands=0., peakMass=0.,
                 peakSigma=0., secPeakMass=0., secPeakSigma=0.):
        if funcName not in self.__implFunc:
            raise ValueError(f'Function \'{funcName}\' not implemented')
        self.funcName = funcName
        self.minMass = minMass
        self.maxMass = maxMass
        self.peakMass = peakMass
        self.peakDelta = peakSigma * numSigmaSideBands
        self.secPeakMass = secPeakMass
        self.secPeakDelta = secPeakSigma * numSigmaSideBands
        self.funcSBCallable = None
        self.funcFullCallable = None

        self.removePeak = False
        self.removeSecPeak = False
        if self.peakMass > 0. and self.peakDelta > 0.:
            self.removePeak = True
        if self.secPeakMass > 0. and self.secPeakDelta > 0.:
            self.removeSecPeak = True

        self.mPi = TDatabasePDG.Instance().GetParticle(211).Mass()

    def _ExpoIntegralNorm(self, x, par):
        '''
        Exponential function normalized to its integral.
        See AliHFInvMassFitter::FitFunction4Bkg for more information.

        Parameters
        ----------
        - x: function variable
        - par: function parameters
            par[0]: normalisation (integral of background)
            par[1]: expo slope
        '''
        norm = par[0] * par[1] / (TMath.Exp(par[1] * self.maxMass) - TMath.Exp(par[1] * self.minMass))
        return norm * TMath.Exp(par[1] * x[0])

    def _Pol0IntegralNorm(self, x, par): # pylint: disable=unused-argument
        '''
        Constant Function normalized to its integral.

        Parameters
        ----------
        - x: function variable
        - par: function parameters
            par[0]: normalisation (integral of background)
        '''
        return par[0] / (self.maxMass - self.minMass)

    def _Pol1IntegralNorm(self, x, par):
        '''
        Linear function normalized to its integral.
        See AliHFInvMassFitter::FitFunction4Bkg for more information.

        Parameters
        ----------
        - x: function variable
        - par: function parameters
            par[0]: normalisation (integral of background)
            par[1]: angular coefficient
        '''
        return par[0] / (self.maxMass - self.minMass) + par[1] * (x[0] - 0.5 * (self.maxMass + self.minMass))

    def _Pol2IntegralNorm(self, x, par):
        '''
        Second order polinomial function normalized to its integral.
        See AliHFInvMassFitter::FitFunction4Bkg for more information.

        Parameters
        ----------
        - x: function variable
        - par: function parameters
            par[0]: normalisation (integral of background)
            par[1]: a
            par[2]: b
        '''
        firstTerm = par[0] / (self.maxMass - self.minMass)
        secondTerm = par[1] * (x[0] - 0.5 * (self.maxMass + self.minMass))
        thirdTerm = par[2] * (x[0]**2 - 1 / 3. * (self.maxMass**3 - self.minMass**3) / (self.maxMass - self.minMass))
        return firstTerm + secondTerm + thirdTerm

    def _Pol3IntegralNorm(self, x, par):
        '''
        Third order polinomial function normalized to its integral.
        See AliHFInvMassFitter::FitFunction4Bkg for more information.

        Parameters
        ----------
        - x: function variable
        - par: function parameters
            par[0]: normalisation (integral of background)
            par[1]: a
            par[2]: b
            par[3]: c
        '''
        firstTerm = par[0] / (self.maxMass - self.minMass)
        secondTerm = par[1] * (x[0] - 0.5 * (self.maxMass + self.minMass))
        thirdTerm = par[2] * (x[0]**2 - 1 / 3. * (self.maxMass**3 - self.minMass**3) / (self.maxMass - self.minMass))
        fourthTerm = par[3] * (x[0]**3 - 1 / 4. * (self.maxMass**4 - self.minMass**4) / (self.maxMass - self.minMass))
        return firstTerm + secondTerm + thirdTerm + fourthTerm

    def _ExpoPowIntegralNorm(self, x, par):
        '''
        Exponential times power law function normalized to its integral for D* background.

        Parameters
        ----------
        - x: function variable
        - par: function parameters
            par[0]: normalisation (integral of background)
            par[1]: expo slope
        '''

        return par[0] * TMath.Sqrt(x[0] - self.mPi) * TMath.Exp(-1. * par[1] * (x[0] - self.mPi))


    def _SideBandsFunc(self, x, par):
        '''
        Function where only sidebands are considered.

        Parameters
        ----------
        - x: function variable
        - par: function parameters
        '''
        if self.removePeak and TMath.Abs(x[0] - self.peakMass) < self.peakDelta:
            TF1.RejectPoint()
            return 0
        if self.removeSecPeak and TMath.Abs(x[0] - self.secPeakMass) < self.secPeakDelta:
            TF1.RejectPoint()
            return 0

        return getattr(self, self.__implFunc[self.funcName])(x, par)

    def GetSideBandsFunc(self, integral):
        '''
        Return the ROOT.TF1 function defined on the sidebands

        Parameters
        --------------------------------------
        integral: integral of the histogram to fit, obtained with TH1.Integral('width')

        Returns
        ---------------------------------------
        funcBkgSB: ROOT.TF1
            Background function
        '''
        self.funcSBCallable = self._SideBandsFunc # trick to keep away the garbage collector
        funcBkgSB = TF1('bkgSBfunc', self.funcSBCallable, self.minMass, self.maxMass, self.__numPar[self.funcName])
        funcBkgSB.SetParName(0, 'BkgInt')
        funcBkgSB.SetParameter(0, integral)
        for iPar in range(1, self.__numPar[self.funcName]):
            funcBkgSB.SetParameter(iPar, 1.)
        funcBkgSB.SetLineColor(kBlue+2)
        return funcBkgSB

    def GetFullRangeFunc(self, func):
        '''
        Return the ROOT.TF1 function defined on the full range

        Parameters
        --------------------------------------
        func: function from GetSideBandsFunc() after the histogram fit

        Returns
        ---------------------------------------
        funcBkg: ROOT.TF1
            Background function
        '''
        self.funcFullCallable = getattr(self, self.__implFunc[self.funcName]) # trick to keep away the garbage collector
        funcBkg = TF1('bkgFunc', self.funcFullCallable, self.minMass, self.maxMass, self.__numPar[self.funcName])
        funcBkg.SetParName(0, 'BkgInt')
        for iPar in range(0, self.__numPar[self.funcName]):
            funcBkg.SetParameter(iPar, func.GetParameter(iPar))
        funcBkg.SetLineColor(kGreen+2)
        return funcBkg

def RebinHisto(h_orig, reb, first_use = 0):
    '''
    Rebin histogram, from bin firstUse to lastUse
    Use all bins if firstUse=-1
    If ngroup is not an exact divider of the number of bins,
    the bin width is kept as reb*original width
    and the range of rebinned histogram is adapted
    '''
    
    n_bin_orig = h_orig.GetNbinsX()
    first_bin_orig = 1
    last_bin_orig = n_bin_orig
    n_bin_orig_used = n_bin_orig
    n_bin_final = n_bin_orig / reb
    
    if first_use >= 1: 
        first_bin_orig = first_use
        n_bin_final = (n_bin_orig-first_use+1) / reb
        n_bin_orig_used = n_bin_final * reb
        last_bin_orig = first_bin_orig + n_bin_orig_used - 1
    else:
        exc = n_bin_orig_used % reb
        if exc != 0: 
            n_bin_orig_used -= exc
            last_bin_orig = first_bin_orig + n_bin_orig_used - 1

    n_bin_final = round(n_bin_final)
    print(f"Rebin from {n_bin_orig} bins to {n_bin_final} bins -- Used bins = {n_bin_orig_used} in range {first_bin_orig}-{last_bin_orig}\n")
    
    low_lim = h_orig.GetXaxis().GetBinLowEdge(first_bin_orig)
    hi_lim = h_orig.GetXaxis().GetBinUpEdge(last_bin_orig)
    hRebin = TH1D(f"{h_orig.GetName()}-rebin", h_orig.GetTitle(), n_bin_final, low_lim, hi_lim)
    last_summed = first_bin_orig-1
    
    for iBin in range(1, n_bin_final+1):
        sum = 0.
        sume2 = 0.
        for _ in range(reb):
            sum += h_orig.GetBinContent(last_summed+1)
            sume2 += (h_orig.GetBinError(last_summed+1) * h_orig.GetBinError(last_summed+1))
            last_summed += 1
            
        hRebin.SetBinContent(iBin, sum)
        hRebin.SetBinError(iBin, TMath.Sqrt(sume2))
    
    return hRebin

def get_tree(input_file, tables, query_signal=None):
    """
    Helper function to get correlated backgrounds tree from file
    """

    print(f"Opening file {input_file} to get correlated backgrounds trees {tables}")
    dfs_list = [[] for _ in range(len(tables))]
    with uproot.open(input_file) as f:
        for key in f.keys():
            for i_table, table in enumerate(tables):
                if table in key:
                    dfs_list[i_table].append(f[key].arrays(library="pd"))

    merged_single_dfs = []
    for df in dfs_list:
        merged_single_dfs.append(pd.concat([single_df for single_df in df], ignore_index=True))
    full_df = pd.concat(merged_single_dfs, axis=0)

    if query_signal:
        print(f"Applying query to select signal: {query_signal}")
        full_df = full_df.query(query_signal)

    print(f"Full tree with {len(full_df)} entries, columns\n: {full_df.columns.to_list()}")
    return full_df

def get_histo(input_file, sparse_dicts, cfg, pt_bin_fit_cfg):
    """
    Helper function to get histogram from file
    """

    if cfg['data_type'] == "DplusTask":
        if cfg["is_data"]:
            sparse = input_file.Get("hf-task-dplus/hSparseMass")
            dict_entry = "Data"
        else:
            sparse = input_file.Get("hf-task-dplus/hSparseMassPrompt")
            dict_entry = "RecoPrompt"
        
        # Apply selections
        print(f"\n\nsparse_dicts: {sparse_dicts}\n\n")
        sparse.GetAxis(sparse_dicts[dict_entry]['Pt']).SetRangeUser(pt_bin_fit_cfg['pt_range'][0], pt_bin_fit_cfg['pt_range'][1])
        if pt_bin_fit_cfg.get('score_bkg_max') and cfg.get('correlated_bkgs'):
            if cfg['correlated_bkgs'].get('apply_ml_score_sel'):
                sparse.GetAxis(sparse_dicts[dict_entry]['score_bkg']).SetRangeUser(0, pt_bin_fit_cfg['score_bkg_max'])
        histo = sparse.Projection(0)

    return histo

def get_data_model_dicts(config, data_type="DplusTask"):
    
    axes_dict = {}
    if data_type == "DplusTask":
        ### Data
        axes_dict['Data'] = {
            'Mass': 0,
            'Pt': 1,
            'score_bkg': 2,
            'score_fd': 4,
            'cent': 5
        }

        ### MC
        axes_dict['RecoPrompt'] = {
            'Mass': 0,
            'Pt': 1,
            'score_bkg': 2,
            'score_prompt': 3,
            'score_fd': 4,
            'cent': 5,
            'occ': 6,
        }
        axes_dict['RecoFD'] = {
            'Mass': 0,
            'Pt': 1,
            'score_bkg': 2,
            'score_prompt': 3,
            'score_fd': 4,
            'cent': 5,
            'occ': 6,
            'pt_bmoth': 7,
            'flag_bhad': 8,
        }
        axes_dict['GenPrompt'] = {
            'Pt': 0,
            'y': 1,
            'cent': 2,
            'occ': 3
        }
        axes_dict['GenFD'] = {
            'Pt': 0,
            'y': 1,
            'cent': 2,
            'occ': 3,
            'pt_bmoth': 4,
            'flag_bhad': 5,
        }

    elif data_type == "DsTask":
        ### Data
        axes_dict['Data'] = {
            'Mass': 0,
            'Pt': 1,
            'cent': 2,
            'score_bkg': 3,
            'score_fd': 5
        }

        ### MC
        axes_dict['RecoPrompt'] = {
        }
        axes_dict['RecoFD'] = {
        }
        axes_dict['GenPrompt'] = {
        }
        axes_dict['GenFD'] = {
        }

    elif data_type == "DplusCorrelator":
        axes_dict["Data"] = {
            'Mass': 'fMD',
            'Pt': 'fPtD',
            'score_bkg': 'fMlScoreBkg',
            'score_prompt': 'fMlScorePrompt',
            'score_fd': 'fMlScoreNonPrompt'
        }

    elif data_type == "DplusTree":
        print(f"Getting data model dicts for data_type: {data_type}")
        axes_dict["Data"] = {
            'Mass': 'fM',
            'Pt': 'fPt',
            'cent': 'fCentrality',
            'score_bkg': 'fMlScore0',
            'score_prompt': 'fMlScorePrompt',
            'score_fd': 'fMlScore1'
        }

        ### MC
        axes_dict['RecoPrompt'] = {
            'Mass': 'fM',
            'Pt': 'fPt',
            'cent': 'fCentrality',
            'score_bkg': 'fMlScore0',
            'score_prompt': 'fMlScorePrompt',
            'score_fd': 'fMlScore1'
        }
        axes_dict['RecoFD'] = {
            'Mass': 'fM',
            'Pt': 'fPt',
            'cent': 'fCentrality',
            'score_bkg': 'fMlScore0',
            'score_prompt': 'fMlScorePrompt',
            'score_fd': 'fMlScore1'
        }
        axes_dict['GenPrompt'] = {
            'Mass': 'fM',
            'Pt': 'fPt',
            'cent': 'fCentrality'
        }
        axes_dict['GenFD'] = {
            'Mass': 'fM',
            'Pt': 'fPt',
            'cent': 'fCentrality',
        }
        
    elif data_type == "PreprocessedTree":
        axes_dict["Data"] = {
            'Mass': 'fM',
            'Pt': 'fPt',
            'cent': 'fCentrality',
            'score_bkg': 'fMlScore0',
            'score_prompt': 'fMlScorePrompt',
            'score_fd': 'fMlScore1'
        }
        ### MC
        axes_dict['RecoPrompt'] = {
            'Mass': 'fM',
            'Pt': 'fPt',
            'cent': 'fCentrality',
            'score_bkg': 'fMlScore0',
            'score_prompt': 'fMlScorePrompt',
            'score_fd': 'fMlScore1'
        }
        axes_dict['RecoFD'] = {
            'Mass': 'fM',
            'Pt': 'fPt',
            'cent': 'fCentrality',
            'score_bkg': 'fMlScore0',
            'score_prompt': 'fMlScorePrompt',
            'score_fd': 'fMlScore1'
        }
        axes_dict['GenPrompt'] = {
            'Pt': 'Pt',
            'cent': 'Centrality'
        }
        axes_dict['GenFD'] = {
            'Pt': 'Pt',
            'cent': 'Centrality'
        }
    else:
        logger(f"Data model for data_type {data_type} not implemented yet.", level='FATAL')

    print(f"\n\nData model dicts for data_type {data_type}: {axes_dict}")
    return axes_dict

def get_signal_pars_dict(sgn_func, pt_limits):
    """
    Helper function to get signal pars dict
    """
    signal_pars_dict = {}
    if sgn_func == ["gaussian"]:
        signal_pars = ['rawyield', 'mu', 'sigma']
    elif sgn_func == ["doublegaus"]:
        signal_pars = ['rawyield', 'mu', 'sigma1', 'sigma2']
    elif sgn_func == ["doublecbsymm"]:
        signal_pars = ['rawyield', 'mu', 'sigma', 'alpha', 'n']
    elif sgn_func == ["doublecb"]:
        signal_pars = ['rawyield', 'mu', 'sigma', 'alphal', 'nl', 'alphar', 'nr']
    elif sgn_func == ["doublecb", "doublecb"]:
        signal_pars = ['rawyield1', 'mu1', 'sigma1', 'alphal1', 'nl1', 'alphar1', 'nr1',
                       'rawyield2', 'mu2', 'sigma2', 'alphal2', 'nl2', 'alphar2', 'nr2']
    elif sgn_func == ["genergausexptailsymm"]:
        signal_pars = ['rawyield', 'mu', 'sigma', 'alpha']
    elif sgn_func == ["genergausexptail"]:
        signal_pars = ['rawyield', 'mu', 'sigmal', 'alphal', 'sigmar', 'alphar']
    elif sgn_func == ["bifurgaus"]:
        signal_pars = ['rawyield', 'mu', 'sigmal', 'alphal']
    elif sgn_func == ["cauchy"]:
        signal_pars = ['rawyield', 'mu', 'gamma']
    elif sgn_func == ["voigtian"]:
        signal_pars = ['rawyield', 'mu', 'sigma', 'gamma']
    else:
        raise ValueError(f"Signal function '{sgn_func}' not recognized.")

    for par in signal_pars:
        signal_pars_dict[par] = TH1F(f"hist_{par}", f";#it{{p}}_{{T}} (GeV/#it{{c}}); {par}",
                                     len(pt_limits)-1, pt_limits)
    return signal_pars_dict

def convert_flarefly_par_name(par_name):
    """
    Helper function to convert flarefly parameter names to common names
    """
    name_map = {
        'rawyield': None,
        'mu': 'Mean',
        'sigma': 'Sigma',
        'alphal': 'Alpha1',
        'nl': 'N1',
        'alphar': 'Alpha2',
        'nr': 'N2',
    }
    return name_map[par_name]
