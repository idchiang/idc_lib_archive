from time import clock, ctime
import multiprocessing as mp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd
import astropy.units as u
from astropy.constants import N_A
from corner import corner
# from scipy.stats.stats import pearsonr
from .idc_functions import SEMBB


plt.ioff()
parallel_rounds = {'SE': 3, 'FB': 1, 'BE': 3, 'WD': 3, 'PL': 12}
ndims = {'SE': 3, 'FB': 2, 'FBPT': 1, 'PB': 2, 'BEMFB': 4, 'WD': 3,
         'BE': 3, 'PL': 4}
wl = np.array([100.0, 160.0, 250.0, 350.0, 500.0])


def cal_err(masked_dist, masked_pr, ML=None):
    idx = np.argsort(masked_dist)
    sorted_dist = masked_dist[idx]
    sorted_pr = masked_pr[idx]
    csp = np.cumsum(sorted_pr)[:-1]
    csp = np.append(0, csp / csp[-1])
    results = np.interp([0.16, 0.5, 0.84], csp, sorted_dist).tolist()
    if ML is not None:
        results[1] = ML
    return max(results[2] - results[1], results[1] - results[0])


def exp_and_error(var, pr, varname):
    Z = np.sum(pr)
    expected = np.sum(var * pr) / Z
    err = cal_err(var.flatten(), pr.flatten(), expected)
    print('Expected', varname + ':', expected)
    print('    ' + ' ' * len(varname) + 'error:', err)
    return expected, err


def SE_calibration_vs_G14(nop=6):
    method_abbr = 'SE'
    MWSED = np.array([0.71, 1.53, 1.08, 0.56, 0.25])
    df = pd.DataFrame()
    # Correct mode should be 100-Sum_square with Fixen values, or 5
    # Karl's method is 4
    print('################################################')
    print('      Comparing our calibration to G14          ')
    print('################################################')
    #
    # MMy COV
    #
    DCOU = 10.0 / 100.0
    DUNU = 1.0 / 100.0
    FCOU = 2.0 / 100.0
    FUNU = 0.5 / 100.0
    myMat2 = np.array([[DUNU + DCOU, 0, 0, 0, 0],
                       [0, FCOU + FUNU, FCOU, FCOU, FCOU],
                       [0, FCOU, FCOU + FUNU, FCOU, FCOU],
                       [0, FCOU, FCOU, FCOU + FUNU, FCOU],
                       [0, FCOU, FCOU, FCOU, FCOU + FUNU]])**2
    #
    # G14 COV
    #
    COU = (5.0 / 100.0)**2
    UNU = (2.5 / 100.0)**2
    G14Mat2 = np.array([[COU + UNU, COU, COU, COU, COU],
                        [COU, COU + UNU, COU, COU, COU],
                        [COU, COU, COU + UNU, COU, COU],
                        [COU, COU, COU, COU + UNU, COU],
                        [COU, COU, COU, COU, COU + UNU]])
    #
    MWSigmaD = (1e20 * 1.0079 * u.g / N_A.value).to(u.M_sun).value * \
        ((1 * u.pc).to(u.cm).value)**2 / 150.
    print(MWSigmaD)
    #
    # Build fitting grid: G14 grid
    #
    print('################################################')
    print('            Calculating: G14 grid               ')
    print('################################################')
    cov_method = 1

    s = pd.Series()
    logsigma_step = 0.1
    min_logsigma = -4.
    max_logsigma = 1.
    T_step = 1.
    min_T = 5.
    max_T = 75.
    beta_step = 0.25
    min_beta = -1.0
    max_beta = 4.0
    #
    MWcali_mat2 = G14Mat2 if cov_method else myMat2
    #
    logsigmas_1d = np.arange(min_logsigma, max_logsigma, logsigma_step)
    Ts_1d = np.arange(min_T, max_T, T_step)
    betas_1d = np.arange(min_beta, max_beta, beta_step)
    logsigmas, Ts, betas = np.meshgrid(logsigmas_1d, Ts_1d, betas_1d)
    #
    sigmas = 10**logsigmas

    def fitting_model(wl):
        return SEMBB(wl, sigmas, Ts, betas, kappa160=1.)

    def split_herschel(ri, r_, rounds, _wl, wlr, output):
        tic = clock()
        rw = ri + r_ * nop
        lenwls = wlr[rw + 1] - wlr[rw]
        last_time = clock()
        result = np.zeros(list(logsigmas.shape) + [lenwls])
        print("   --process", ri, "starts... (" + ctime() + ") (round",
              (r_ + 1), "of", str(rounds) + ")")
        for i in range(lenwls):
            result[..., i] = fitting_model(_wl[i + wlr[rw]])
            current_time = clock()
            # print progress every 10 mins
            if (current_time > last_time + 600.):
                last_time = current_time
                print("     --process", ri,
                      str(round(100. * (i + 1) / lenwls, 1)) +
                      "% Done. (round", (r_ + 1), "of",
                      str(rounds) + ")")
        output.put((ri, rw, result))
        print("   --process", ri, "Done. Elapsed time:",
              round(clock()-tic, 3), "s. (" + ctime() + ")")
    # Building models
    models_ = np.zeros(list(logsigmas.shape) + [5])
    timeout = 1e-6
    # Applying RSRFs to generate fake-observed models
    instrs = ['PACS', 'SPIRE']
    rounds = parallel_rounds[method_abbr]
    for instr in range(2):
        print(" --Constructing", instrs[instr], "RSRF model... (" +
              ctime() + ")")
        ttic = clock()
        _rsrf = pd.read_csv("data/RSRF/" + instrs[instr] + "_RSRF.csv")
        _wl = _rsrf['Wavelength'].values
        h_models = np.zeros(list(logsigmas.shape) + [len(_wl)])
        wlr = [int(ri * len(_wl) / float(nop * rounds)) for ri in
               range(nop * rounds + 1)]
        if instr == 0:
            rsps = [_rsrf['PACS_100'].values,
                    _rsrf['PACS_160'].values]
            range_ = range(0, 2)
        elif instr == 1:
            rsps = [[], [], _rsrf['SPIRE_250'].values,
                    _rsrf['SPIRE_350'].values,
                    _rsrf['SPIRE_500'].values]
            range_ = range(2, 5)
        del _rsrf
        # Parallel code
        for r_ in range(rounds):
            print("\n   --" + method_abbr, instrs[instr] + ":Round",
                  (r_ + 1), "of", rounds, '\n')
            q = mp.Queue()
            processes = [mp.Process(target=split_herschel,
                         args=(ri, r_, rounds, _wl, wlr, q))
                         for ri in range(nop)]
            for p in processes:
                p.start()
            for p in processes:
                p.join(timeout)
            for p in processes:
                ri, rw, result = q.get()
                print("     --Got result from process", ri)
                h_models[..., wlr[rw]:wlr[rw+1]] = result
                del ri, rw, result
            del processes, q, p
        # Parallel code ends
        print("   --Calculating response function integrals")
        for i in range_:
            models_[..., i] = \
                np.sum(h_models * rsps[i], axis=-1) / \
                np.sum(rsps[i] * _wl / wl[i])
        del _wl, rsps, h_models, range_
        print("   --Done. Elapsed time:", round(clock()-ttic, 3),
              "s.\n")
    print('################################################')
    print('          Model generation complete             ')
    print('################################################')
    # Start fitting
    tic = clock()
    models = models_
    del models_
    temp_matrix = np.empty_like(models)
    diff = models - MWSED
    sed_vec = MWSED.reshape(1, 5)
    C = sed_vec.T * MWcali_mat2 * sed_vec
    cov_n1 = np.linalg.inv(sed_vec.T * MWcali_mat2 * sed_vec)
    for j in range(5):
        temp_matrix[..., j] = np.sum(diff * cov_n1[:, j], axis=-1)
    chi2 = np.sum(temp_matrix * diff, axis=-1)
    #
    Q = np.sqrt((2 * np.pi)**5 * np.abs(np.linalg.det(C)))
    pr = np.exp(-0.5 * chi2) / Q
    logkappa160s = logsigmas - np.log10(MWSigmaD)
    logkappa160, logkappa160_err = \
        exp_and_error(logkappa160s, pr, 'logkappa_160')
    kappa160Q = 10**logkappa160
    TQ, T_err = exp_and_error(Ts, pr, 'T')
    betaQ, beta_err = exp_and_error(betas, pr, 'beta')
    #
    r_chi2 = chi2 / (5. - ndims[method_abbr])
    """ Find the (s, t) that gives Maximum likelihood """
    am_idx = np.unravel_index(chi2.argmin(), chi2.shape)
    """ Probability and mask """
    pr = np.exp(-0.5 * chi2)
    print('Best fit r_chi^2:', r_chi2[am_idx])
    """ kappa 160 """
    logkappa160s = logsigmas - np.log10(MWSigmaD)
    logkappa160, logkappa160_err = \
        exp_and_error(logkappa160s, pr, 'logkappa_160')
    kappa160 = 10**logkappa160
    logsigma, _ = exp_and_error(logsigmas, pr, 'logsigmas')
    #
    # All steps
    print('Best fit kappa160:', kappa160)
    wl_complete = np.linspace(1, 1000, 1000)
    #
    samples = np.array([logkappa160s.flatten(), Ts.flatten(),
                        betas.flatten()])
    labels = [r'$\log\kappa_{160}$', r'$T$', r'$\beta$']
    T, T_err = exp_and_error(Ts, pr, 'T')
    beta, beta_err = exp_and_error(betas, pr, 'beta')
    model_complete = SEMBB(wl_complete, MWSigmaD, T, beta,
                           kappa160=kappa160)
    model_gordon = SEMBB(wl_complete, MWSigmaD, 17.2,
                         1.96, 9.6 * np.pi)
    #
    s.name = 'G14Grid_'
    s.name += 'G14COV' if cov_method else 'MyCOV'
    max_idx = np.unravel_index(np.argmax(pr), logsigmas.shape)
    s['kappa_MAX'] = 10**logsigmas[max_idx] / MWSigmaD
    s['kappa_EXP'] = kappa160
    s['T_MAX'] = Ts[max_idx]
    s['T_EXP'] = T
    s['beta_MAX'] = betas[max_idx]
    s['beta_EXP'] = beta
    #
    s['kappa_EXPQ'] = kappa160Q
    s['T_EXPQ'] = TQ
    s['beta_EXPQ'] = betaQ
    #
    pr2 = np.sum(pr, axis=(0, 2))
    logsigma_1dtest, _ = exp_and_error(logsigmas_1d, pr2, 'logsigma')
    kappa160_1dtest = 10**(logsigma_1dtest - np.log10(MWSigmaD))
    pr2 = np.sum(pr, axis=(1, 2))
    T_1dtest, T_err = exp_and_error(Ts_1d, pr2, 'T')
    pr2 = np.sum(pr, axis=(0, 1))
    beta_1dtest, beta_err = exp_and_error(betas_1d, pr2, 'beta')
    s['kappa_EXP_1D'] = kappa160_1dtest
    s['T_EXP_1D'] = T_1dtest
    s['beta_EXP_1D'] = beta_1dtest
    #
    df = df.append(s)
    fig = corner(samples.T, labels=labels, quantities=(0.16, 0.84),
                 show_titles=True, title_kwargs={"fontsize": 12},
                 weights=pr.flatten())
    fn = 'output/_CALI_Corner_SE_G14Grid_'
    fn += 'G14COV_' if cov_method else 'MyCOV_'
    fn += '.pdf'
    with PdfPages(fn) as pp:
        pp.savefig(fig)
    fig, ax = plt.subplots(figsize=(10, 7.5))
    ax.loglog(wl_complete, model_gordon, label='G14EXP')
    # ax.loglog(wl, mode_integrated, 'x', ms=15, label='fitting (int)')
    ax.loglog(wl_complete, model_complete, label='fitting')
    ax.scatter(wl, MWSED, c='m', marker='s', zorder=0, label='MWSED', s=15)
    # ax.loglog(wl, gordon_integrated, 'x', ms=15, label='G14 (int)')
    ax.legend()
    ax.set_ylim(0.03, 3.0)
    ax.set_xlim(80, 1000)
    ax.set_ylabel(r'SED [$MJy\,sr^{-1}\,(10^{20}' +
                  '(H\,Atom)\,cm^{-2})^{-1}$]')
    ax.set_xlabel(r'Wavelength ($\mu m$)')
    fn = 'output/_CALI_Model_SE_G14Grid_'
    fn += 'G14COV_' if cov_method else 'MyCOV_'
    fn += '.pdf'
    with PdfPages(fn) as pp:
        pp.savefig(fig)
    print(" --Done. Elapsed time:", round(clock()-tic, 3), "s.")
    #
    df.to_csv('output/Calibration_vs_G14.csv')
    return df
