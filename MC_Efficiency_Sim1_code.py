import os
import SVAR.MoG as MoG
import SVAR.estSVAR
import SVAR.estimatorGMM
import SVAR.SVARutil
import SVAR.SVARutilGMM
import numpy as np
import pickle
import copy
from scipy.stats import norm
import scipy.stats


def runTests(avarInd, avarUnc, GMM_out, B_true, out):
    [n, n] = np.shape(B_true)

    # Coverage 90% w.
    alpha = 0.1
    z = norm.ppf(1 - alpha / 2)
    s = np.sqrt(np.diag(avarInd))
    lowerInd = SVAR.get_BVector(GMM_out['B_est']) - z * s / np.sqrt(GMM_out['options']['T'])
    upperInd = SVAR.get_BVector(GMM_out['B_est']) + z * s / np.sqrt(GMM_out['options']['T'])
    coverageInd = (lowerInd < SVAR.get_BVector(B_true)) & \
                  (upperInd > SVAR.get_BVector(B_true))
    out['lowerInd90'] = lowerInd
    out['upperInd90'] = upperInd
    out['coverageInd90'] = coverageInd

    s = np.sqrt(np.diag(avarUnc))
    lowerUnc = SVAR.get_BVector(GMM_out['B_est']) - z * s / np.sqrt(GMM_out['options']['T'])
    upperUnc = SVAR.get_BVector(GMM_out['B_est']) + z * s / np.sqrt(GMM_out['options']['T'])
    coverageUnc = (lowerUnc < SVAR.get_BVector(B_true)) & \
                  (upperUnc > SVAR.get_BVector(B_true))
    out['lowerUnc90'] = lowerUnc
    out['upperUnc90'] = upperUnc
    out['coverageUnc90'] = coverageUnc

    # Coverage 68% w.
    alpha = 0.32
    z = norm.ppf(1 - alpha / 2)
    s = np.sqrt(np.diag(avarInd))
    lowerInd = SVAR.get_BVector(GMM_out['B_est']) - z * s / np.sqrt(GMM_out['options']['T'])
    upperInd = SVAR.get_BVector(GMM_out['B_est']) + z * s / np.sqrt(GMM_out['options']['T'])
    coverageInd = (lowerInd < SVAR.get_BVector(B_true)) & \
                  (upperInd > SVAR.get_BVector(B_true))
    out['lowerInd68'] = lowerInd
    out['upperInd68'] = upperInd
    out['coverageInd68'] = coverageInd

    s = np.sqrt(np.diag(avarUnc))
    lowerUnc = SVAR.get_BVector(GMM_out['B_est']) - z * s / np.sqrt(GMM_out['options']['T'])
    upperUnc = SVAR.get_BVector(GMM_out['B_est']) + z * s / np.sqrt(GMM_out['options']['T'])
    coverageUnc = (lowerUnc < SVAR.get_BVector(B_true)) & \
                  (upperUnc > SVAR.get_BVector(B_true))
    out['lowerUnc68'] = lowerUnc
    out['upperUnc68'] = upperUnc
    out['coverageUnc68'] = coverageUnc

    GMM_outsave = copy.deepcopy(GMM_out)
    testrest = B_true
    # Wald-B0
    B = GMM_outsave['B_est']
    restrictions = GMM_outsave['options']['restrictions']
    wald_stat, wald_stat_p = SVAR.SVARutilGMM.waldRest(B, avarInd, restrictions,
                                                       GMM_out['options']['T'], testrest)
    out['WaldB0Ind'] = wald_stat
    out['WaldB0PInd'] = wald_stat_p

    B = GMM_outsave['B_est']
    restrictions = GMM_outsave['options']['restrictions']
    wald_stat, wald_stat_p = SVAR.SVARutilGMM.waldRest(B, avarUnc, restrictions,
                                                       GMM_out['options']['T'], testrest)
    out['WaldB0Unc'] = wald_stat
    out['WaldB0PUnc'] = wald_stat_p




    GMM_outsave = copy.deepcopy(GMM_out)
    testrest = np.full([n, n], np.nan)
    testrest[n - 1, n - 2] = 5
    # Wald-B41
    B = GMM_outsave['B_est']
    restrictions = GMM_outsave['options']['restrictions']
    wald_stat, wald_stat_p = SVAR.SVARutilGMM.waldRest(B, avarInd, restrictions,
                                                       GMM_out['options']['T'], testrest)
    out['Waldb41Ind'] = wald_stat
    out['Waldb41PInd'] = wald_stat_p

    B = GMM_outsave['B_est']
    restrictions = GMM_outsave['options']['restrictions']
    wald_stat, wald_stat_p = SVAR.SVARutilGMM.waldRest(B, avarUnc, restrictions,
                                                       GMM_out['options']['T'], testrest)
    out['Waldb41Unc'] = wald_stat
    out['Waldb41PUnc'] = wald_stat_p

    GMM_outsave = copy.deepcopy(GMM_out)
    testrest = np.full([n, n], np.nan)
    testrest[n - 1, n - 2] = 5.5
    # Wald-B41
    B = GMM_outsave['B_est']
    restrictions = GMM_outsave['options']['restrictions']
    wald_stat, wald_stat_p = SVAR.SVARutilGMM.waldRest(B, avarInd, restrictions,
                                                       GMM_out['options']['T'], testrest)
    out['Waldb41Ind55'] = wald_stat
    out['Waldb41PInd55'] = wald_stat_p

    B = GMM_outsave['B_est']
    restrictions = GMM_outsave['options']['restrictions']
    wald_stat, wald_stat_p = SVAR.SVARutilGMM.waldRest(B, avarUnc, restrictions,
                                                       GMM_out['options']['T'], testrest)
    out['Waldb41Unc55'] = wald_stat
    out['Waldb41PUnc55'] = wald_stat_p

    GMM_outsave = copy.deepcopy(GMM_out)
    testrest = np.full([n, n], np.nan)
    testrest[n - 1, n - 2] = 6.0
    # Wald-B41
    B = GMM_outsave['B_est']
    restrictions = GMM_outsave['options']['restrictions']
    wald_stat, wald_stat_p = SVAR.SVARutilGMM.waldRest(B, avarInd, restrictions,
                                                       GMM_out['options']['T'], testrest)
    out['Waldb41Ind60'] = wald_stat
    out['Waldb41PInd60'] = wald_stat_p

    B = GMM_outsave['B_est']
    restrictions = GMM_outsave['options']['restrictions']
    wald_stat, wald_stat_p = SVAR.SVARutilGMM.waldRest(B, avarUnc, restrictions,
                                                       GMM_out['options']['T'], testrest)
    out['Waldb41Unc60'] = wald_stat
    out['Waldb41PUnc60'] = wald_stat_p

    GMM_outsave = copy.deepcopy(GMM_out)
    testrest = np.full([n, n], np.nan)
    testrest[n - 1, n - 2] = 6.5
    # Wald-B41
    B = GMM_outsave['B_est']
    restrictions = GMM_outsave['options']['restrictions']
    wald_stat, wald_stat_p = SVAR.SVARutilGMM.waldRest(B, avarInd, restrictions,
                                                       GMM_out['options']['T'], testrest)
    out['Waldb41Ind65'] = wald_stat
    out['Waldb41PInd65'] = wald_stat_p

    B = GMM_outsave['B_est']
    restrictions = GMM_outsave['options']['restrictions']
    wald_stat, wald_stat_p = SVAR.SVARutilGMM.waldRest(B, avarUnc, restrictions,
                                                       GMM_out['options']['T'], testrest)
    out['Waldb41Unc65'] = wald_stat
    out['Waldb41PUnc65'] = wald_stat_p


    GMM_outsave = copy.deepcopy(GMM_out)
    testrest = np.full([n, n], np.nan)
    testrest[n - 1, n - 2] = 7
    # Wald-B41
    B = GMM_outsave['B_est']
    restrictions = GMM_outsave['options']['restrictions']
    wald_stat, wald_stat_p = SVAR.SVARutilGMM.waldRest(B, avarInd, restrictions,
                                                       GMM_out['options']['T'], testrest)
    out['Waldb41Ind70'] = wald_stat
    out['Waldb41PInd70'] = wald_stat_p

    B = GMM_outsave['B_est']
    restrictions = GMM_outsave['options']['restrictions']
    wald_stat, wald_stat_p = SVAR.SVARutilGMM.waldRest(B, avarUnc, restrictions,
                                                       GMM_out['options']['T'], testrest)
    out['Waldb41Unc70'] = wald_stat
    out['Waldb41PUnc70'] = wald_stat_p


    GMM_outsave = copy.deepcopy(GMM_out)
    testrest = np.full([n, n], np.nan)
    testrest[n - 1, n - 2] = 4.5
    # Wald-B41
    B = GMM_outsave['B_est']
    restrictions = GMM_outsave['options']['restrictions']
    wald_stat, wald_stat_p = SVAR.SVARutilGMM.waldRest(B, avarInd, restrictions,
                                                       GMM_out['options']['T'], testrest)
    out['Waldb41Ind45'] = wald_stat
    out['Waldb41PInd45'] = wald_stat_p

    B = GMM_outsave['B_est']
    restrictions = GMM_outsave['options']['restrictions']
    wald_stat, wald_stat_p = SVAR.SVARutilGMM.waldRest(B, avarUnc, restrictions,
                                                       GMM_out['options']['T'], testrest)
    out['Waldb41Unc45'] = wald_stat
    out['Waldb41PUnc45'] = wald_stat_p


    GMM_outsave = copy.deepcopy(GMM_out)
    testrest = np.full([n, n], np.nan)
    testrest[n - 1, n - 2] = 4.0
    # Wald-B41
    B = GMM_outsave['B_est']
    restrictions = GMM_outsave['options']['restrictions']
    wald_stat, wald_stat_p = SVAR.SVARutilGMM.waldRest(B, avarInd, restrictions,
                                                       GMM_out['options']['T'], testrest)
    out['Waldb41Ind40'] = wald_stat
    out['Waldb41PInd40'] = wald_stat_p

    B = GMM_outsave['B_est']
    restrictions = GMM_outsave['options']['restrictions']
    wald_stat, wald_stat_p = SVAR.SVARutilGMM.waldRest(B, avarUnc, restrictions,
                                                       GMM_out['options']['T'], testrest)
    out['Waldb41Unc40'] = wald_stat
    out['Waldb41PUnc40'] = wald_stat_p






    GMM_outsave = copy.deepcopy(GMM_out)
    testrest = np.full([n, n], np.nan)
    testrest[n - 1, n - 2] = 3.5
    # Wald-B41
    B = GMM_outsave['B_est']
    restrictions = GMM_outsave['options']['restrictions']
    wald_stat, wald_stat_p = SVAR.SVARutilGMM.waldRest(B, avarInd, restrictions,
                                                       GMM_out['options']['T'], testrest)
    out['Waldb41Ind35'] = wald_stat
    out['Waldb41PInd35'] = wald_stat_p

    B = GMM_outsave['B_est']
    restrictions = GMM_outsave['options']['restrictions']
    wald_stat, wald_stat_p = SVAR.SVARutilGMM.waldRest(B, avarUnc, restrictions,
                                                       GMM_out['options']['T'], testrest)
    out['Waldb41Unc35'] = wald_stat
    out['Waldb41PUnc35'] = wald_stat_p




    GMM_outsave = copy.deepcopy(GMM_out)
    testrest = np.full([n, n], np.nan)
    testrest[n - 1, n - 2]= 3
    # Wald-B41
    B = GMM_outsave['B_est']
    restrictions = GMM_outsave['options']['restrictions']
    wald_stat, wald_stat_p = SVAR.SVARutilGMM.waldRest(B, avarInd, restrictions,
                                                       GMM_out['options']['T'], testrest)
    out['Waldb41Ind30'] = wald_stat
    out['Waldb41PInd30'] = wald_stat_p

    B = GMM_outsave['B_est']
    restrictions = GMM_outsave['options']['restrictions']
    wald_stat, wald_stat_p = SVAR.SVARutilGMM.waldRest(B, avarUnc, restrictions,
                                                       GMM_out['options']['T'], testrest)
    out['Waldb41Unc30'] = wald_stat
    out['Waldb41PUnc30'] = wald_stat_p

    return out




def OneMCIteration(path, jobno):
    print(jobno)
    np.random.seed(jobno)

    # SVAR Settings
    N = [5]
    T = [200, 300, 400,500,600,700,  800,900,1000, 5000]

    # shocks
    mu1, sigma1 = (-0.2, np.power(0.7, 2))
    mu2, sigma2 = (0.7501187648456057, np.power(1.4832737225521377, 2))
    lamb = 0.7895
    Omega = np.array([[mu1, sigma1], [mu2, sigma2]])

    mcit_out = dict()
    printOutput = False
    if False:
        mcit_out = dict()
        n= 5
        T_this = 200
        printOutput = True

    for T_this in T:
        for n in N:


            # Specitfy B_true
            B_true = np.array([[1, 0  , 0  , 0, 0],
                               [0.5, 1, 0 , 0, 0],
                               [0.5, 0.5, 1, 0, 0],
                               [0.5, 0.5,  0.5  , 1, 0],
                               [0.5, 0.5,  0.5  , 0.5 , 1 ]])

            V = np.linalg.cholesky(np.matmul(B_true, np.transpose(B_true)))
            O = np.matmul(np.linalg.inv(V), B_true)
            b_true = SVAR.SVARutil.get_Skewsym(O)
            B_true = B_true * 10

            # Draw structural shocks
            eps = np.empty([T_this, n])
            omega = np.zeros([n, 6])
            for i in range(n):
                    eps[:, i] = MoG.MoG_rnd(Omega, lamb, T_this)
                    momentstmp = SVAR.MoG.MoG_Moments(Omega, lamb)[1:]
                    for i in range(n):
                        omega[i, :] = momentstmp

            # Generate reduced form shocks
            u = np.matmul(B_true, np.transpose(eps))
            u = np.transpose(u)

            # General Output
            mcit_out['estimators'] = []
            mcit_out['T'] = T_this
            mcit_out['n'] = n
            mcit_out['B_true'] = B_true
            mcit_out['b_true'] = b_true


            restrictions_none = np.full([n, n], np.nan)
            restrictions_rec = SVAR.SVARutil.getRestrictions_recursive(np.eye(n))

            addThirdMoments = True
            addFourthMoments = True

            momentsChol = SVAR.SVARutilGMM.get_Moments('GMM', n, blocks=False,
                                                      addThirdMoments=False,
                                                      addFourthMoments=False,
                                                      moments_MeanIndep=False,
                                                      onlybivariate=False)
            S0Chol = SVAR.estimatorGMM.get_S_Indep(momentsChol, momentsChol, omega)
            G0Chol = SVAR.SVARutilGMM.get_G_Indep(momentsChol, B_true, omega, restrictions_rec)
            W0Chol = np.linalg.inv(S0Chol)
            VChol = SVAR.estimatorGMM.get_Avar(n, G0Chol, S0Chol, W=W0Chol, restrictions=restrictions_rec)
            VChol = VChol[np.isnan(VChol) == False].reshape((np.sum(np.isnan(restrictions_rec)),
                                                 np.sum(np.isnan(restrictions_rec))))
            mcit_out['VChol'] = VChol


            momentsAll =  SVAR.SVARutilGMM.get_Moments('GMM', n, blocks=False,
                                               addThirdMoments=addThirdMoments,
                                               addFourthMoments=addFourthMoments,
                                                    moments_MeanIndep=False,
                                                    onlybivariate=False)
            S0 = SVAR.estimatorGMM.get_S_Indep(momentsAll, momentsAll, omega)
            G0 = SVAR.SVARutilGMM.get_G_Indep(momentsAll, B_true, omega, restrictions_rec)
            W0 = np.linalg.inv(S0)
            V = SVAR.estimatorGMM.get_Avar(n, G0, S0, W=W0, restrictions=restrictions_rec)
            V = V[np.isnan(V) == False].reshape((np.sum(np.isnan(restrictions_rec)),
                                                 np.sum(np.isnan(restrictions_rec))))
            mcit_out['V'] = V


            momentsAllBiv =  SVAR.SVARutilGMM.get_Moments('GMM', n, blocks=False,
                                               addThirdMoments=addThirdMoments,
                                               addFourthMoments=addFourthMoments,
                                                    moments_MeanIndep=False,
                                                    onlybivariate=True)
            S0Biv = SVAR.estimatorGMM.get_S_Indep(momentsAllBiv, momentsAllBiv, omega)
            G0Biv = SVAR.SVARutilGMM.get_G_Indep(momentsAllBiv, B_true, omega, restrictions_rec)
            W0Biv  = np.linalg.inv(S0Biv)
            VBiv = SVAR.estimatorGMM.get_Avar(n, G0Biv, S0Biv, W=W0Biv, restrictions=restrictions_rec)
            VBiv = VBiv[np.isnan(VBiv) == False].reshape((np.sum(np.isnan(restrictions_rec)),
                                                 np.sum(np.isnan(restrictions_rec))))
            mcit_out['VBiv'] = VBiv

            ############ Cholesky
            if True:
                try:
                    estimator = 'GMM'
                    estimator_name = 'Cholesky'
                    prepOptions = dict()
                    prepOptions['printOutput'] = printOutput
                    prepOptions['bstartopt'] = 'Rec'
                    prepOptions['n_rec'] = n
                    prepOptions['addThirdMoments'] = False
                    prepOptions['addFourthMoments'] = False
                    prepOptions['moments_MeanIndep'] = False
                    prepOptions['moments_blocks'] = False
                    prepOptions['onlybivariate'] = False
                    prepOptions['Wpara'] = 'Uncorrelated'
                    prepOptions['Avarparametric'] = 'Uncorrelated'
                    prepOptions['Wstartopt'] = 'I'
                    prepOptions['S_func'] = False
                    prepOptions['kstep'] = 1
                    prepOptions['WupdateInOutput'] = False

                    GMM_out = SVAR.SVARest(u, estimator=estimator, prepOptions=prepOptions)

                    # Output estimator
                    mcit_out['estimators'].append(estimator_name)
                    mcit_out[estimator_name] = dict()
                    mcit_out[estimator_name]['b_est'] = GMM_out['b_est']
                    mcit_out[estimator_name]['B_est'] = GMM_out['B_est']

                    # Estimates of S,G, Avar
                    hatSind = SVAR.SVARutilGMM.get_S_Indep(GMM_out['options']['moments'], GMM_out['options']['moments'],
                                                           omega=GMM_out['omega'])
                    hatSunc = SVAR.SVARutilGMM.get_S(u, GMM_out['b_est'], GMM_out['options']['moments'],
                                                     GMM_out['options']['restrictions'])

                    hatGind = SVAR.SVARutilGMM.get_G_Indep(GMM_out['options']['moments'], GMM_out['B_est'],
                                                           GMM_out['omega'], GMM_out['options']['restrictions'])
                    hatGunc = GMM_out['options']['Jacobian'](u=u, b=GMM_out['b_est'],
                                                             restrictions=GMM_out['options']['restrictions'],
                                                             CoOmega=SVAR.SVARutil.get_CoOmega(GMM_out['e']))

                    avarInd = SVAR.estimatorGMM.get_Avar(GMM_out['options']['n'], hatGind, hatSind,
                                                         W=np.linalg.inv(hatSind),
                                                         restrictions=GMM_out['options']['restrictions'])
                    avarUnc = SVAR.estimatorGMM.get_Avar(GMM_out['options']['n'], hatGunc, hatSunc,
                                                         W=np.linalg.inv(hatSunc),
                                                         restrictions=GMM_out['options']['restrictions'])

                    # Run Tests
                    mcit_out[estimator_name] = runTests(avarInd, avarUnc, GMM_out, B_true, mcit_out[estimator_name])

                except:
                    print("Error: " + estimator_name)


 

            ############ CUE
            if True:
                try:
                    estimator = 'CUE'
                    estimator_name = 'CUE-MI'
                    prepOptions = dict()
                    prepOptions['printOutput'] = printOutput
                    prepOptions['bstartopt'] = 'Rec'
                    prepOptions['n_rec'] = n
                    prepOptions['addThirdMoments'] = addThirdMoments
                    prepOptions['addFourthMoments'] = addFourthMoments
                    prepOptions['moments_MeanIndep'] = False
                    prepOptions['moments_blocks'] = False
                    prepOptions['onlybivariate'] = False
                    prepOptions['Wpara'] = 'Independent'
                    prepOptions['Avarparametric'] = 'Independent'
                    prepOptions['S_func'] = True
                    prepOptions['WupdateInOutput'] = False



                    GMM_out = SVAR.SVARest(u, estimator=estimator, prepOptions=prepOptions)


                    # Output estimator
                    mcit_out['estimators'].append(estimator_name)
                    mcit_out[estimator_name] = dict()
                    mcit_out[estimator_name]['b_est'] = GMM_out['b_est']
                    mcit_out[estimator_name]['B_est'] = GMM_out['B_est']

                    # Estimates of S,G, Avar
                    hatSind = SVAR.SVARutilGMM.get_S_Indep(GMM_out['options']['moments'], GMM_out['options']['moments'],
                                                           omega=GMM_out['omega'])
                    hatSunc = SVAR.SVARutilGMM.get_S(u, GMM_out['b_est'], GMM_out['options']['moments'],
                                                     GMM_out['options']['restrictions'])

                    hatGind = SVAR.SVARutilGMM.get_G_Indep(GMM_out['options']['moments'], GMM_out['B_est'],
                                                           GMM_out['omega'], GMM_out['options']['restrictions'])
                    hatGunc = GMM_out['options']['Jacobian'](u=u, b=GMM_out['b_est'],
                                                             restrictions=GMM_out['options']['restrictions'],
                                                             CoOmega=SVAR.SVARutil.get_CoOmega(GMM_out['e']))

                    avarInd = SVAR.estimatorGMM.get_Avar(GMM_out['options']['n'], hatGind, hatSind,
                                                         W=np.linalg.inv(hatSind),
                                                         restrictions=GMM_out['options']['restrictions'])
                    avarUnc = SVAR.estimatorGMM.get_Avar(GMM_out['options']['n'], hatGunc, hatSunc,
                                                         W=np.linalg.inv(hatSunc),
                                                         restrictions=GMM_out['options']['restrictions'])

                    # Run Tests
                    mcit_out[estimator_name] = runTests(avarInd, avarUnc, GMM_out, B_true,mcit_out[estimator_name])


                except:
                    print("Error: "+ estimator_name)

            if True:
                try:
                    estimator = 'CUE'
                    estimator_name = 'CUE-MI-Biv'
                    prepOptions = dict()
                    prepOptions['printOutput'] = printOutput
                    prepOptions['bstartopt'] = 'Rec'
                    prepOptions['n_rec'] = n
                    prepOptions['addThirdMoments'] = addThirdMoments
                    prepOptions['addFourthMoments'] = addFourthMoments
                    prepOptions['moments_MeanIndep'] = False
                    prepOptions['moments_blocks'] = False
                    prepOptions['onlybivariate'] = True
                    prepOptions['Wpara'] = 'Independent'
                    prepOptions['Avarparametric'] = 'Independent'
                    prepOptions['S_func'] = True
                    prepOptions['WupdateInOutput'] = False


                    GMM_out = SVAR.SVARest(u, estimator=estimator, prepOptions=prepOptions)

                    # Output estimator
                    mcit_out['estimators'].append(estimator_name)
                    mcit_out[estimator_name] = dict()
                    mcit_out[estimator_name]['b_est'] = GMM_out['b_est']
                    mcit_out[estimator_name]['B_est'] = GMM_out['B_est']

                    # Estimates of S,G, Avar
                    hatSind = SVAR.SVARutilGMM.get_S_Indep(GMM_out['options']['moments'], GMM_out['options']['moments'],
                                                           omega=GMM_out['omega'])
                    hatSunc = SVAR.SVARutilGMM.get_S(u, GMM_out['b_est'], GMM_out['options']['moments'],
                                                     GMM_out['options']['restrictions'])

                    hatGind = SVAR.SVARutilGMM.get_G_Indep(GMM_out['options']['moments'], GMM_out['B_est'],
                                                           GMM_out['omega'], GMM_out['options']['restrictions'])
                    hatGunc = GMM_out['options']['Jacobian'](u=u, b=GMM_out['b_est'],
                                                             restrictions=GMM_out['options']['restrictions'],
                                                             CoOmega=SVAR.SVARutil.get_CoOmega(GMM_out['e']))

                    avarInd = SVAR.estimatorGMM.get_Avar(GMM_out['options']['n'], hatGind, hatSind,
                                                         W=np.linalg.inv(hatSind),
                                                         restrictions=GMM_out['options']['restrictions'])
                    avarUnc = SVAR.estimatorGMM.get_Avar(GMM_out['options']['n'], hatGunc, hatSunc,
                                                         W=np.linalg.inv(hatSunc),
                                                         restrictions=GMM_out['options']['restrictions'])

                    # Run Tests
                    mcit_out[estimator_name] = runTests(avarInd, avarUnc, GMM_out, B_true, mcit_out[estimator_name])


                except:
                    print("Error: " + estimator_name)


 




            if True:
                # Save results
                file_name = path + "/MCjobno_" + str(jobno) + "_n_" + str(n) + "_T_" + str(
                    T_this) + ".data"
                with open(file_name, 'wb') as filehandle:
                    pickle.dump(mcit_out, filehandle)
                    # print('saved: ', file_name)

    return mcit_out
