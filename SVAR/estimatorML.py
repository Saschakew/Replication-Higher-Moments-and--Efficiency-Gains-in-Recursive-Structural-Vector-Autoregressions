import numpy as np
import SVAR
import scipy


def stpdf(e, df_lambdai, df_qi):

    p=2
    v = (df_qi ** (-1 / p)) / np.sqrt((3 * df_lambdai ** 2 + 1) * (1 / (2 * df_qi - 2)) - 4 * df_lambdai ** 2 / np.pi * (scipy.special.gamma(df_qi - 0.5) ** 2) / (scipy.special.gamma(df_qi) ** 2))
    #v = 1

    m = (2 * v * df_lambdai * df_qi ** (1 / p) * scipy.special.gamma(df_qi - 0.5)) / (np.sqrt(np.pi) * scipy.special.gamma(df_qi))

    f1 = (v * (np.pi * df_qi) ** (1 / p) * scipy.special.gamma(df_qi))
    f2 = (abs(e+m)** p)
    f3 =  (df_qi * v ** p * (df_lambdai *  np.sign(e + m) + 1)** p)
    f4 = (( f2 / f3 ) + 1)
    f = scipy.special.gamma(0.5+df_qi) / (f1  * f4 ** (1 / p + df_qi))

    return f
def LogLike_st(u, b_vec, df_lambda,  df_q , restrictions, blocks=False,whiten=False):
    e = SVAR.SVARutil.innovation(u, b_vec, restrictions=restrictions, whiten=whiten, blocks=blocks)
    T, n = np.shape(u)

    B = SVAR.get_BMatrix(b_vec, restrictions=restrictions,  whiten=whiten,
                             blocks=blocks)

    loglikelihood = - T * np.log(abs(np.linalg.det(B))) #+ np.sum(np.log(stpdf(e, df_lambda, df_q)))
    for i in range(n):
        loglikelihood = loglikelihood + np.sum(np.log(stpdf(e[:,i], df_lambda[i], df_q[i])))

    return loglikelihood

def ML_avar(n):
    # ToDo: V_est for ML
    V_est = np.full([n * n, n * n], np.nan)
    return V_est



def prepareOptions(u,
                   df_lambda = 0, df_q = 5,
                   bstart=[], bstartopt='Rec',
                   restrictions=[],blocks=False, n_rec=False,
                   printOutput=True,
                    estimator = 'ML'
                   ):
    options = dict()


    options['estimator'] = estimator

    T, n = np.shape(u)
    options['T'] = T
    options['n'] = n

    options['df_lambda'] = np.full([n],df_lambda)
    options['df_q'] = np.full([n],df_q)

    options['printOutput'] = printOutput

    options['whiten'] = False


    restrictions, blocks = SVAR.estPrepare.prepare_blocks_restrictions(n, n_rec, blocks, restrictions=restrictions)
    options['restrictions'] = restrictions
    options['blocks'] = blocks


    bstart = SVAR.estPrepare.prepare_bstart('GMM', bstart, u, options, bstartopt=bstartopt)
    options['bstart'] = bstart

    return options


def SVARout(est_SVAR, options, u):
    T, n = np.shape(u)
    out_SVAR = dict()
    out_SVAR['options'] = options

    def array_to_ndarrays(concatenated_array, lengthB, n):
        bstart = concatenated_array[:lengthB]
        df_lambda = concatenated_array[lengthB: lengthB + n]
        df_q = concatenated_array[lengthB + n: lengthB + 2 * n]
        return bstart, df_lambda, df_q

    best, df_lambdaest, df_qest = array_to_ndarrays(est_SVAR['x'], np.size(options['bstart']), n)

    b_est = best
    out_SVAR['b_est'] = b_est

    B_est = SVAR.get_BMatrix(b_est, restrictions=options['restrictions'], whiten=options['whiten'],
                             blocks=options['blocks'])

    out_SVAR['B_est'] = B_est

    e = SVAR.innovation(u, SVAR.get_BVector(B_est, restrictions=options['restrictions']),
                        restrictions=options['restrictions'],
                        blocks=options['blocks'])
    out_SVAR['e'] = e

    Omega_all = SVAR.SVARutil.get_Omega(e)
    out_SVAR['Omega_all'] = Omega_all
    omega = SVAR.SVARutil.get_Omega_Moments(e)
    out_SVAR['omega'] = omega

    out_SVAR['loss'] = est_SVAR['fun']

    V_est = ML_avar(options['n'])
    out_SVAR['Avar_est'] = V_est

    if options['printOutput']:
        SVAR.estOutput.print_out(n, T, out_SVAR)

    return out_SVAR












