from scipy import optimize as opt 
import numpy as np
import pandas as pd
import SVAR
import SVAR.SVARutilGMM
import SVAR.estOutput
import SVAR.estimatorPML
import SVAR.estimatorGMM
import SVAR.estimatorGMMW
import SVAR.estimatorGMMWF
import SVAR.estimatorCholesky 
import SVAR.estimatorML
from scipy import optimize
import copy
from scipy.optimize import fmin_slsqp

def SVARest(u, estimator='GMM', options=dict(), prepOptions=dict(), prepared=False):
    if estimator == 'GMM':
        if not (prepared):
            prepOptions['estimator'] = 'GMM'
            options = SVAR.estimatorGMM.prepareOptions(u=u, **prepOptions)

        def optimize_this(options):
            this_lossGMM = lambda b_vec: SVAR.estimatorGMM.loss(u, b_vec,
                                                             restrictions=options['restrictions'],
                                                             moments=options['moments'],
                                                             moments_powerindex=options['moments_powerindex'],
                                                             W=options['W'] )

            if options['lambd'] != 0:


                this_loss = lambda b_vec: this_lossGMM(b_vec ) +  \
                                          SVAR.SVARutilGMM.penalty(options['lambd'],
                                              options['weights'], b_vec,
                                             [], options['NormalizeA'], type=options['PenaltyType'],
                                             restrictions=options['SoftRestrictions'],
                                             blocks=options['blocks'], whiten=False)
            else:
                this_loss = lambda b_vec: this_lossGMM(b_vec )

            this_grad = lambda b_vec: SVAR.estimatorGMM.gradient(u, b_vec,
                                                                 Jacobian=options['Jacobian'],
                                                                 W=options['W'],
                                                                 restrictions=options['restrictions'],
                                                                 moments=options['moments'],
                                                                 moments_powerindex=options['moments_powerindex'])
            # this_grad = []

            optim_start = options['bstart']
            if np.shape(optim_start)[0] == 0:
                # raise ValueError('No free parameters.')
                ret = dict()
                ret['x'] = np.array([])
                ret['fun'] = this_loss(ret['x'])
            else:

                cons = list()
                if not(np.array(options['Bcetenter']).size == 0):
                    Bstart = np.matmul(options['Bcetenter'], np.linalg.inv(np.diag(np.linalg.norm(options['Bcetenter'],axis=0))))

                    def constraint(b):
                        B = SVAR.get_BMatrix(b, whiten=False)
                        Bscaled = np.matmul(B, np.linalg.inv(np.diag(np.linalg.norm(B,axis=0))))
                        Brecentered = np.matmul(np.linalg.inv(Bstart),Bscaled)
                        Bfin = np.matmul(Brecentered, np.linalg.inv(np.diag(np.linalg.norm(Brecentered, axis=0))))


                        Bfin = Bfin - np.diag(Bfin)

                        Bfin[np.triu_indices(np.size(Bfin, 1), k=1)] = 0
                        Bfin = -np.transpose(Bfin)

                        out = Bfin[np.triu_indices(np.size(Bfin, 1), k=1)]

                        # if np.abs(np.linalg.det(B)) > 1e-5:
                        #     out = np.append(out, 1)
                        # else:
                        #     out = np.append(out, 0)


                        return out


                    cons.append({'type': 'ineq', 'fun': constraint})

                optionsopt = {
                     # 'maxiter': 50000,         # Limit maximum iterations
                     #    'ftol': 1e-9,            # Tolerance for the relative error in the objective function
                     #    'eps': 1e-11,             # Step size for numerical approximation of gradients
                     #    'disp': True,            # Print convergence messages during optimization
                     #    'iprint': 1,             # Verbosity level (0 - silent, 1 - moderate, 2 - high)
                    }

                ret_tmp = opt.minimize(this_loss, optim_start , method= options['method'],
                                       constraints=cons,
                                       options=optionsopt, jac=this_grad)

                # ret_tmp = opt.minimize(this_loss, optim_start, method=options['method'],
                #                        options=optionsopt)


                # ret_tmp = opt.minimize(this_loss, optim_start, method='L-BFGS-B', jac=this_grad,
                #              bounds=None,
                #              tol=None,
                #              callback=None,
                #              options={'disp': False,
                #                       'maxcor': 10,
                #                       'ftol': 2.220446049250313e-09,
                #                       'gtol': 1e-09,
                #                       'eps': 1e-12,
                #                       'maxfun': 15000,
                #                       'maxiter': 15000,
                #                       'iprint': - 1,
                #                       'maxls': 40,
                #                       'finite_diff_rel_step': None})

                if ret_tmp['success']==False:
                    print('Optimization failed')
                    print(ret_tmp['message'])
                ret = dict()
                ret['x'] = ret_tmp.x
                ret['fun'] = ret_tmp.fun

                if ret['fun'] < 0:
                    raise ValueError("Error: Negative loss")
            return ret

        est_SVAR = optimize_this(options)

        for k in range(1, options['kstep']):
            bstart = est_SVAR['x']
            options['W'] = SVAR.SVARutilGMM.get_W_opt(u, b=bstart, restrictions=options['restrictions'],
                                                      moments=options['moments'],
                                                      Wpara=options['Wpara'],
                                                      S_func=options['S_func'])
            est_SVAR = optimize_this(options)


        out_SVAR = SVAR.estimatorGMM.SVARout(est_SVAR, options, u)

    elif estimator == 'ML':
        if not (prepared):
            prepOptions['estimator'] = 'ML'
            options = SVAR.estimatorML.prepareOptions(u=u, **prepOptions)

        def optimize_this(options):
            this_lossML = lambda b_vec,df_lambda,df_q: SVAR.estimatorML.LogLike_st(u, b_vec,   df_lambda, df_q,
                                                                    options['restrictions'],
                                                             blocks=options['blocks'],whiten=options['whiten'])

            lengthB = np.size(options['bstart'] )

            this_loss = lambda theta_vec: -this_lossML( *array_to_ndarrays(theta_vec,lengthB,options['n']) )

            def array_to_ndarrays(concatenated_array,lengthB,n):
                bstart = concatenated_array[:lengthB]
                df_lambda = concatenated_array[lengthB: lengthB+n]
                df_q = concatenated_array[lengthB+n: lengthB+2*n]
                return bstart, df_lambda, df_q

            optim_start = np.concatenate([options['bstart'], options['df_lambda'], options['df_q']])
            if np.shape(options['bstart'])[0] == 0:
                # raise ValueError('No free parameters.')
                ret = dict()
                ret['x'] = np.array([])
                ret['fun'] = this_loss(ret['x'])
            else:

                # optionsOpt = {'disp': True }

                # Define your bounds for each parameter
                bounds_b_vec = [(None, None)]*lengthB
                bounds_df_lambda = [(-0.9999, 0.9999)]*options['n']
                bounds_df_q = [(2.0001, 10000)]*options['n']

                # Combine bounds for all parameters
                bounds =  bounds_b_vec + bounds_df_lambda + bounds_df_q

                ret_tmp = opt.minimize(this_loss, optim_start, method='L-BFGS-B',   bounds=bounds )


                if ret_tmp['success']==False:
                    print('Optimization failed')
                    print(ret_tmp['message'])
                ret = dict()
                ret['x'] = ret_tmp.x
                ret['fun'] = ret_tmp.fun

                if ret['fun'] < 0:
                    raise ValueError("Error: Negative loss")
            return ret

        est_SVAR = optimize_this(options)

        out_SVAR = SVAR.estimatorML.SVARout(est_SVAR, options, u)



    elif estimator =='CSUE':
        if not (prepared):
            prepOptions['estimator'] = 'CSUE'
            options = SVAR.estimatorGMM.prepareOptions(u=u, **prepOptions)

        def optimize_this(options):

            # def get_d(b_vec, u, options):
            #     innovations = SVAR.innovation(u, b_vec, blocks=options['blocks'])
            #     squared_sum = np.sum(np.power(innovations, 2), axis=0)
            #     return np.divide(1, np.sqrt(squared_sum / options['T']))

            get_d = lambda b_vec: np.divide(1,np.sqrt(np.sum(np.power(SVAR.innovation(u, b_vec, restrictions=options['restrictions'] ,blocks=options['blocks'] ), 2),axis=0)/ options['T']))
            # def get_dprod(b_vec, u, options):
            #     d = get_d(b_vec, u, options)
            #     return np.prod(np.power(d, options['moments']), axis=1)
            get_dprod = lambda b_vec: np.prod(np.power(get_d(b_vec), options['moments'] ),axis=1)

            # def Wupd(b_vec, u, options):
            #     dprod = get_dprod(b_vec, u, options)
            #     return np.matmul(np.matmul(np.diag(dprod), options['W']), np.diag(dprod))

            Wupd = lambda b_vec: np.matmul(np.matmul(  np.diag( get_dprod(b_vec)  ) ,options['W']), np.diag( get_dprod(b_vec)  ) )

            this_lossGMM= lambda b_vec:   SVAR.estimatorGMM.loss(u, b_vec,
                                              restrictions=options['restrictions'],
                                              moments=options['moments'],
                                              moments_powerindex=options['moments_powerindex'],
                                              W=Wupd(b_vec ),
                                              blocks=options['blocks'])


            # this_lossGMM = lambda b_vec: SVAR.estimatorGMM.loss(u, b_vec,
            #                                                  restrictions=options['restrictions'],
            #                                                  moments=options['moments'],
            #                                                  moments_powerindex=options['moments_powerindex'],
            #                                                  W=Wupd(b_vec,  options),
            #                                                  blocks=options['blocks'] )

            if options['Wpara'] == 'Independent':
                this_grad = lambda b_vec: SVAR.estimatorGMM.gradient_scalecont(u, b_vec, options['Jacobian'], Wupd,
                                                                               options['restrictions'],
                                                                               options['moments'],
                                                                               options['moments_powerindex'])
            else:
                this_grad = []

            if options['lambd'] != 0:
                if options['UseVarpen']:
                    this_varpen = lambda b_vec: np.sum(np.power(np.var(SVAR.innovation(u, b_vec, restrictions=options['restrictions'], blocks=options['blocks']),axis=0)-1,2)) / options['n']
                else:
                    this_varpen = lambda b_vec: 1

                this_loss = lambda b_vec: this_lossGMM(b_vec ) +  \
                                          SVAR.SVARutilGMM.penalty(options['lambd'],
                                              options['weights'],   b_vec,
                                             [], options['NormalizeA'], type=options['PenaltyType'],
                                             SoftRestrictions=options['SoftRestrictions'], restrictions=options['restrictions'],
                                             blocks=options['blocks'],  whiten=False) + \
                                          this_varpen(b_vec )


            else:
                this_loss = lambda b_vec: this_lossGMM(b_vec )



            if options['Wpara'] == 'Independent':
                if options['lambd'] != 0:

                    if options['UseVarpen']:
                        def compute_varpen_gradient(u, b, restrictions, options):
                            # Compute the gradient of the variance penalty term
                            h = 1e-8
                            grad = np.zeros_like(b)
                            for i in range(len(b)):
                                b_plus = b.copy()
                                b_plus[i] += h
                                varpen_plus = np.sum(
                                    np.power(np.var(SVAR.innovation(u, b_plus, restrictions=restrictions), axis=0) - 1,
                                             2))
                                varpen_minus = np.sum(
                                    np.power(np.var(SVAR.innovation(u, b, restrictions=restrictions), axis=0) - 1, 2))
                                grad[i] = (varpen_plus - varpen_minus) / (h * options['n'])
                            return grad

                        this_grad = lambda b_vec: SVAR.estimatorGMM.gradient_scalecont(u, b_vec, options['Jacobian'],
                                                                                       Wupd,
                                                                                       options['restrictions'],
                                                                                       options['moments'],
                                                                                       options[
                                                                                           'moments_powerindex']) + 2 * \
                                                  options['lambd'] * options['weights'] * b_vec + compute_varpen_gradient(u, b_vec,  options['restrictions'], options)

                    else:
                        this_grad = lambda b_vec: SVAR.estimatorGMM.gradient_scalecont(u, b_vec, options['Jacobian'], Wupd,
                                                                               options['restrictions'],
                                                                               options['moments'],
                                                                               options['moments_powerindex']) + 2 * options['lambd'] * options['weights'] * b_vec

                else:
                    this_grad = lambda b_vec: SVAR.estimatorGMM.gradient_scalecont(u, b_vec, options['Jacobian'], Wupd,
                                                                               options['restrictions'],
                                                                               options['moments'],
                                                                               options['moments_powerindex'])
            else:
                this_grad = []



            optim_start = options['bstart']
            if np.shape(optim_start)[0] == 0:
                # raise ValueError('No free parameters.')
                ret = dict()
                ret['x'] = np.array([])
                ret['fun'] = this_loss(ret['x'])
            else:
                # ret_tmp = opt.minimize(this_loss , optim_start, method='L-BFGS-B', jac=this_grad )

                cons = list()
                if not(np.array(options['Bcetenter']).size == 0):
                    Bstart = np.matmul(options['Bcetenter'], np.linalg.inv(np.diag(np.linalg.norm(options['Bcetenter'],axis=0))))

                    def constraint(b):
                        B = SVAR.get_BMatrix(b, restrictions=options['restrictions'], whiten=False)
                        Bscaled = np.matmul(B, np.linalg.inv(np.diag(np.linalg.norm(B,axis=0))))
                        Brecentered = np.matmul(np.linalg.inv(Bstart),Bscaled)
                        Bfin = np.matmul(Brecentered, np.linalg.inv(np.diag(np.linalg.norm(Brecentered, axis=0))))


                        Bfin = Bfin - np.diag(Bfin)

                        Bfin[np.triu_indices(np.size(Bfin, 1), k=1)] = 0
                        Bfin = -np.transpose(Bfin)

                        out = Bfin[np.triu_indices(np.size(Bfin, 1), k=1)]

                        # if np.abs(np.linalg.det(B)) > 1e-5:
                        #     out = np.append(out, 1)
                        # else:
                        #     out = np.append(out, 0)


                        return out


                    cons.append({'type': 'ineq', 'fun': constraint})

                optionsopt = {
                     'maxiter': 200000,     # Further increase maximum iterations
                    'ftol': 1e-12,         # Lower tolerance for the relative error
                    # 'gtol': 1e-12,         # Tolerance for termination by the norm of the gradient
                    'eps': 1e-13,          # Smaller step size for numerical approximation of gradients
                    'disp': False,          # Print convergence messages during optimization
                    # 'iprint': 2,           # Increase verbosity level for high-level output
                    'finite_diff_rel_step': 1e-8,  # Relative step size for finite difference gradients
                    #'maxls': 40,           # Maximum number of line search steps
                   # 'finite_diff_bounds': (1e-6, 1e-4)  # Finite difference bounds for gradient approximation
                    }

                ret_tmp = opt.minimize(this_loss, optim_start , method= options['method'],
                                       constraints=cons, jac=this_grad,
                                        options=optionsopt)



                # ret_tmp = opt.minimize(this_loss, optim_start, method=options['method'],
                #                        bounds=None,
                #                          tol=None,
                #                          callback=None,
                #                          options={'disp': False,
                #                                   'maxcor': 10,
                #                                   'ftol': 2.220446049250313e-09,
                #                                   'gtol': 1e-09,
                #                                   'eps': 1e-12,
                #                                   'maxfun': 15000,
                #                                   'maxiter': 15000,
                #                                   'iprint': - 1,
                #                                   'maxls': 40,
                #                                   'finite_diff_rel_step': None})

                if ret_tmp['success'] == False:
                    print('Optimization failed')
                    print(ret_tmp['message'])

                    if "Inequality constraints incompatible" in ret_tmp['message']:
                        print('Retrying with new starting value')
                        new_start = SVAR.get_BVector(options['Bcetenter'], restrictions=options['restrictions'], whiten=False)
                        ret_tmp = opt.minimize(this_loss, new_start, method=options['method'],
                                            constraints=cons, jac=this_grad,
                                            options=optionsopt)
                        
                        if ret_tmp['success'] == False:
                            print('Optimization failed again')
                            print(ret_tmp['message'])
                        else:
                            print('Optimization succeeded with new starting value')

                ret = dict()
                ret['x'] = ret_tmp.x
                ret['fun'] = ret_tmp.fun


            return ret

        est_SVAR = optimize_this(options)


        for k in range(1, options['kstep']):
            bstart = est_SVAR['x']
            options['W'] = SVAR.SVARutilGMM.get_W_opt(u, b=bstart, restrictions=options['restrictions'],
                                                      moments=options['moments'],
                                                      Wpara=options['Wpara'],
                                                      S_func=options['S_func'])
            est_SVAR = optimize_this(options)

        out_SVAR = SVAR.estimatorGMM.SVARout(est_SVAR, options, u)

    elif estimator == 'CUE':
        if not (prepared):
            prepOptions['estimator'] = 'CUE'
            options = SVAR.estimatorGMM.prepareOptions(u=u, **prepOptions)

        def optimize_this(options):
            this_loss = lambda b_vec: SVAR.estimatorGMM.loss(u, b_vec,
                                                             restrictions=options['restrictions'],
                                                             moments=options['moments'],
                                                             moments_powerindex=options['moments_powerindex'],
                                                             W=SVAR.SVARutilGMM.get_W_opt(u, b=b_vec,
                                                                                          restrictions=options[
                                                                                               'restrictions'],
                                                                                          moments=options['moments'],
                                                                                          Wpara=options['Wpara'],
                                                                                          S_func=options['S_func']))

            if options['Wpara'] == 'Independent':
                this_grad = lambda b_vec: SVAR.estimatorGMM.gradient_cont(u, b_vec,
                                                                          Jacobian=options['Jacobian'],
                                                                          W=SVAR.SVARutilGMM.get_W_opt(u, b=b_vec,
                                                                                                       restrictions=
                                                                                                       options[
                                                                                                           'restrictions'],
                                                                                                       moments=options[
                                                                                                           'moments'],
                                                                                                       Wpara=options[
                                                                                                           'Wpara'],
                                                                                                       S_func=options[
                                                                                                           'S_func']),
                                                                          restrictions=options['restrictions'],
                                                                          moments=options['moments'],
                                                                          moments_powerindex=options[
                                                                              'moments_powerindex'],
                                                                          Sdel_func=options['Sdel_func'])
            else:
                this_grad = []


            optim_start = options['bstart']
            if np.shape(optim_start)[0] == 0:
                # raise ValueError('No free parameters.')
                ret = dict()
                ret['x'] = np.array([])
                ret['fun'] = this_loss(ret['x'])
            else:
                # ret_tmp = opt.minimize(this_loss, optim_start, method='L-BFGS-B', jac=this_grad,options={'disp': True})
                ret_tmp = opt.minimize(this_loss, optim_start, method='L-BFGS-B', jac=this_grad )
                if ret_tmp['success']==False:
                    print('Optimization failed')
                    print(ret_tmp['message'])
                ret = dict()
                ret['x'] = ret_tmp.x
                ret['fun'] = ret_tmp.fun


                if ret['fun'] < 0:
                    raise ValueError("Error: Negative loss")
            return ret

        est_SVAR = optimize_this(options)

        options['W'] = SVAR.SVARutilGMM.get_W_opt(u, b=est_SVAR['x'], restrictions=options['restrictions'],
                                                  moments=options['moments'],
                                                  Wpara=options['Wpara'],
                                                  S_func=options['S_func'])
        out_SVAR = SVAR.estimatorGMM.SVARout(est_SVAR, options, u)

    elif estimator == 'GMM_W':
        if not (prepared):
            prepOptions['estimator'] = 'GMM_W'
            options = SVAR.estimatorGMMW.prepareOptions(u=u, **prepOptions)
        z, options['V'] = SVAR.do_whitening(u, white=True)

        def optimize_this(options):
            this_loss = lambda b_vec: SVAR.estimatorGMMW.loss(z, b_vec,
                                                              restrictions=options['restrictions'],
                                                              moments=options['moments'],
                                                              moments_powerindex=options['moments_powerindex'],
                                                              W=options['W'],
                                                              blocks=options['blocks'])
            #this_grad =  lambda b_vec: optimize.approx_fprime(b_vec, this_loss, epsilon=1.4901161193847656e-08)
            this_grad =[]

            optim_start = options['bstart']
            if np.shape(optim_start)[0] == 0:
                # raise ValueError('No free parameters.')
                ret = dict()
                ret['x'] = np.array([])
                ret['fun'] = this_loss(ret['x'])
            else:
                ret_tmp = opt.minimize(this_loss, optim_start, method='L-BFGS-B', jac=this_grad)
                if ret_tmp['success']==False:
                    print('Optimization failed')
                    print(ret_tmp['message'])
                ret = dict()
                ret['x'] = ret_tmp.x
                ret['fun'] = ret_tmp.fun

                if ret['fun'] < 0:
                    raise ValueError("Error: Negative loss")

            return ret

        est_SVAR = optimize_this(options)

        out_SVAR = SVAR.estimatorGMMW.SVARout(est_SVAR, options, u)

    elif estimator == 'GMM_WF':
        if not (prepared):
            prepOptions['estimator'] = 'GMM_WF'
            options = SVAR.estimatorGMMWF.prepareOptions(u=u, **prepOptions)
        z, options['V'] = SVAR.do_whitening(u, white=True)

        def optimize_this(options):
            if options['moments_MeanIndep']:
                this_loss = lambda b_vec: SVAR.estimatorGMMWF.loss(z, b_vec,
                                                                   restrictions=options['restrictions'],
                                                                   moments=options['moments'],
                                                                   moments_powerindex=options['moments_powerindex'],
                                                                   blocks=options['blocks'])-\
                                        SVAR.estimatorGMMWF.loss_MIcorrection(z,b_vec,restrictions=options['restrictions'],
                                                                  moments=options['moments_MIcorrection'],
                                                                  moments_powerindex=options['moments_MIcorrection_powerindex'], blocks=options['blocks'])
            else:
                this_loss = lambda b_vec: SVAR.estimatorGMMWF.loss(z, b_vec,
                                                                   restrictions=options['restrictions'],
                                                                   moments=options['moments'],
                                                                   moments_powerindex=options['moments_powerindex'],
                                                                   blocks=options['blocks'])
            #this_grad =  lambda b_vec: optimize.approx_fprime(b_vec, this_loss, epsilon=1.4901161193847656e-08)
            this_grad =[]

            optim_start = options['bstart']
            if np.shape(optim_start)[0] == 0:
                # raise ValueError('No free parameters.')
                ret = dict()
                ret['x'] = np.array([])
                ret['fun'] = this_loss(ret['x'])
            else:

                ret_tmp = opt.minimize(this_loss, optim_start, method='L-BFGS-B', jac=this_grad)

                if ret_tmp['success'] == False:
                    print('Optimization failed')
                    print(ret_tmp['message'])
                ret = dict()
                ret['x'] = ret_tmp.x
                ret['fun'] = ret_tmp.fun
                # if ret['fun'] < 0:
                #     raise ValueError("Error: Negative loss")
            return ret

        est_SVAR = optimize_this(options)

        out_SVAR = SVAR.estimatorGMMWF.SVARout(est_SVAR, options, u)

    elif estimator == 'PML':
        if not (prepared):
            prepOptions['estimator'] = 'PML'
            options = SVAR.estimatorPML.prepareOptions(u=u, **prepOptions)
        z, options['V'] = SVAR.do_whitening(u, white=True)

        def optimize_this(options):
            this_loss = lambda b_vec: SVAR.estimatorPML.LogLike_t(z, b_vec, options['df_t'], blocks=options['blocks'],
                                                                  whiten=options['whiten'])
            this_grad = []

            optim_start = options['bstart']
            if np.shape(optim_start)[0] == 0:
                # raise ValueError('No free parameters.')
                ret = dict()
                ret['x'] = np.array([])
                ret['fun'] = this_loss(ret['x'])
            else:
                ret_tmp = opt.minimize(this_loss, optim_start, method='L-BFGS-B', jac=this_grad)
                if ret_tmp['success'] == False:
                    print('Optimization failed')
                    print(ret_tmp['message'])
                ret = dict()
                ret['x'] = ret_tmp.x
                ret['fun'] = ret_tmp.fun


            return ret

        est_SVAR = optimize_this(options)

        out_SVAR = SVAR.estimatorPML.SVARout(est_SVAR, options, u)

    elif estimator == 'RidgeBWF':
        if not (prepared):
            prepOptions['estimator'] = 'RidgeBWF'
            options = SVAR.estimatorRidgeBWF.prepareOptions(u=u, **prepOptions)
        z, options['V'] = SVAR.do_whitening(u, white=True)

        def optimize_this(options):
            this_lossGMM = lambda b_vec: SVAR.estimatorRidgeBWF.loss(z, b_vec,
                                                                  restrictions=options['restrictions'],
                                                                  moments=options['moments'],
                                                                  moments_powerindex=options['moments_powerindex'],
                                                                  blocks=options['blocks'])

            if options['moments_MeanIndep']:
                this_loss = lambda b_vec: this_lossGMM(b_vec) + \
                                          SVAR.SVARutilGMM.penalty(options['lambd'], options['weights'], b_vec,
                                                                         options['V'],options['NormalizeA'], type=options['PenaltyType'],
                                                                         restrictions=options['SoftRestrictions'],
                                                                  blocks=options['blocks'],whiten=True)-\
                                        SVAR.estimatorGMMWF.loss_MIcorrection(z,b_vec,restrictions=[],
                                                                  moments=options['moments_MIcorrection'],
                                                                  moments_powerindex=options['moments_MIcorrection_powerindex'], blocks=options['blocks'])
            else:
                this_loss = lambda b_vec: this_lossGMM(b_vec) + \
                                          SVAR.SVARutilGMM.penalty(options['lambd'], options['weights'], b_vec,
                                                                         options['V'], options['NormalizeA'], type=options['PenaltyType'],
                                                                         SoftRestrictions=options['SoftRestrictions'], restrictions=options['restrictions'],
                                                                  blocks=options['blocks'],whiten=True)


            optim_start = options['bstart']
            if np.shape(optim_start)[0] == 0:
                # raise ValueError('No free parameters.')
                ret = dict()
                ret['x'] = np.array([])
                ret['fun'] = this_loss(ret['x'])
            else:


                cons = list()
                if not(np.array(options['Bcetenter']).size == 0):
                    Bstart = np.matmul(options['Bcetenter'], np.linalg.inv(np.diag(np.linalg.norm(options['Bcetenter'],axis=0))))

                    def constraint(b):
                        B = SVAR.get_BMatrix(b, whiten=True)
                        B = np.matmul(options['V'], B)
                        Bscaled = np.matmul(B, np.linalg.inv(np.diag(np.linalg.norm(B,axis=0))))
                        Brecentered = np.matmul(np.linalg.inv(Bstart),Bscaled)
                        Bfin = np.matmul(Brecentered, np.linalg.inv(np.diag(np.linalg.norm(Brecentered, axis=0))))


                        Bfin = Bfin - np.diag(Bfin)

                        Bfin[np.triu_indices(np.size(Bfin, 1), k=1)] = 0
                        Bfin = -np.transpose(Bfin)

                        return Bfin[np.triu_indices(np.size(Bfin, 1), k=1)]
                    cons.append({'type': 'ineq', 'fun': constraint})



                ret_tmp = opt.minimize(this_loss, optim_start, method = options['method'],
                                       constraints=cons,
                                       options={'maxiter': 5000})





                if ret_tmp['success'] == False:
                    print('Optimization failed')
                    print(ret_tmp['message'])
                ret = dict()
                ret['x'] = ret_tmp.x
                ret['fun'] = ret_tmp.fun
                # if ret['fun'] < 0:
                #     raise ValueError("Error: Negative loss")
            return ret

        est_SVAR = optimize_this(options)

        out_SVAR = SVAR.estimatorRidgeBWF.SVARout(est_SVAR, options, u)


    elif estimator == 'Cholesky':
        out_SVAR = SVAR.estimatorCholesky.get_B_Cholesky(u)

    else:
        print('Unknown estimator')

    return out_SVAR

