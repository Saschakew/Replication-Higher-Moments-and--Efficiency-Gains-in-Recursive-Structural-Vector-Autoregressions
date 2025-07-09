import pickle
import numpy as np
import pandas as pd
import os
import SVAR.SVARutil
import matplotlib.pyplot as plt

pd.set_option('display.max_rows', 170)
pd.set_option('display.max_columns', 170)
pd.set_option('display.width', 170)

# Specify version
version = "MC_Efficiency_Sim1"


LatexB= False
LatexBcov= False
LatexWald= False
LatexWald2= False
show_mean = False
show_var = False


showallestimators = False
useestimators = list([   'Cholesky',
                         'CUE-MI',
                         'CUE-MI-Biv' ])
# ['Cholesky', 'GMM', 'GMM-Biv', 'CUE-MI', 'CUE-MI-Biv', 'GMM-MI Wopt', 'GMM-MI Wopt-Biv']
useestimators_names = [r'Cholesky $\hat{B}_{\mathbf{2}}$',
  r'CUE $\hat{B}_{\mathbf{2} + \mathbf{SK}}$',
  r'CUE bivariate $\hat{B}_{\mathbf{2} + \tilde{\mathbf{S}} \tilde{\mathbf{K}}}$']


gen_Latex = False

b_save = dict()
q90_save = dict()
q10_save = dict()
rejection_save = dict()
var_save = dict()
for estimator in useestimators:
    b_save[estimator] = np.zeros(10)
    q90_save[estimator] = np.zeros(10)
    q10_save[estimator] = np.zeros(10)
    var_save[estimator] = np.zeros(10)
    rejection_save[estimator] = np.zeros([9,10])




# Load all files of version
path = os.path.join("MCResults", version)
# Load collected data or create empty Data file
path_VersionData = os.path.join(path, str(version + ".data"))
try:
    with open(path_VersionData, 'rb') as filehandle:
        # read the data as binary data stream
        df = pickle.load(filehandle)
except:
    print("Run MC_collect_data.py")

def get_latex_B(estimator,thisMean, thisMed, thisQIQ , thisSD):
    latex_string = estimator + ' & '

    latex_string += '$' + str(thisMean[12]) + '$ & '
    latex_string += '$' + str(thisMed[12]) + '$ & '
    latex_string += '$' + str(thisQIQ[12]) + '$ & '
    latex_string += '$' + str(thisSD[12]) + '$ & '

    latex_string += '$' + str(thisMean[-1]) + '$ & '
    latex_string += '$' + str(thisMed[-1]) + '$ & '
    latex_string += '$' + str(thisQIQ[-1]) + '$ & '
    latex_string += '$' + str(thisSD[-1]) + '$ & '

    latex_string += '$' + str(thisMean[3]) + '$ & '
    latex_string += '$' + str(thisMed[3]) + '$ & '
    latex_string += '$' + str(thisQIQ[3]) + '$ & '
    latex_string += '$' + str(thisSD[3]) + '$   '


    print(latex_string)

def get_latex_Wald(estimator,thisWaldIndB0, thisWaldIndBrec, thisWaldIndB14,thisWaldUncB0, thisWaldUncBrec, thisWaldUncB14):
    latex_string = estimator + ' & '

    latex_string += '$' + str(thisWaldIndB0) + '$ & '
    latex_string += '$' + str(thisWaldUncB0) + '$ & '

    latex_string += '$' + str(thisWaldIndBrec) + '$ & '
    latex_string += '$' + str(thisWaldUncBrec) + '$ & '

    latex_string += '$' + str(thisWaldIndB14) + '$ & '
    latex_string += '$' + str(thisWaldUncB14) + '$   '

    print(latex_string)

def get_latex_BCov(estimator,thisCov68, thisCov90,thisCov268, thisCov290):
    latex_string =  estimator + ' & '

    #latex_string += '$' + str(thisCov68[12]) + '$ & '
    latex_string += '$' + str(thisCov90[12]) + '$ & '
    #latex_string += '$' + str(thisCov268[12]) + '$ & '
    latex_string += '$' + str(thisCov290[12]) + '$ & '

    #latex_string += '$' + str(thisCov68[-1]) + '$ & '
    latex_string += '$' + str(thisCov90[-1]) + '$ & '
    #latex_string += '$' + str(thisCov268[-1]) + '$ & '
    latex_string += '$' + str(thisCov290[-1]) + '$ & '

    #latex_string += '$' + str(thisCov68[3]) + '$ & '
    latex_string += '$' + str(thisCov90[3]) + '$ &  '
    #latex_string += '$' + str(thisCov268[3]) + '$ & '
    latex_string += '$' + str(thisCov290[3]) + '$   '

    print(latex_string)

def get_latex_Wald2(estimator,thisB41_30,  thisB41_35, thisB41_40,thisB41_45, thisB41_5,thisB41_55 , thisB41_60 , thisB41_65 , thisB41_70  ):
    latex_string = estimator + ' & '

    latex_string += '$' + str(thisB41_30 ) + '$ & '
    latex_string += '$' + str(thisB41_35 ) + '$ & '
    latex_string += '$' + str(thisB41_40 ) + '$ & '
    latex_string += '$' + str(thisB41_45 ) + '$ & '
    latex_string += '$' + str(thisB41_5 ) + '$ & '
    latex_string += '$' + str(thisB41_55 ) + '$ & '
    latex_string += '$' + str(thisB41_60 ) + '$ & '
    latex_string += '$' + str(thisB41_65 ) + '$ & '
    latex_string += '$' + str(thisB41_70 ) + '$ & '

    print(latex_string)

def get_latex(this_std, this_bias):
    n = int(np.sqrt(np.shape(this_std)))
    counter = 0
    latex_string = '$\\begin{bmatrix}'
    for i in range(n):
        for j in range(n):
            latex_string += '\\underset{(' + str(this_std[counter]) + ')}{' + str(this_bias[counter]) + '}'
            if j < n - 1:
                latex_string += ' & '
            counter += 1
        latex_string += ' \\\\ '

    latex_string += '\\end{bmatrix}$ '
    return latex_string

def get_latex_med(this_q25,this_q75, this_med):
    n = int(np.sqrt(np.shape(this_std)))
    counter = 0
    latex_string = '$\\begin{bmatrix}'
    for i in range(n):
        for j in range(n):
            latex_string += '\\underset{ ' + str(this_q25[counter]) + '/' + str(this_q75[counter]) + ' }{' + str(this_med[counter]) + '}'
            if j < n - 1:
                latex_string += ' & '
            counter += 1
        latex_string += ' \\\\ '

    latex_string += '\\end{bmatrix}$ '
    return latex_string

N = np.unique(df.n)
T = np.unique(df["T"])

for n in N:
    for it, t in enumerate(T):
        df_this = df[df.n == n]
        df_this = df_this[df_this["T"] == t]
        numMC_n = np.size(df_this[['n']])

        if numMC_n != 0:
            # B_true = df_this.B[df_this.index[0]]
            B_true = df_this.B_true[df_this.index[0]]
            b_true = SVAR.SVARutil.get_BVector(B_true)
            B_true_rel = np.divide(B_true,np.diag(B_true))



            if showallestimators:
                # # Calculate Bias at each simulation
                estimators = np.unique([df['estimators']])
                try:
                    if type(estimators[0]) == list:
                        allsize = np.zeros(np.size(estimators), dtype=int)
                        for estidxthis, estimators_this in enumerate(estimators):
                            allsize[estidxthis] = np.size(estimators_this)
                        estimators = estimators[np.argmax(allsize)]
                except:
                    pass
            else:
                estimators = useestimators


            bias = dict()
            coverageUnc68 = dict()
            coverageInd68 = dict()
            coverageUnc90 = dict()
            coverageInd90 = dict()
            JpvalueSind = dict()
            JpvalueSunc = dict()
            WaldB0PUnc = dict()
            WaldB0PInd = dict()
            WaldRecPInd = dict()
            WaldRecPUnc = dict()
            Waldb14PInd = dict()
            Waldb14PUnc = dict()

            Waldb41PInd30 = dict()
            Waldb41PInd35 = dict()
            Waldb41PInd40 = dict()
            Waldb41PInd45 = dict()
            Waldb41PInd = dict()
            Waldb41PInd55 = dict()
            Waldb41PInd60 = dict()
            Waldb41PInd65 = dict()
            Waldb41PInd70 = dict()


            countzeros = dict()







            countEstimators = dict()

            for idx_estimator, estimator in enumerate(estimators):
                bias[estimator] = np.full([df_this.shape[0], np.size(b_true)] ,np.nan)
                coverageUnc68[estimator] = np.full([df_this.shape[0], np.size(b_true)] ,np.nan)
                coverageInd68[estimator] = np.full([df_this.shape[0], np.size(b_true)] ,np.nan)
                coverageUnc90[estimator] = np.full([df_this.shape[0], np.size(b_true)] ,np.nan)
                coverageInd90[estimator] = np.full([df_this.shape[0], np.size(b_true)] ,np.nan)

                JpvalueSind[estimator] = np.full([df_this.shape[0]], np.nan)
                JpvalueSunc[estimator] = np.full([df_this.shape[0]], np.nan)
                WaldB0PUnc[estimator] = np.full([df_this.shape[0]], np.nan)
                WaldB0PInd[estimator] = np.full([df_this.shape[0]], np.nan)
                WaldRecPInd[estimator] = np.full([df_this.shape[0]], np.nan)
                WaldRecPUnc[estimator] = np.full([df_this.shape[0]], np.nan)
                Waldb14PInd[estimator] = np.full([df_this.shape[0]], np.nan)
                Waldb14PUnc[estimator] = np.full([df_this.shape[0]], np.nan)
                Waldb41PInd30[estimator] = np.full([df_this.shape[0]], np.nan)
                Waldb41PInd35[estimator] = np.full([df_this.shape[0]], np.nan)
                Waldb41PInd40[estimator] = np.full([df_this.shape[0]], np.nan)
                Waldb41PInd45[estimator] = np.full([df_this.shape[0]], np.nan)
                Waldb41PInd[estimator] = np.full([df_this.shape[0]], np.nan)
                Waldb41PInd55[estimator] = np.full([df_this.shape[0]], np.nan)
                Waldb41PInd60[estimator] = np.full([df_this.shape[0]], np.nan)
                Waldb41PInd65[estimator] = np.full([df_this.shape[0]], np.nan)
                Waldb41PInd70[estimator] = np.full([df_this.shape[0]], np.nan)

                countEstimators[estimator] = 0


                countzeros[estimator] = np.zeros((np.size(b_true)))







            k = 0
            for index, row in df_this.iterrows():
                # b_true = row['b_true']

                for estimator in estimators:
                    countEstimators[estimator] += 1
                    B_est = row[estimator]['B_est']


                    try:
                        coverageUnc68[estimator][k] = row[estimator]['coverageUnc68']
                    except:
                        0
                    try:
                        coverageInd68[estimator][k] = row[estimator]['coverageInd68']
                    except:
                        0
                    try:
                        coverageUnc90[estimator][k] = row[estimator]['coverageUnc90']
                    except:
                        0
                    try:
                        coverageInd90[estimator][k] = row[estimator]['coverageInd90']
                    except:
                        0

                    try:
                        JpvalueSind[estimator][k] =  row[estimator]['JpvalueSind']
                    except:
                        0
                    try:
                        JpvalueSunc[estimator][k] =  row[estimator]['JpvalueSunc']
                    except:
                        0


                    try:
                        WaldB0PUnc[estimator][k] = row[estimator]['WaldB0PUnc']
                    except:
                        0
                    try:
                        WaldB0PInd[estimator][k] = row[estimator]['WaldB0PInd']
                    except:
                        0

                    try:
                        WaldRecPInd[estimator][k] = row[estimator]['WaldRecPInd']
                    except:
                        0
                    try:
                        WaldRecPUnc[estimator][k] = row[estimator]['WaldRecPUnc']
                    except:
                        0

                    try:
                        Waldb14PInd[estimator][k] = row[estimator]['Waldb14PInd']
                    except:
                        0
                    try:
                        Waldb14PUnc[estimator][k] = row[estimator]['Waldb14PUnc']
                    except:
                        0

                    try:
                        Waldb41PInd30[estimator][k] = row[estimator]['Waldb41PInd30']
                        Waldb41PInd35[estimator][k] = row[estimator]['Waldb41PInd35']
                        Waldb41PInd40[estimator][k] = row[estimator]['Waldb41PInd40']
                        Waldb41PInd45[estimator][k] = row[estimator]['Waldb41PInd45']
                        Waldb41PInd[estimator][k] = row[estimator]['Waldb41PInd']
                        Waldb41PInd55[estimator][k] = row[estimator]['Waldb41PInd55']
                        Waldb41PInd60[estimator][k] = row[estimator]['Waldb41PInd60']
                        Waldb41PInd65[estimator][k] = row[estimator]['Waldb41PInd65']
                        Waldb41PInd70[estimator][k] = row[estimator]['Waldb41PInd70']
                    except:
                        0


                    bias[estimator][k, :] = SVAR.SVARutil.get_BVector(B_est) - b_true



                    # k = 1
                    b_this = SVAR.SVARutil.get_BVector(B_est)
                    countzeros[estimator] = countzeros[estimator] + (b_this == 0)




                k += 1




            print("")
            # Print output
            print("For n=", n, ", T=", t, " and ", numMC_n, " simulations")
            print("T:", t)
            print("saved")
            print(countEstimators)

            out = pd.DataFrame()
            colnames = []
            # Generate output table
            counter = 0
            for estimator in estimators:
                bias_this = bias[estimator]


                show_elem=-2
                outMean = np.round(np.nanmean(bias_this, axis=0)  + b_true, 2)
                b_save[estimator][it] = outMean[show_elem]

                out_this = [outMean]
                out = pd.concat([out, pd.DataFrame(out_this)], ignore_index=True)
                this_name = estimator + '_mean'
                colnames.append(this_name)
                counter += 1

                q10_save[estimator][it] = np.round(np.nanquantile(bias_this, 0.1, axis=0) + b_true, 2)[show_elem]
                q90_save[estimator][it] = np.round(np.nanquantile(bias_this, 0.9, axis=0) + b_true, 2)[show_elem]

                var_save[estimator][it] = np.nanmean(np.power(np.sqrt(t) * bias_this, 2), axis=0)[show_elem]

                out_this = [np.nanmean(np.power(np.sqrt(t) * bias_this, 2), axis=0)]
                out = pd.concat([out, pd.DataFrame(out_this)], ignore_index=True)
                this_name = estimator + '_var'
                colnames.append(this_name)
                counter += 1

                outMedian = np.round(np.nanmedian(bias_this, axis=0)   + b_true, 2)

                IQ = np.round(np.nanquantile(bias_this, 0.75, axis=0) - np.nanquantile(bias_this, 0.25, axis=0)  , 2)


                outQ10 = np.round(np.nanquantile(bias_this, 0.1, axis=0) + b_true, 2)


                outQ90 = np.round(np.nanquantile(bias_this, 0.9, axis=0) + b_true, 2)



                outVar =   np.round(np.nanmean(np.power(bias_this,2),axis=0), 2)


                outCovInd68 = np.round( np.nansum(coverageInd68[estimator], axis=0)/ np.shape(coverageInd68[estimator])[0] * 100 , 0)


                outCovInd90 = np.round(np.nansum(coverageInd90[estimator], axis=0) / np.shape(coverageInd68[estimator])[0] * 100, 0)


                outCovUnc68 = np.round( np.nansum(coverageUnc68[estimator], axis=0)/ np.shape(coverageInd68[estimator])[0] * 100 , 0)


                outCovUnc90 = np.round(np.nansum(coverageUnc90[estimator], axis=0) / np.shape(coverageInd68[estimator])[0] * 100, 0)


                alpha = 0.1
                out_JSind = np.round(np.sum(JpvalueSind[estimator]<alpha, axis=0) / np.shape(JpvalueSind[estimator])[0] * 100, 2)


                alpha = 0.1
                out_JSunc = np.round(np.sum(JpvalueSunc[estimator]<alpha, axis=0) / np.shape(JpvalueSunc[estimator])[0] * 100, 2)


                alpha = 0.1
                outWaldB0Pind = np.round(
                    np.sum(WaldB0PInd[estimator] < alpha, axis=0) / np.shape(WaldB0PInd[estimator])[0] * 100,
                    0)



                alpha = 0.1
                outWaldB0PUnc = np.round(
                    np.sum(WaldB0PUnc[estimator] < alpha, axis=0) / np.shape(WaldB0PUnc[estimator])[0] * 100,
                    0)



                alpha = 0.1
                outWaldRecPInd = np.round(
                    np.sum(WaldRecPInd[estimator] < alpha, axis=0) / np.shape(WaldRecPInd[estimator])[0] * 100,
                    0)



                alpha = 0.1
                outWaldRecPUnc = np.round(
                    np.sum(WaldRecPUnc[estimator] < alpha, axis=0) / np.shape(WaldRecPUnc[estimator])[0] * 100,
                    0)



                alpha = 0.1
                outWaldb14PInd = np.round(
                    np.sum(Waldb14PInd[estimator] < alpha, axis=0) / np.shape(Waldb14PInd[estimator])[0] * 100,
                    0)


                alpha = 0.1
                outWaldb14PUnc = np.round(
                    np.sum(Waldb14PUnc[estimator] < alpha, axis=0) / np.shape(Waldb14PUnc[estimator])[0] * 100,
                    0)


                alpha = 0.1
                outWaldb41PInd30 = np.round(
                    np.sum(Waldb41PInd30[estimator] < alpha, axis=0) / np.shape(Waldb41PInd30[estimator])[0] * 100,
                    0)
                outWaldb41PInd35 = np.round(
                    np.sum(Waldb41PInd35[estimator] < alpha, axis=0) / np.shape(Waldb41PInd35[estimator])[0] * 100,
                    0)
                outWaldb41PInd40 = np.round(
                    np.sum(Waldb41PInd40[estimator] < alpha, axis=0) / np.shape(Waldb41PInd40[estimator])[0] * 100,
                    0)
                outWaldb41PInd45 = np.round(
                    np.sum(Waldb41PInd45[estimator] < alpha, axis=0) / np.shape(Waldb41PInd45[estimator])[0] * 100,
                    0)
                outWaldb41PInd = np.round(
                    np.sum(Waldb41PInd[estimator] < alpha, axis=0) / np.shape(Waldb41PInd[estimator])[0] * 100,
                    0)
                outWaldb41PInd55 = np.round(
                    np.sum(Waldb41PInd55[estimator] < alpha, axis=0) / np.shape(Waldb41PInd55[estimator])[0] * 100,
                    0)
                outWaldb41PInd60 = np.round(
                    np.sum(Waldb41PInd60[estimator] < alpha, axis=0) / np.shape(Waldb41PInd60[estimator])[0] * 100,
                    0)
                outWaldb41PInd65 = np.round(
                    np.sum(Waldb41PInd65[estimator] < alpha, axis=0) / np.shape(Waldb41PInd65[estimator])[0] * 100,
                    0)
                outWaldb41PInd70 = np.round(
                    np.sum(Waldb41PInd70[estimator] < alpha, axis=0) / np.shape(Waldb41PInd70[estimator])[0] * 100,
                    0)





                print(" ")
                if LatexB:
                    print("LatexB")
                    get_latex_B(estimator,outMean, outMedian, IQ, outVar)
                if LatexBcov:
                    print("LatexBcov")
                    get_latex_BCov(estimator, outCovInd68, outCovInd90, outCovUnc68, outCovUnc90)
                if LatexWald:
                    print("LatexWald")
                    get_latex_Wald(estimator,outWaldB0Pind, outWaldRecPInd, outWaldb14PInd,   outWaldB0PUnc, outWaldRecPUnc,  outWaldb14PUnc)
                if LatexWald2:
                    print("LatexWald2")
                    get_latex_Wald2(estimator, outWaldb41PInd30, outWaldb41PInd35, outWaldb41PInd40, outWaldb41PInd45, outWaldb41PInd, outWaldb41PInd55, outWaldb41PInd60, outWaldb41PInd65, outWaldb41PInd70)
                rejection_save[estimator][:,it] = np.array([ outWaldb41PInd30, outWaldb41PInd35, outWaldb41PInd40, outWaldb41PInd45, outWaldb41PInd, outWaldb41PInd55, outWaldb41PInd60, outWaldb41PInd65, outWaldb41PInd70])


                print(' ')




            out = out.T
            colSums = np.sum(np.abs(out), axis=0)
            colSumsDat = pd.DataFrame([colSums], columns = out.columns)
            colSumsDat.rename(index={0: 'Sum abs'}, inplace=True)
            out = pd.concat([out, colSumsDat])

            # Label rows and coloums
            counter = 0
            for i in range(n):
                for j in range(n):
                    this_string = 'b(' + str(i + 1) + ',' + str(j + 1) + ')'
                    out.rename(index={counter: this_string}, inplace=True)
                    counter += 1


            out.columns = colnames
            cols = out.columns.to_list()

            print(' ')
            print(out)

            if gen_Latex == True:
                for estimator in estimators:
                    print('estimator: ', estimator)
                    est_string_std = estimator + '_var'
                    est_string_bias = estimator + '_mean'
                    this_std = out[est_string_std]
                    this_std = np.round(this_std, 2)
                    this_bias = out[est_string_bias]
                    this_bias = np.round(this_bias, 2)
                    this_string = get_latex(this_std, this_bias)
                    print(this_string)

    # Plots
    if True:


        mycolors = np.array(['black', 'blue', 'red'])

        var_avar = np.array([127,  102 ,    102])

        fig, axs = plt.subplots(1, 3, figsize=(15, 5), tight_layout=True)
        plt.subplots_adjust(left=0.03, right=0.97 ,top=0.95)
        for idx_estimator, estimator in enumerate(estimators[1:]):
            axs[idx_estimator].plot(T[:-1], b_save['Cholesky'][:-1], color=mycolors[0], label=useestimators_names[0])
            axs[idx_estimator].fill_between(T[:-1], q10_save['Cholesky'][:-1], q90_save['Cholesky'][:-1],
                                            color=mycolors[0], alpha=0.5)

            axs[idx_estimator].plot(T[:-1], b_save[estimator][:-1], color=mycolors[idx_estimator + 1],
                                    label=useestimators_names[idx_estimator+1])
            axs[idx_estimator].fill_between(T[:-1], q10_save[estimator][:-1], q90_save[estimator][:-1],
                                            color=mycolors[idx_estimator + 1], alpha=0.5)

            axs[idx_estimator].set_ylim([3.5, 6.5])
            axs[idx_estimator].set_title(" Median and quantiles ")
            axs[idx_estimator].set_xlabel('Sample size')
            axs[idx_estimator].legend()

        for idx_estimator, estimator in enumerate(estimators):
            axs[2].axhline(y=var_avar[idx_estimator], color=mycolors[idx_estimator ], linestyle='--')
            axs[2].plot(T[:-1], var_save[estimator][:-1], color=mycolors[idx_estimator ], label=useestimators_names[idx_estimator])
            axs[2].set_xlabel('Sample size')
            axs[2].set_title(" Scaled variance ")
            axs[2].legend(loc='upper center')

        fig_name = 'MCMeanPlot.pdf'
        fig.savefig(fig_name, format='pdf', dpi=1200)
        plt.show()



        rejectionvalues = np.array([3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7])
        fig, axs = plt.subplots(1, 5, figsize=(20, 5),sharey=True,tight_layout=True)
        plt.subplots_adjust(left=0.03, right=0.97 ,top=0.95)
        for idx_T, plot_t in enumerate(np.array([200, 300, 400, 500, 1000])):
            for idx_estimator, estimator in enumerate(estimators):
                axs[idx_T].axhline(y=10, color='black', linestyle='--')
                axs[idx_T].plot(rejectionvalues[1:-1], rejection_save[estimator][1:-1, plot_t == T],
                                color=mycolors[idx_estimator], label=useestimators_names[idx_estimator])
                # axs[idx_T].set_ylim([3.5, 6.5])
                axs[idx_T].set_title("Sample size: " + str(T[plot_t == T][0]))
                axs[idx_T].set_xlabel(r'$b$')
                if plot_t == 400:
                    axs[idx_T].legend(loc='upper center')
        fig_name = 'MCRejectionPlot.pdf'
        fig.savefig(fig_name, format='pdf', dpi=1200)
        plt.show()





