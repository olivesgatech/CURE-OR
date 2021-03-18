import pandas as pd
import os, sys
from dataloader import CUREORrecognitionData
from utils import *

def challenging_conditions(AWSDir, AzureDir, resultDir, common=False, topN=5):
    cureor = CUREORrecognitionData(AWSDir, AzureDir, common, topN)
    result_path = os.path.join(resultDir, 'Challenging_conditions')
    if not os.path.exists(result_path + '/Data'): os.makedirs(result_path + '/Data')
    if not os.path.exists(result_path + '/Plots'): os.makedirs(result_path + '/Plots')

    resultsAWS, resultsAzure = cureor.resultsAWS, cureor.resultsAzure
    cTypes, levels = cureor.cTypes, cureor.levels
    numObj = len(cureor.awsObj)
    numObjf = float(numObj)
    objs = 'selObj' if common == False else 'comObj'

    try:
        for results, app in zip([resultsAWS, resultsAzure], ['AWS', 'Azure']):
            color_file = '%s_%s_color_N_%d'%(app,objs,topN)
            gray_file = '%s_%s_gray_N_%d'%(app,objs,topN)
            dfNumGTcolor = pd.read_csv(os.path.join(result_path, 'Data', color_file+'.csv'), index_col=0)
            dfNumGTgray = pd.read_csv(os.path.join(result_path, 'Data', gray_file+'.csv'), index_col=0)
            print('Read in existing result files for %s'%app)

            plot_challenging_types([dfNumGTcolor, dfNumGTgray],
                                   [color_file, gray_file],
                                   result_path,
                                   cureor.cTypes)

    except IOError:
        print('Generating new result files')
        for results, app in zip([resultsAWS, resultsAzure], ['AWS', 'Azure']):
            dfNumGTcolor = pd.DataFrame(index=cTypes.values()[1:9], columns=['Original']+levels)
            dfNumGTgray = pd.DataFrame(index=cTypes.values()[10:], columns=['Original']+levels)

            for i in cTypes.keys():
                ct, ct_tmp = [], 0

                for j in range(numObj): ct_tmp += len(results[i][j])

                ct.append(ct_tmp) # Original & Level 1

                if i == 0: dfNumGTcolor.Original= ct[0]/numObjf/125*100
                elif i == 9: dfNumGTgray.Original = ct[0]/numObjf/125*100
                else:
                    levels_tmp = levels[:-1] if i in [1, 10] else levels
                    for lev in range(len(levels_tmp) - 1):
                        ct_tmp = 0
                        for j in range(numObj): 
                            ct_tmp += len(results[i][(lev + 1)*numObj + j])
                        ct.append(ct_tmp)

                    if i in range(1,9): 
                        for j in range(len(levels_tmp)):
                            dfNumGTcolor.ix[i - 1,levels_tmp[j]] = ct[j]/numObjf/125*100
                    else: 
                        for j in range(len(levels_tmp)):
                            dfNumGTgray.ix[i - 10,levels_tmp[j]] = ct[j]/numObjf/125*100


            dfNumGTcolor.to_csv(os.path.join(result_path, 'Data', '%s_%s_color_N_%d.csv'%(app,objs,topN)))
            dfNumGTgray.to_csv(os.path.join(result_path, 'Data', '%s_%s_gray_N_%d.csv'%(app,objs,topN)))

            plot_challenging_types([dfNumGTcolor, dfNumGTgray],
                                   [color_file, gray_file],
                                   result_path,
                                   cTypes)

    return

def challenging_conditions_cf(AWSDir, AzureDir, resultDir, common=False, topN=5):
    result_path = os.path.join(resultDir, 'Challenging_conditions_cf')
    if not os.path.exists(result_path + '/Data'): os.makedirs(result_path + '/Data')
    if not os.path.exists(result_path + '/Plots'): os.makedirs(result_path + '/Plots')

    cureor = CUREORrecognitionData(AWSDir, AzureDir, common=common, topN=topN, cf=True)
    cfAWS, cfAzure = cureor.cfAWS, cureor.cfAzure

    plot_challenging_types_cf(cfAWS, 'AWS', result_path)
    plot_challenging_types_cf(cfAzure, 'Azure', result_path)

def IQA(AWSDir, AzureDir, resultDir):
    result_path = os.path.join(resultDir, 'IQA')
    if not os.path.exists(result_path + '/Data'): os.makedirs(result_path + '/Data')
    if not os.path.exists(result_path + '/Plots'): os.makedirs(result_path + '/Plots')

    cureor = CUREORrecognitionData(AWSDir, AzureDir, common=True, topN=5, IQA=True)
    IQA_vals = cureor.IQA_vals
    perf_vals = cureor.perf_vals

    scatter_plot_IQA(IQA_vals, perf_vals, result_path)

def acquisition_conditions(AWSDir, AzureDir, resultDir, common=False):
    result_path = os.path.join(resultDir, 'Acquisition_conditions')
    if not os.path.exists(result_path + '/Data'): os.makedirs(result_path + '/Data')
    if not os.path.exists(result_path + '/Plots'): os.makedirs(result_path + '/Plots')

    cureor = CUREORrecognitionData(AWSDir, AzureDir, common=common, topN=5)
    resultsAWS, resultsAzure = cureor.resultsAWS, cureor.resultsAzure

    cTypeIndex = cureor.cTypes.values()[1:9]
    postfix = 'selObj' if common == False else 'comObj'
    conditions = [[cureor.bgs, 0, 'bgs'],
                  [cureor.devs, 2, 'devs'],
                  [cureor.persps, 4, 'persps']]

    for c in conditions:
        index, idLoc, cStr = c

        try:
            dfAWS = pd.read_csv(os.path.join(result_path, 'Data', 'AWS_%s_%s.csv'%(cStr, postfix)), index_col=0)
            dfAzure = pd.read_csv(os.path.join(result_path, 'Data', 'Azure_%s_%s.csv'%(cStr, postfix)), index_col=0)

        except IOError:
            dfAWS = pd.DataFrame(index=[ind[3:] for ind in cTypeIndex], columns=index)
            dfAzure = pd.DataFrame(index=[ind[3:] for ind in cTypeIndex], columns=index)

            # AWS
            numObj = len(cureor.awsObj)
            numObjf = float(numObj)
            for i in cureor.cTypes.keys()[0:9]:
                ct = np.zeros(len(index)) # ex) aType==2: white, texture 1&2, 3d1&2
                if i not in [0,9]: # ignore original images
                    levels_tmp = cureor.levels[:-1] if i in [1,10] else cureor.levels
                    for lev in range(len(levels_tmp)):
                        for obj in resultsAWS[i][lev*numObj:lev*numObj + numObj]:
                            for img in obj.index:
                                idVal = int(img[idLoc]) - 1
                                ct[idVal] += 1
                    ind = i - 1 if i < 9 else i - 2 # color: ind = i - 1; grayscale: ind = i - 2
                    for rank in range(5): dfAWS.ix[ind, rank] = ct[rank]/numObjf/len(levels_tmp)/25*100

            ## Azure
            numObj = len(cureor.azureObj)
            numObjf = float(numObj)
            for i in cureor.cTypes.keys()[0:9]:
                ct = np.zeros(len(index)) # ex) aType==2: white, texture 1&2, 3d1&2
                if i not in [0,9]: # ignore original images
                    levels_tmp = cureor.levels[:-1] if i in [1,10] else cureor.levels
                    for lev in range(len(levels_tmp)):
                        for obj in resultsAzure[i][lev*numObj:lev*numObj + numObj]:
                            for img in obj.index:
                                idVal = int(img[idLoc]) - 1
                                ct[idVal] += 1
                    ind = i - 1 if i < 9 else i - 2 # color: ind = i - 1; grayscale: ind = i - 2
                    for rank in range(5): dfAzure.ix[ind, rank] = ct[rank]/numObjf/len(levels_tmp)/25*100

            dfAWS.to_csv(os.path.join(result_path, 'Data', 'AWS_%s_%s.csv'%(cStr,postfix)))
            dfAzure.to_csv(os.path.join(result_path, 'Data', 'Azure_%s_%s.csv'%(cStr,postfix)))

        plot_acquisition_conditions([dfAWS, dfAzure], ['AWS', 'Azure'], c, postfix, result_path, cureor.cTypes)

def CBIR_performance_estimation(AWSDir, AzureDir, resultDir):
    result_path = os.path.join(resultDir, 'CBIR')
    if not os.path.exists(result_path + '/Data'): os.makedirs(result_path + '/Data')
    if not os.path.exists(result_path + '/Plots'): os.makedirs(result_path + '/Plots')

    cureor = CUREORrecognitionData(AWSDir, AzureDir, common=True, topN=5, CBIR=True)

    corr_all = cureor.load_CBIR_perf_dist()
    for c, df in corr_all.iteritems():
        df.to_csv(os.path.join(result_path, 'Data', '%s.csv'%c))

def main():
    AWSDir, AzureDir, resultDir = 'AWS', 'Azure', 'Results'
    challenging_conditions(AWSDir, AzureDir, resultDir, common=True)
    challenging_conditions_cf(AWSDir, AzureDir, resultDir)
    IQA(AWSDir, AzureDir, resultDir)
    acquisition_conditions(AWSDir, AzureDir, resultDir, common=True)
    CBIR_performance_estimation(AWSDir, AzureDir, resultDir)

if __name__=="__main__":
    main()
