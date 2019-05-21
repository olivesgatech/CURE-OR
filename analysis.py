import pandas as pd
import os

from dataloader import CUREORrecognitionData
from utils import *

def challenging_conditions(AWSDir, AzureDir, resultDir, common=False, topN=5):
    cureor = CUREORrecognitionData(AWSDir, AzureDir, common, topN)
    result_path = os.path.join(resultDir, 'Challenging_conditions')
    if not os.path.exists(result_path + '/CSV'): os.makedirs(result_path + '/CSV')
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
            dfNumGTcolor = pd.read_csv(os.path.join(result_path, 'CSV', color_file+'.csv'), index_col=0)
            dfNumGTgray = pd.read_csv(os.path.join(result_path, 'CSV', gray_file+'.csv'), index_col=0)
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


            dfNumGTcolor.to_csv(os.path.join(result_path, 'CSV', '%s_%s_color_N_%d.csv'%(app,objs,topN)))
            dfNumGTgray.to_csv(os.path.join(result_path, 'CSV', '%s_%s_gray_N_%d.csv'%(app,objs,topN)))

            plot_challenging_types([dfNumGTcolor, dfNumGTgray],
                                   [color_file, gray_file],
                                   result_path)

    return

def challenging_conditions_cf(AWSDir, AzureDir, resultDir, common=False, topN=5):
    cureor = CUREORrecognitionData(AWSDir, AzureDir, common, topN)
    result_path = os.path.join(resultDir, 'Challenging_conditions_cf')
    if not os.path.exists(result_path + '/CSV'): os.makedirs(result_path + '/CSV')
    if not os.path.exists(result_path + '/Plots'): os.makedirs(result_path + '/Plots')

    cfAWS, cfAzure = cureor.cfAWS, cureor.cfAzure

    plot_challenging_types_cf(cfAWS, 'AWS', result_path)
    plot_challenging_types_cf(cfAzure, 'Azure', result_path)

# def IQA(IQADir):


def main():
    # challenging_conditions('AWS', 'Azure', 'Results', common=True)
    challenging_conditions_cf('AWS', 'Azure', 'Results')
    # IQA('IQA')
if __name__=="__main__":
    main()
