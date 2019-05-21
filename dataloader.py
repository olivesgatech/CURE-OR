import os, csv, json, sys
import pandas as pd
import numpy as np
from collections import OrderedDict

class CUREORrecognitionData:
    def __init__(self, AWSDir, AzureDir, resultDir='Results/', common=False, topN=5, cf=False, IQA=False):
        # AWSDir : directory of AWS recognition results
        # AzureDir : directory of Azure recognition results
        # Common: 10 common objects between AWS and Azure (default: False -> Top objects instead)

        self.AWSDir = AWSDir
        self.AzureDir = AzureDir
        self.resultDir = resultDir
        self.common = common
        self.topN = topN

        self.categories = OrderedDict([\
	                ('Toy', [20,21,37,38,39,40,41,42,62,63,76,78,83,84,85,86,87,88,89,90,91,92,93]),\
	                ('Personal Belongings', [9,11,12,13,24,25,53,64,69,99]),\
	                ('Office Supplies', [26,34,46,47,48,49,50,51,52,59,60,61,66,95]),\
	                ('Household', [1,2,6,7,15,16,17,22,27,28,30,33,43,45,55,58,65,67,70,71,72,73,74,75,82,97,100]),\
	                ('Sports/Entertainment', [5,8,10,14,18,19,68,79,80,81]),\
	                ('Health/Personal Care', [3,4,23,29,31,32,35,36,44,54,56,57,77,94,96,98])])

        self.cTypes = {	0: '01_no_challenge',
			            1: '02_resize',
			            2: '03_underexposure',
			            3: '04_overexposure',
			            4: '05_blur',
			            5: '06_contrast',
			            6: '07_dirtylens1',
			            7: '08_dirtylens2', 
			            8: '09_saltpepper',
			            9: '10_grayscale_no_challenge',
			            10: '11_grayscale_resize',
			            11: '12_grayscale_underexposure',
			            12: '13_grayscale_overexposure',
			            13: '14_grayscale_blur',
			            14: '15_grayscale_contrast',
			            15: '16_grayscale_dirtylens1',
			            16: '17_grayscale_dirtylens2',
			            17: '18_grayscale_saltpepper'}

        self.levels = ['Level_1','Level_2','Level_3','Level_4','Level_5']
        self.bgs = ['white','texture1','texture2','3d1','3d2']
        self.devs = ['iPhone','HTC','LG','Logitech','DSLR_JPG']
        self.persps = ['0 deg', '90 deg', '180 deg', '270 deg', 'Overhead']

        self.awsObj = [40,42,85,87,12,24,69,99,34,60,61,16,65,71,97,5,8,10,68,4,23,36,94] # 23 objects
        self.azureObj = [40,41,62,91,12,25,69,99,51,52,61,66,2,6,70,71,5,18,68,81,23,77,94]
        self.commonObj = list(set(self.awsObj) & set(self.azureObj)) # common objects between aws and azure
        if self.common:
            self.awsObj = self.commonObj
            self.azureObj = self.commonObj

        ## Import ground truth label lists: AWS, Azure separately
        with open(self.AWSDir + '/ground_truth.csv','r') as f:
            reader = csv.reader(f)
            self.gtAWS = [[item for item in row if item!= ''] for row in reader]

        with open(self.AzureDir + '/ground_truth.csv','r') as f:
            reader = csv.reader(f)
            self.gtAzure = [[item for item in row if item!= ''] for row in reader]

        if cf == True:
            with open(self.AWSDir + '/ground_truth.csv','r') as f:
                reader = csv.reader(f)
                catLabels = [[] for _ in range(len(self.categories))]
                for row in reader:
                    obj = int(row[0][0:3])
                    row = [x for x in row if x]
                    catInd = 0
                    for cat in self.categories:
                        if obj in self.categories[cat]: catLabels[catInd] += row[1:]
                        catInd += 1

                catLabels = [list(set(x)) for x in catLabels]
                catLabelsDictKeys = list(set([x for cat in catLabels for x in cat]))
                catLabelsDict = {key: [] for key in catLabelsDictKeys}
                for key in catLabelsDict.keys():
                    for cat in range(len(catLabels)):
                        if key in catLabels[cat]:
                            catLabelsDict[key].append(cat)
                self.catLabelsAWS = catLabels
                self.catLabelsDictAWS = catLabelsDict


            with open(self.AzureDir + '/ground_truth.csv','r') as f:
                reader = csv.reader(f)
                catLabels = [[] for _ in range(len(self.categories))]
                for row in reader:
                    obj = int(row[0][0:3])
                    row = [x for x in row if x]
                    catInd = 0
                    for cat in self.categories:
                        if obj in self.categories[cat]: catLabels[catInd] += row[1:]
                        catInd += 1

                catLabels = [list(set(x)) for x in catLabels]
                catLabelsDictKeys = list(set([x for cat in catLabels for x in cat]))
                catLabelsDict = {key: [] for key in catLabelsDictKeys}
                for key in catLabelsDict.keys():
                    for cat in range(len(catLabels)):
                        if key in catLabels[cat]:
                            catLabelsDict[key].append(cat)
                self.catLabelsAzure = catLabels
                self.catLabelsDictAzure = catLabelsDict

        ## Import the list of object names
        with open('cure_or_objects.txt') as file:
            self.cure_or_objects = file.readlines()

        self.resultsAWS, self.resultsAzure = self.load_recognition_results()
        if cf == True: self.cfAWS, self.cfAzure = self.load_confusion_matrix()
        if IQA == True: self.IQA_vals, self.perf_vals = self.load_IQA_results()

    def load_recognition_results(self):
        resultsAWS, resultsAzure = [], []
        minConf = 0

        awsObj, azureObj = self.awsObj, self.azureObj

		## AWS
        resultsAWS = []
        for i in self.cTypes.keys():
			tmpAWS = []
			###csv read: append df -> list of lists (challenges) of objects
			if i in [0,9]: # original images (color, grayscale)
				for obj in awsObj: 
					tmpAWS.append(pd.read_csv(os.path.join(self.AWSDir,
                                                           self.cTypes[i],
                                                           'By_object_conf_%d_N_%d_sameConf'%(minConf,self.topN),
                                                           self.cure_or_objects[obj - 1][:-2]+'.csv'),index_col=0))
			else: # challenges
				levels_tmp = self.levels[:-1] if i in [1, 10] else self.levels # resize: 4 levels only
				for lev in levels_tmp:
					for obj in awsObj:
						tmpAWS.append(pd.read_csv(os.path.join(self.AWSDir,
                                                               self.cTypes[i],
                                                               'By_object_conf_%d_N_%d_%s_sameConf'%(minConf,self.topN,lev),
                                                               self.cure_or_objects[obj - 1][:-2]+'.csv'),index_col=0))

			resultsAWS.append(tmpAWS)

		## Azure
        resultsAzure = []
        for i in self.cTypes.keys():
            tmpAzure = []
            ###csv read: append df -> list of lists (challenges) of objects
            if i in [0,9]: # original images (color, grayscale)
                for obj in azureObj: 
                    tmpAzure.append(pd.read_csv(os.path.join(self.AzureDir,
                                                             self.cTypes[i],
                                                             'By_object_conf_%d_N_%d_sameConf'%(minConf,self.topN),
                                                             self.cure_or_objects[obj - 1][:-2]+'.csv'),index_col=0))
            else: # challenges
                levels_tmp = self.levels[:-1] if i in [1, 10] else self.levels # resize: 4 levels only
                for lev in levels_tmp:
                    for obj in azureObj:
                        tmpAzure.append(pd.read_csv(os.path.join(self.AzureDir,
                                                                 self.cTypes[i],
                                                                 'By_object_conf_%d_N_%d_%s_sameConf'%(minConf,self.topN,lev),
                                                                 self.cure_or_objects[obj - 1][:-2]+'.csv'),index_col=0))

            resultsAzure.append(tmpAzure)

        return resultsAWS, resultsAzure


    def load_confusion_matrix(self):
        nCols = 7 # 'others' column

        outputLoc = os.path.join(self.resultDir, 'Challenging_conditions_cf', 'CSV')
        if not os.path.exists(outputLoc): os.makedirs(outputLoc)

        try:
            cfMatAWS, cfMatAzure = [], []
            for levCT in range(6):
                cfMatAWS.append(pd.read_csv(os.path.join(outputLoc, 'AWS_lev%d_top1_cf.csv'%levCT), index_col=0))
                cfMatAzure.append(pd.read_csv(os.path.join(outputLoc, 'Azure_lev%d_top1_cf.csv'%levCT), index_col=0))
            print('Loading existing confusion matrix data')

        except IOError:
            print('Generating data for confusion matrix')

	        # List of confusion matrices
            cfMatAWS = [pd.DataFrame(0, index=range(6), columns=range(nCols)) for _ in range(6)]
            cfMatAzure = [pd.DataFrame(0, index=range(6), columns=range(nCols)) for _ in range(6)]

            for i in self.cTypes.keys():
                # No challenge: color & grayscale -> one level only (level 0)
                if i in [0, 9]:
                    cfMatAWSPT = cfMatAWS[0]
                    cfMatAzurePT = cfMatAzure[0]
                    for bg in self.bgs:
                        for dev in self.devs:
                            # AWS
                            filename = os.path.join('AWS', self.cTypes[i], '_'.join([self.cTypes[i],bg,dev])+'.txt')
                            cfMatAWS[0] = self._confusion_matrix_each_AWS(filename, cfMatAWSPT)

                            # Azure
                            filename = os.path.join('Azure', self.cTypes[i], '_'.join([self.cTypes[i],bg,dev])+'.txt')
                            cfMatAzure[0] = self._confusion_matrix_each_Azure(filename, cfMatAzurePT)
                else:
                    levels_tmp = self.levels[:-1] if i in [1, 10] else self.levels
                    for lev in levels_tmp:
                        cfMatAWSPT = cfMatAWS[self.levels.index(lev)+1]
                        cfMatAzurePT = cfMatAzure[self.levels.index(lev)+1]
                        for bg in self.bgs:
                            for dev in self.devs:
                                # AWS
                                filename = os.path.join('AWS', self.cTypes[i], '_'.join([self.cTypes[i],lev,bg,dev])+'.txt')
                                cfMatAWS[self.levels.index(lev)+1] = self._confusion_matrix_each_AWS(filename, cfMatAWSPT)

                                # Azure
                                filename = os.path.join('Azure', self.cTypes[i], '_'.join([self.cTypes[i],lev,bg,dev])+'.txt')
                                cfMatAzure[self.levels.index(lev)+1] = self._confusion_matrix_each_Azure(filename, cfMatAzurePT)

            cfMatInd = self.categories.keys()
            cfMatInd.append('others')

            for app, cfMat in zip(['AWS', 'Azure'], [cfMatAWS, cfMatAzure]):
                levCT = 0
                for cf in cfMat: 
                    cf.columns = cfMatInd[:nCols]
                    cf = cf.rename(index={i: cfMatInd[i] for i in range(len(cf.index))})
                    cf.to_csv(os.path.join(outputLoc, '%s_lev%d_top1_cf.csv'%(app, levCT)))
                    levCT += 1

            cfMatAWS, cfMatAzure = [], []
            for levCT in range(6):
                cfMatAWS.append(pd.read_csv(os.path.join(outputLoc, 'AWS_lev%d_top1_cf.csv'%levCT), index_col=0))
                cfMatAzure.append(pd.read_csv(os.path.join(outputLoc, 'Azure_lev%d_top1_cf.csv'%levCT), index_col=0))

        return cfMatAWS, cfMatAzure

    def _confusion_matrix_each_AWS(self, filename, cfMatPT):

        with open(filename) as f:
            content = f.readlines()

        imgInd = np.array([l for l in range(len(content)) if content[l][0] != '{'])
        imgInd = np.append(imgInd,len(content)) # Ignore the last imgInd!

        categories = self.categories
        for k in range(len(imgInd)-1):
            obj = int(content[imgInd[k]][6:9])
            if obj in self.awsObj:

                # Determine the correct category of the object: inputCat (0 to 5)
                catInd = 0
                for cat in categories:
                    if obj in categories[cat]: 
                        inputCat = catInd
                        break
                    catInd += 1

                n = 0

                for j in range(imgInd[k]+1, imgInd[k+1]):
                    if n < 1:
                        line = json.loads(content[j])
                        maxConf = line['Confidence']
                        labels = [line['Name']]

                        # Check next line's confidence & collect if the same confidence value
                        if j < imgInd[k+1]:
                            for m in range(j, imgInd[k+1] - 1):
                                line = json.loads(content[m])
                                if line['Confidence'] >= maxConf: labels.append(line['Name'])
                                else: break

                        # Correct category
                        if bool(set(labels) & set(self.catLabelsAWS[inputCat])): 
                            
                            cfMatPT.ix[inputCat, inputCat] += 1

                        # Wrong category 
                        else: 
                            if bool(set(labels) & set(self.catLabelsDictAWS.keys())): # the label exists in dict
                                validLabels = list(set(labels) & set(self.catLabelsDictAWS.keys()))
                                for label in validLabels:
                                    score = 1.0/len(validLabels)/len(self.catLabelsDictAWS[label])
                                    for ind in self.catLabelsDictAWS[label]:
                                        cfMatPT.ix[inputCat, ind] += score
                            # 'Others' category
                            else:
                                cfMatPT.ix[inputCat, 6] += 1

                        n += 1
        return cfMatPT

    def _confusion_matrix_each_Azure(self, filename, cfMatPT):
        with open(filename) as f:
            content = f.readlines()

        categories = self.categories

        for k in range(len(content)):
            line = json.loads(content[k])
            obj = int(line['imgName'][6:9])

            if obj in self.azureObj:
                # Determine the correct category of the object: inputCat (0 to 5)
                catInd = 0
                for cat in categories:
                    if obj in categories[cat]: 
                        inputCat = catInd
                        break
                    catInd += 1

                n = 0

                for j in range(len(line['tags'])):
                    if n < 1:
                        tag = line['tags'][j]
                        maxConf = tag['confidence']
                        labels = [tag['name']]

                        if j < len(line['tags']):
                            for m in range(j + 1, len(line['tags'])):
                                tag = line['tags'][m]
                                if tag['confidence'] >= maxConf: labels.append(tag['name'])
                                else: break

                        # Correct category
                        if bool(set(labels) & set(self.catLabelsAzure[inputCat])): 
                            cfMatPT.ix[inputCat, inputCat] += 1

                        # Wrong category 
                        else: 
                            if bool(set(labels) & set(self.catLabelsDictAzure.keys())): # the label exists in dict
                                validLabels = list(set(labels) & set(self.catLabelsDictAzure.keys()))
                                for label in validLabels:
                                    score = 1.0/len(validLabels)/len(self.catLabelsDictAzure[label])
                                    for ind in self.catLabelsDictAzure[label]:
                                        cfMatPT.ix[inputCat, ind] += score
                            # 'Others' category
                            else:
                                cfMatPT.ix[inputCat, 6] += 1

                        n += 1

        return cfMatPT


    def load_IQA_results(self):
        self._prepare_IQA_results()
        return self._run_IQA_matlab_script()

    def _prepare_IQA_results(self):
        numObjAWS = len(self.awsObj)
        numObjAzure = len(self.azureObj)

        try:
            iqaAWS = pd.read_csv(os.path.join(self.resultDir, 'IQA', 'Data',
                                              'AWS_color_N_%d_obj_%d.csv'%(self.topN, numObjAWS)),
                                 index_col=0)
            iqaAzure = pd.read_csv(os.path.join(self.resultDir, 'IQA', 'Data',
                                                'Azure_color_N_%d_obj_%d.csv'%(self.topN, numObjAzure)),
                                   index_col=0)

        except IOError:
            columns = ['%d_%d%d%d'%(lev,i,j,k) for lev in range(6) for i in range(1,6) for j in range(1,6) for k in range(1,6)]
            iqaAWS = pd.DataFrame(0, index=self.cTypes.values()[1:9], columns=columns)
            iqaAzure = pd.DataFrame(0, index=self.cTypes.values()[1:9], columns=columns)

            for cType in self.cTypes.keys()[:9]:

                if cType == 0:
                    for obj in self.resultsAWS[cType]:
                        for img in obj.index:
                            bg, dev, persp = img[0], img[2], img[4]
                            iqaAWS.ix[:, '0_%s%s%s'%(bg,dev,persp)] += 1

                    for obj in self.resultsAzure[cType]:
                        for img in obj.index:
                            bg, dev, persp = img[0], img[2], img[4]
                            iqaAzure.ix[:, '0_%s%s%s'%(bg,dev,persp)] += 1
                else:
                    levels_tmp = self.levels[:-1] if i == 1 else self.levels
                    for lev in range(len(levels_tmp)):
                        for obj in self.resultsAWS[cType][lev*numObjAWS:lev*numObjAWS + numObjAWS]:
                            for img in obj.index:
                                bg,dev,persp = img[0], img[2], img[4]
                                iqaAWS.ix[cType - 1, '%d_%s%s%s'%(lev+1,bg,dev,persp)] += 1

                        for obj in self.resultsAzure[cType][lev*numObjAzure:lev*numObjAzure + numObjAzure]:
                            for img in obj.index:
                                bg,dev,persp = img[0], img[2], img[4]
                                iqaAzure.ix[cType - 1, '%d_%s%s%s'%(lev+1,bg,dev,persp)] += 1


            iqaAWS = iqaAWS / numObjAWS * 100
            iqaAzure = iqaAzure / numObjAzure * 100

            iqaAWS.to_csv(os.path.join(self.resultDir, 'IQA', 'Data', 'AWS_color_N_%d_obj_%d.csv'%(self.topN, numObjAWS)))
            iqaAzure.to_csv(os.path.join(self.resultDir, 'IQA', 'Data', 'Azure_color_N_%d_obj_%d.csv'%(self.topN, numObjAzure)))

        # return iqaAWS, iqaAzure
        return

    def _run_IQA_matlab_script(self):
        try:
            IQA_vals = pd.read_csv('Results/IQA/Data/IQA_concat_allLev.csv', header=None)
            perf_vals = pd.read_csv('Results/IQA/Data/Perf_concat_allLev.csv', header=None)
            return IQA_vals, perf_vals

        except IOError:
            print('No data found: Run IQA_loader.m file on Matlab or download IQA data!')
            sys.exit(1)
