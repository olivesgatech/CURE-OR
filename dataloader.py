import os, csv
import pandas as pd
from collections import OrderedDict

class CUREORrecognitionData:
    def __init__(self, AWSdir, Azuredir, common=False, topN=5):
        # AWSdir : directory of AWS recognition results
        # Azuredir : directory of Azure recognition results
        # Common: 10 common objects between AWS and Azure (default: False -> Top objects instead)

        self.AWSdir = AWSdir
        self.Azuredir = Azuredir
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
        self.devs = ['iPhone','HTC','LG','Logitech','Nikon']
        self.persps = ['0 deg', '90 deg', '180 deg', '270 deg', 'Overhead']

        self.awsObj = [40,42,85,87,12,24,69,99,34,60,61,16,65,71,97,5,8,10,68,4,23,36,94] # 23 objects
        self.azureObj = [40,41,62,91,12,25,69,99,51,52,61,66,2,6,70,71,5,18,68,81,23,77,94]
        self.commonObj = list(set(self.awsObj) & set(self.azureObj)) # common objects between aws and azure
        if self.common:
            self.awsObj = self.commonObj
            self.azureObj = self.commonObj

        ## Import ground truth label lists: AWS, Azure separately
        with open(self.AWSdir + '/ground_truth.csv','r') as f:
            reader = csv.reader(f)
            self.gtAWS = [[item for item in row if item!= ''] for row in reader]

        with open(self.Azuredir + '/ground_truth.csv','r') as f:
            reader = csv.reader(f)
            self.gtAzure = [[item for item in row if item!= ''] for row in reader]

        ## Import the list of object names
        with open('cure_or_objects.txt') as file:
            self.cure_or_objects = file.readlines()

        self.resultsAWS, self.resultsAzure = self.load_recognition_results()

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
					tmpAWS.append(pd.read_csv(os.path.join(self.AWSdir,
                                                           self.cTypes[i],
                                                           'By_object_conf_%d_N_%d_sameConf'%(minConf,self.topN),
                                                           self.cure_or_objects[obj - 1][:-2]+'.csv'),index_col=0))
			else: # challenges
				levels_tmp = self.levels[:-1] if i in [1, 10] else self.levels # resize: 4 levels only
				for lev in levels_tmp:
					for obj in awsObj:
						tmpAWS.append(pd.read_csv(os.path.join(self.AWSdir,
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
                    tmpAzure.append(pd.read_csv(os.path.join(self.Azuredir,
                                                             self.cTypes[i],
                                                             'By_object_conf_%d_N_%d_sameConf'%(minConf,self.topN),
                                                             self.cure_or_objects[obj - 1][:-2]+'.csv'),index_col=0))
            else: # challenges
                levels_tmp = self.levels[:-1] if i in [1, 10] else self.levels # resize: 4 levels only
                for lev in levels_tmp:
                    for obj in azureObj:
                        tmpAzure.append(pd.read_csv(os.path.join(self.Azuredir,
                                                                 self.cTypes[i],
                                                                 'By_object_conf_%d_N_%d_%s_sameConf'%(minConf,self.topN,lev),
                                                                 self.cure_or_objects[obj - 1][:-2]+'.csv'),index_col=0))

            resultsAzure.append(tmpAzure)

        return resultsAWS, resultsAzure

