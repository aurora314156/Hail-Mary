import os, time, json
import xlwt
from xlwt import Workbook
from xlrd import open_workbook
from xlutils.copy import copy

class SaveLog():
    def __init__(self, dataType="", Process_dataset="", model="", Accuracy="", CostTime="", AccuracyList=""):
        self.dataType = dataType
        self.Process_dataset = Process_dataset
        self.model = model
        self.Accuracy = Accuracy
        self.CostTime = CostTime
        self.AccuracyList = AccuracyList

    def saveLogTxt(self):
        with open('log.txt', 'a') as log:
            log.write(self.dataType)
            log.write(self.Process_dataset)
            log.write(self.model)
            log.write(self.Accuracy)
            log.write(self.CostTime)

    def saveLogExcel(self):
        rb = open_workbook("experiment.xls")
        wb = copy(rb)

        sheet = wb.get_sheet(0)

        sheet.write(0, 0, 'model')
        sheet.write(0, 1, 'test')
        sheet.write(0, 2, 'dev')
        sheet.write(0, 3, 'train') 
        sheet.write(0, 4, '')
        sheet.write(0, 5, '')
        sheet.write(0, 6, 'model')
        sheet.write(0, 7, 'test')
        sheet.write(0, 8, 'dev')
        sheet.write(0, 9, 'train')
        for i in range(1,11):
            sheet.write(i, 0, str(i))

        ind_x, ind_y = 1, 1
        for a in self.AccuracyList:
            sheet.write(ind_y, ind_x, str(a))
            ind_x += 1
            if ind_x % 8 == 0:
                ind_y += 1
                ind_x = 1
                continue
            if ind_x % 4 == 0:
                ind_x +=1
                continue
            
        wb.save('experiment.xls')