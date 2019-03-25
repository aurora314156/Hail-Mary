import xlwt
from xlwt import Workbook
from xlrd import open_workbook
from xlutils.copy import copy

def saveLogExcel(AccuracyList):
    rb = open_workbook("experiment.xls")
    wb = copy(rb)

    sheet = wb.get_sheet(0)

    sheet.write(0, 0, 'model') 
    sheet.write(0, 1, 'test') 
    sheet.write(0, 2, 'train') 
    sheet.write(0, 3, 'dev') 
    sheet.write(0, 4, '')
    sheet.write(0, 5, 'test')
    sheet.write(0, 6, 'train')
    sheet.write(0, 7, 'dev')
    for i in range(1,11):
        sheet.write(i, 0, str(i))

    ind_x, ind_y = 1, 1
    for a in AccuracyList:
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

AccuracyList = [1,2,3,4,5,6,1,2,3,4,5,6]
saveLogExcel(AccuracyList)