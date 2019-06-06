import os, shutil


def eraseBertTmpFiles():
    shutil.rmtree("/project/Divh/tmp")
    os.mkdir("/project/Divh/tmp")
    allFiles = os.listdir(os.getcwd())
    currentPath = os.getcwd()
    for a in allFiles:
        filePath = os.path.join(currentPath, a)
        print(filePath)
        if a[:3] == "tmp":
            shutil.rmtree(filePath)
    