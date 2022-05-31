import os

Path = r'C:\Users\PeterChuang\Desktop\yolov5\Mypaper\detect_data_separate\valid\old_labels'
save_path = r'C:\Users\PeterChuang\Desktop\yolov5\Mypaper\detect_data_separate\valid\labels'
allFileList = os.listdir(Path)
MAX = 0
for file in allFileList:
    path = os.path.join(Path,file)
    f = open(path, 'r')
    f = f.readlines()
    f = f[0]
    f = f.split(' ')
    f1 = float(f[0])
    f2 = float(f[1])
    txt = str(f1/180) + ' ' + str(f2/4000)
    path = os.path.join(save_path,file)
    f = open(path, 'w')
    f.write(txt)
    f.close()
