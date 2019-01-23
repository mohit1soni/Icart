import csv
import os
from PIL import Image

def readTrafficSigns(rootpath):
    csv.register_dialect('md',delimiter=',',quoting=csv.QUOTE_NONE,skipinitialspace=True)
    files=os.listdir(rootpath)
    for file1 in files:
        print(file1)
        image = Image.open(rootpath+'/'+file1)
        image.save(rootpath+'/'+file1.replace("Test",""))

def delete_multiple(rootpath):
    files=os.listdir(rootpath)
    for file1 in files:
        if file1.endswith(".ppm"):
            os.remove(rootpath+'/'+file1)

def modify_csv(path):
    gtfile=open(path+'/'+"test_labels.csv")
    gtreader=csv.reader(gtfile)
    next(gtreader)
    with open(path+'/'+"new_test_labels.csv",'a',newline='') as f:
        writer=csv.writer(f,dialect='excel')
        for row in gtreader:
            row[0]=row[0].replace(".ppm",".jpg")
            new_row=[row[0],row[1],row[2],row[3],row[4],row[5],row[6],row[7]]
            writer.writerow(new_row)
    gtfile.close()

def main():
    path_image="../../workspace/training_demo/images/Test"
    #path_csv="../../workspace/training_demo/annotations"
    #modify_csv(path_csv)
    readTrafficSigns(path_image)
    # delete_multiple(path_image)

if __name__ == "__main__":
    main()