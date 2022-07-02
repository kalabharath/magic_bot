import sys, os, glob
import datetime

def get_all_files(path):
    return glob.glob(path)


def extract_n_plus_3_or_4(file_name):
    date = file_name.split("_")[2]
    tdate = date.strip('.png')
    print (tdate)
    file_prefix = file_name.split("_")[0]+'_'+file_name.split("_")[1]+'_'
    for i in range(3, 6):
        next_date = datetime.datetime.strptime(tdate, '%d%b%y') + datetime.timedelta(days=i)
        next_date = next_date.strftime('%d%b%y')
        next_file_name = file_prefix+next_date+'.png'
        # print(next_file_name)
        if os.path.isfile(next_file_name):
            print(next_file_name)
            os.system("mv "+next_file_name+" ./")
            return True

if __name__ == '__main__':
    files = get_all_files('./data/*.png')
    for file in files:
        extract_n_plus_3_or_4(file)

