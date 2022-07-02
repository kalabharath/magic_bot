import sys, os, glob
import random


def select_random_file(path, total_count):
    files = glob.glob(path)
    random.shuffle(files)
    count = 0
    for f in files:
        if random.randint(0, 1):
            if random.randint(0, 1):
                mv_file = os.system("mv "+f+" ./")
                count += 1
                if count == total_count:
                    break

    return True

# main
if __name__ == '__main__':
    total_number_of_files = int (sys.argv[1])
    select_random_file('./data/*radar*.png', total_number_of_files)