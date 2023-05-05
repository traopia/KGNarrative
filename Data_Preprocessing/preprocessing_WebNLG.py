from src_preprocessing.completeMiner import *
from src_preprocessing.utils_webnlg_preprocess import *

def main():

    #1 merge files
    merge_of_merge()

    #2 create dict file (it takes a while beceause it queries DBpedia)
    create_file_format()

    #3 remove short test instances
    remove_short_test()

    #4 Add the reification of the triples - this takes a while because 
    reification()

    #5 Remove data points for which the reification didnt work
    remove_if_not_reification()

    #5 Format the file in final experimental format
    format()

if __name__ == "__main__":
    main()