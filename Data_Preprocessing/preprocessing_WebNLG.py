# -*- coding: utf-8 -*-
from src_preprocessing.completeMiner import *
from src_preprocessing.utils_webnlg_preprocess import *

def main():

    #1 merge files
    print("Merging files...")
    merge_of_merge()

    #2 create dict file (it takes a while beceause it queries DBpedia)
    print("Creating dict file...\n")
    create_file_format()

    #3 remove short test instances
    print("Removing short test instances...\n")
    remove_short_test()

    #4 Add the reification of the triples - this takes a while because 
    print("Adding reification...\n")
    reification()

    #5 Remove data points for which the reification didnt work
    print("Removing data points for which the reification didnt work...\n")
    remove_if_not_reification()

    #5 Format the file in final experimental format
    print("Formatting the file in final experimental format...\n")
    format()

if __name__ == "__main__":
    print("Starting preprocessing...")
    main()