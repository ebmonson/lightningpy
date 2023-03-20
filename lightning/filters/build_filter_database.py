#!/usr/bin/env python

'''
    build_filter_database.py

    Make a binary database containing filter names
    and where they can be found in the filter directory.
'''

import numpy as np
import json

def main():

    filter_arr = np.loadtxt('filter_yellowpages.txt', dtype='<U64')

    filter_dict = dict(filter_arr)

    print(filter_dict['SUBARU_V'])
    print(filter_dict['WFC3_F160W'])

    with open('filter_yellowpages.json', 'w') as file:
        json.dump(filter_dict, file)
    # end with

# end definition

if(__name__ == '__main__'):

    main()

# end if
