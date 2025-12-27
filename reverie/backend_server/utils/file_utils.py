"""
File: file_utils.py
Description: CSV utility functions for reading maze data files.
"""

import csv


def read_file_to_list(curr_file, header=False, strip_trail=True):
    """
    Reads in a csv file to a list of list. If header is True, it returns a
    tuple with (header row, all rows)
    ARGS:
      curr_file: path to the current csv file.
    RETURNS:
      List of list where the component lists are the rows of the file.
    """
    analysis_list = []
    with open(curr_file) as f_analysis_file:
        data_reader = csv.reader(f_analysis_file, delimiter=",")
        for row in data_reader:
            if strip_trail:
                row = [i.strip() for i in row]
            analysis_list.append(row)

    if header:
        return analysis_list[0], analysis_list[1:]
    return analysis_list
