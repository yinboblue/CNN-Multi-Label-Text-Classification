# -*- coding: utf-8 -*-
# Randolph's Package: Files
#
# Copyright (C) 2015-2017 Randolph
# Author: Randolph <chinawolfman@hotmail.com>

"""This module brings useful file function."""

import re
import sys
import os
import os.path
import pip

from subprocess import call


def upgrade_package():
    """Upgrade all installed python3 packages."""
    for dist in pip.get_installed_distributions():
        call("pip3 install --upgrade " + dist.project_name, shell=True)


def create_list(num, prefix='', postfix='', filetype='.txt'):
    """
    Create the file list.
    :param num: The number of the file
    :param prefix: The prefix of the file
    :param postfix: The postfix of the file
    :param filetype: The file type of the file
    """
    output_file_list = []
    for i in range(num):
        output_file_list.append(prefix + str(i + 1) + postfix + filetype)
    return output_file_list


def list_cur_all_file():
    """Return a list containing the names of the files in the directory(including the hidden files)."""
    file_list = [filename for filename in os.listdir(os.getcwd())]
    print(file_list)
    return file_list


def listdir_nohidden():
    """Return a list containing the names of the files in the directory."""
    file_list = [filename for filename in os.listdir(os.getcwd()) if not filename.startswith('.')]
    print(file_list)
    return file_list


def extract(input_file, output_file):
    """Extract the first column content of the file to the new file."""
    lines = [eachline.strip().split('\t')[0] for eachline in open(input_file, 'r')]
    open(output_file, 'w').write(''.join((item + '\n') for item in lines))


def judge(input_file1, input_file2):
    """To determine whether the first column content of the two files exist duplicate information."""
    lines_1 = [eachline.strip().split('\t')[0] for eachline in open(input_file1, 'r')]
    lines_2 = [eachline.strip().split('\t')[0] for eachline in open(input_file2, 'r')]
    count = 0
    for item in lines_1:
        if item in lines_2:
            count += 1
        else:
            print(item)
    if count > 0:
        print('Total same info number: %d' % count)
    else:
        print('Exactly the same content.')


def copy_files(source_dir, target_dir):
    """Copy all files of the source path to the target path."""
    for file in os.listdir(source_dir):
        source_file = os.path.join(source_dir, file)
        target_file = os.path.join(target_dir, file)
        if os.path.isfile(source_file):
            if not os.path.exists(target_dir):
                os.makedirs(target_dir)
            if not os.path.exists(target_file) or (os.path.exists(target_file) and (
                        os.path.getsize(target_file) != os.path.getsize(source_file))):
                open(target_file, "wb").write(open(source_file, "rb").read())
            if os.path.isdir(source_file):
                copy_files(source_file, target_file)


def remove_file_in_first_dir(target_dir):
    """Delete all files of the target path."""
    for file in os.listdir(target_dir):
        target_file = os.path.join(target_dir, file)
        if os.path.isfile(target_file):
            os.remove(target_file)