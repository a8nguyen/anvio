#!/usr/bin/env python
# -*- coding: utf-8

# Copyright (C) 2014, A. Murat Eren
#
# This program is free software; you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the Free
# Software Foundation; either version 2 of the License, or (at your option)
# any later version.
#
# Please read the COPYING file.

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='BAM file profiler for entropy analysis')
    parser.add_argument('input_file', metavar = 'FILE PATH',
                        help = 'SAM file to analyze')
    parser.add_argument('--sorted', action = 'store_true', default = False,
                        help = 'Flag to define whether BAM file is already sorted')
    parser.add_argument('--indexed', action = 'store_true', default = False,
                        help = 'Flag to define whether BAM file is already indexed')
    parser.add_argument('--list-contigs', action = 'store_true', default = False,
                        help = 'Whend declared, lists contigs in the BAM file and\
                                exits without any further analysis.')
    parser.add_argument('--contigs', default = None,
                        help = 'It is possible to analyze only a group of contigs from\
                                a given BAM file. Contigs of interest can be specified\
                                using a comma separated list, or in a text file where\
                                each line contains a contig name.')

    args = parser.parse_args()
    
    if not os.path.exists(args.input_file):
        print 'No such file: "%s"' % args.input_file
        sys.exit()

    profiler = SamProfiler(args)
    profiler.generate_column_entropy_profile()
    profiler.generate_report()