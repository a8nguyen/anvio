# -*- coding: utf-8
# pylint: disable=line-too-long
"""Simple KMers class to compute kmer frequecies

   This module should not be used for k > 4.
"""

import numpy as np
import itertools

from numba import jit, vectorize
from collections import Counter

import anvio

from anvio.constants import complements


__author__ = "Developers of anvi'o (see AUTHORS.txt)"
__copyright__ = "Copyleft 2015-2018, the Meren Lab (http://merenlab.org/)"
__credits__ = []
__license__ = "GPL 3.0"
__version__ = anvio.__version__
__maintainer__ = "A. Murat Eren"
__email__ = "a.murat.eren@gmail.com"
__status__ = "Development"


def rev_comp(seq):
    return seq.translate(complements)[::-1]


class KMers:
    def __init__(self, k=4, alphabet='ATCG', consider_rev_comps=True):
        self.kmers = {}
        self.alphabet = alphabet
        self.consider_rev_comps = consider_rev_comps
        self.k = k

        self.get_kmers()

    def get_kmers(self):
        k = self.k
        arg = [self.alphabet] * k
        kmers = set()

        for item in itertools.product(*arg):
            kmer = ''.join(item)
            if self.consider_rev_comps:
                if rev_comp(kmer) not in kmers:
                    kmers.add(kmer)
            else:
                kmers.add(kmer)

        self.kmers[k] = kmers


    def get_kmer_frequency(self, sequence, dist_metric_safe=False):
        """Get the kmer frequencies of a sequence

        Parameters
        ==========
        sequence : str OR numpy array (see as_ord)

        dist_metric_safe : bool, False
            If the kmer counts are all 0, make them all 1 so that distance metrics based on kmer
            counts do not blow up.
        """

        k = self.k
        sequence = sequence.upper()

        if len(sequence) < k:
            return None

        if k not in self.kmers:
            self.get_kmers(k)

        kmers = self.kmers[k]

        frequencies = Counter({})
        for i in range(0, len(sequence) - (k - 1)):
            kmer = sequence[i:i + k]

            if self.consider_rev_comps:
                if kmer in kmers:
                    frequencies[kmer] += 1
                else:
                    frequencies[rev_comp(kmer)] += 1
            else:
                frequencies[kmer] += 1

        if dist_metric_safe:
            # we don't want all kmer freq values to be zero. so the distance
            # metrics wouldn't go crazy. instead we fill it with 1. which
            # doesn't affect relative distances.
            if not len(frequencies):
                frequencies = dict(list(zip(kmers, [1] * len(kmers))))

        return frequencies


    def get_kmer_frequency2(self, sequence, dist_metric_safe=False):
        """Get the kmer frequencies of a sequence

        Parameters
        ==========
        sequence : str OR numpy array (see as_ord)

        dist_metric_safe : bool, False
            If the kmer counts are all 0, make them all 1 so that distance metrics based on kmer
            counts do not blow up.
        """

        k = self.k
        sequence = sequence.upper()

        # FIXME very incorrect, just for ease of development
        sequence = sequence.replace('N', '')

        if len(sequence) < k:
            return None

        if k not in self.kmers:
            self.get_kmers(k)

        kmers = self.kmers[k]

        frequencies = _get_kmer_frequency(np.frombuffer(sequence.encode('ascii'), np.uint8), k)
        frequencies = dict(list(zip(kmers, [1] * len(kmers))))

        if dist_metric_safe:
            pass

        return frequencies


@vectorize
def lookup(val):
    if val == 65:
        return 0
    elif val == 67:
        return 1
    elif val == 71:
        return 2
    elif val == 84:
        return 3
    else:
        return 4


@jit(nopython=True)
def _get_kmer_frequency(as_ord, k):
    as_index = lookup(as_ord)
    frequencies = np.zeros((4,4,4,4))

    for i in range(len(as_ord) - (k-1)):
        kmer = as_index[i: i+k]
        frequencies[kmer] += 1

    return frequencies
