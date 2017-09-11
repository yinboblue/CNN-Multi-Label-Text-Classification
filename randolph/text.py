# -*- coding: utf-8 -*-
# Randolph's Package: Texts
#
# Copyright (C) 2015-2017 Randolph
# Author: Randolph <chinawolfman@hotmail.com>

"""
This module brings together some functionality for
text analysis, and provides simple, interactive interfaces.
Functionality includes: concordancing, collocation discovery,
regular expression search over tokenized strings, and
distributional similarity.
"""

import re
import sys
import os
import os.path
import linecache
import chardet
import codecs

from math import log
from collections import defaultdict, Counter
from operator import itemgetter
from functools import reduce
from itertools import islice
from six import text_type
from randolph.probability import FreqDist


class TokenSearcher(object):
    """
    A class that makes it easier to use regular expressions to search
    over tokenized strings.  The tokenized string is converted to a
    string where tokens are marked with angle brackets -- e.g.,
    ``'<the><window><is><still><open>'``.  The regular expression
    passed to the ``findall()`` method is modified to treat angle
    brackets as non-capturing parentheses, in addition to matching the
    token boundaries; and to have ``'.'`` not match the angle brackets.
    """

    def __init__(self, tokens):
        self._raw = ''.join('<' + w + '>' for w in tokens)

    def findall(self, regexp):
        """
        Find instances of the regular expression in the text.
        The text is a list of tokens, and a regexp pattern to match
        a single token must be surrounded by angle brackets.  E.g.

        >>> from randolph.text import Text, TokenSearcher
        >>> text = Text('test.txt')
        >>> TokenSearcher(text.tokens()).findall("<the><.*>")
        >>> TokenSearcher(text.tokens()).findall("<a>(<.*>)<man>")
        monied; nervous; dangerous; white; white; white; pious; queer; good;
        >>> TokenSearcher(text.tokens()).findall("<th.*>{3,}")
        thread through those; the thought that; that the thing; the thing
        that; that that thing; through these than through; them that the;
        through the thick; them that they; thought that the

        :param regexp: A regular expression
        :type regexp: str
        """
        # preprocess the regular expression
        regexp = re.sub(r'\s', '', regexp)
        regexp = re.sub(r'<', '(?:<(?:', regexp)
        regexp = re.sub(r'>', ')>)', regexp)
        regexp = re.sub(r'(?<!\\)\.', '[^>]', regexp)

        # perform the search
        hits = re.findall(regexp, self._raw)

        # Sanity check
        for h in hits:
            if not h.startswith('<') and h.endswith('>'):
                raise ValueError('Bad regexp for TokenSearcher.findall')

        # postprocess the output
        hits = [h[1:-1].split('><') for h in hits]
        return hits


class Text(object):
    """
    This module brings together a variety of functions for
    text analysis, and provides simple, interactive interfaces.
    """

    def __init__(self, filename):
        self.filename = filename

    @property
    def cur_path(self):
        """Returns the path of the file."""
        return os.path.abspath(self.filename)

    @property
    def line_num(self):
        """Returns the line number of the file."""
        count = 1
        with open(self.filename, 'rb') as fin:
            while True:
                buffer = fin.read(8192 * 1024)
                if not buffer:
                    break
                count += buffer.decode('utf-8', 'ignore').count('\n')
        return count

    def tokens(self, keep_punctuation=None):
        """Return the word tokens of the file content."""
        if keep_punctuation is None:
            regex = '\W+'
        else:
            regex = '(\W+)'
        with open(self.filename, 'r') as fin:
            tokens = []
            for eachline in fin:
                line = re.split(regex, eachline)
                for item in line:
                    if item and item != ' ' and item != '.\n':
                        tokens.append(item)
        return tokens

    def sentence_tokens(self, keep_punctuation=None):
        """Return the sentence tokens of the file content."""
        if keep_punctuation is None:
            regex = '\W+'
        else:
            regex = '(\W+)'
        with open(self.filename, 'r') as fin:
            sentence_tokens = []
            for eachline in fin:
                line = []
                eachline = re.split(regex, eachline)
                for item in eachline:
                    if item and item != ' ' and item != '.\n':
                        line.append(item)
                sentence_tokens.append(line)
        return sentence_tokens

    def count(self, word, case_sensitive=True):
        """Count the number of times this word appears in the text."""
        count = 0
        for item in self.tokens():
            if not case_sensitive:
                item = item.lower()
            if item == word:
                count += 1
        return count

    def index(self, word, case_sensitive=True):
        """Find the index of the first occurrence of the word in the text."""
        for index, item in enumerate(self.tokens()):
            if not case_sensitive:
                item = item.lower()
            if item == word:
                return index

    def get_after_lines_content(self, n):
        """Get the content after the N line of the file."""
        string = linecache.getlines(self.filename, n)
        return string

    def get_line_content(self, n):
        """Get the N line content of the file."""
        string = linecache.getline(self.filename, n)
        return string

    # 根据某列(index)内容排序, index 表示需要根据哪一列内容进行排序
    def sort(self, index):
        """
        Sort the file content according to the 'index' row to a new file.
        :param index: The row of the file content need to sort
        """
        sorted_lines = sorted(open(self.filename, 'r'), key=lambda x: float(x.strip().split('\t')[index]), reverse=True)
        open(self.filename, 'w').write(''.join(sorted_lines))

    def detect_file_encoding_format(self):
        """Detect the encoding of the given file."""
        with open(self.filename, 'rb') as f:
            data = f.read()
        source_encoding = chardet.detect(data)
        print(source_encoding)
        return source_encoding

    def convert_file_to_utf8(self):
        """Convert the encoding of the file to 'utf-8'(does not backup the origin file)."""
        with open(self.filename, "rb") as f:
            data = f.read()
        source_encoding = chardet.detect(data)['encoding']
        if source_encoding is None:
            print("??", self.filename)
            return
        print("  ", source_encoding, self.filename)
        if source_encoding != 'utf-8' and source_encoding != 'UTF-8-SIG':
            content = data.decode(source_encoding, 'ignore')  # .encode(source_encoding)
            codecs.open(self.filename, 'w', encoding='utf-8').write(content)

    def concordance(self, word, width=79, lines=25, case_sensitive=True):
        """
        Print a concordance for ``word`` with the specified context window.
        Word matching is case-sensitive default.
        """
        half_width = (width - len(word) - 2) // 2
        context = width // 4  # approx number of words of context

        tokens = self.tokens(keep_punctuation=True)
        offsets = []

        for index, item in enumerate(tokens):
            if not case_sensitive:
                item = item.lower()
            if item == word:
                offsets.append(index)

        if offsets:
            lines = min(lines, len(offsets))
            print("Displaying %s of %s matches:" % (lines, len(offsets)))
            for i in offsets:
                if lines <= 0:
                    break
                left = (' ' * half_width +
                        ' '.join(tokens[i - context:i]))
                right = ' '.join(tokens[i + 1:i + context])
                left = left[-half_width:]
                right = right[:half_width]
                print(left, tokens[i], right)
                lines -= 1
        else:
            print("No matches")

    def collocations(self, num=20, window_size=2):
        """
        Print collocations derived from the text, ignoring stopwords.
        :param num: The maximum number of collocations to print.
        :type num: int
        :param window_size: The number of tokens spanned by a collocation (default=2)
        :type window_size: int
        """
        if not ('_collocations' in self.__dict__ and self._num == num and self._window_size == window_size):
            self._num = num
            self._window_size = window_size

            #print("Building collocations list")
            ignored_words = stopwords.words('english')
            finder = BigramCollocationFinder.from_words(self.tokens, window_size)
            finder.apply_freq_filter(2)
            finder.apply_word_filter(lambda w: len(w) < 3 or w.lower() in ignored_words)
            bigram_measures = BigramAssocMeasures()
            self._collocations = finder.nbest(bigram_measures.likelihood_ratio, num)
        colloc_strings = [w1+' '+w2 for w1, w2 in self._collocations]
        print(tokenwrap(colloc_strings, separator="; "))

    def findall(self, regexp):
        """
        Find instances of the regular expression in the text.
        The text is a list of tokens, and a regexp pattern to match
        a single token must be surrounded by angle brackets.
        """
        return TokenSearcher(self.tokens()).findall(regexp)

    def freq_dist(self):
        freq_dict = FreqDist([word.lower() for word in self.tokens()])
        return freq_dict
