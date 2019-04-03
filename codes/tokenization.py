#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

class CWSTokenizer(object):
    def __init__(self,
                 vocab_file, 
                 max_seq_length=32,
                 segment_token='  ',
                 english_token='<ENG>',
                 number_token='<NUM>',
                 punctuation_token='<PUNC>',
                 bos_token='<BOS>',
                 eos_token='</s>',
                 delimiters='，。'):
        
        vocab = {}
        inv_vocab = {}
        with open(vocab_file, 'r') as fin:
            for index, word in enumerate(fin):
                word = word.strip()
                vocab[word] = index
                inv_vocab[index] = word
                
        self.vocab = vocab
        self.inv_vocab = inv_vocab
        self.max_seq_length = max_seq_length
        self.segment_token = segment_token
        self.english_token = english_token
        self.number_token = number_token
        self.punctuation_token = punctuation_token
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.delimiters = delimiters
        
    def word2id(self, word):
        if word not in self.vocab:
            return self.vocab['<UNK>']
        else:
            return self.vocab[word]
    
    def id2word(self, _id):
        return self.inv_vocab[_id]
    
    def sent_tokenize(self, sent):
        sent = sent.strip()
        untokenized_segments = [len(segment) for segment in sent.split(self.segment_token)]
        
        uchars = list(sent.replace(self.segment_token, ''))
        uchars = [(uchar, self.tokenize(uchar)) for uchar in uchars]
        
        outputs = [[]]
        segments = [[0]]
        curremt_seq_length = 0
        for uchar, token in uchars:
            if len(outputs[0]) == 0:
                outputs[-1].append((uchar, token))
                segments[-1][-1] += 1
                curremt_seq_length += 1
            elif outputs[-1][-1][1] == token and token in (self.english_token, self.punctuation_token, self.number_token):
                if token in (self.english_token, self.number_token):
                    outputs[-1][-1] = (outputs[-1][-1][0] + uchar, token)
                elif token == self.punctuation_token and outputs[-1][-1][0][-1:] == uchar:
                    outputs[-1][-1] = (outputs[-1][-1][0] + uchar, token)
                else:
                    outputs[-1][-1] = (outputs[-1][-1][0] + self.segment_token + uchar, token)
            elif curremt_seq_length == self.max_seq_length - 2:
                outputs[-1].append(('', self.eos_token))
                outputs.append([])
                segments.append([1])
                outputs[-1].append((uchar, token))
                curremt_seq_length = 1
            elif len(set(outputs[-1][-1][0]) & set(self.delimiters)) > 0:
                outputs[-1].append(('', self.eos_token))
                outputs.append([])
                segments.append([1])
                outputs[-1].append((uchar, token))
                curremt_seq_length = 1
            else:
                outputs[-1].append((uchar, token))
                segments[-1][-1] += 1
                curremt_seq_length += 1
                
            untokenized_segments[0] -= 1
            if untokenized_segments[0] == 0:
                del untokenized_segments[0]
                segments[-1].append(0)
        
        outputs[-1].append(('<\\n>', self.eos_token))
                
        uchars = [[self.bos_token] + [uchar for uchar, token in output] for output in outputs]
        tokens = [[self.bos_token] + [token for uchar, token in output] for output in outputs]
        
        for _segments in segments:
            while _segments and _segments[-1] == 0:
                del _segments[-1]
                
        for _uchars, _tokens, _segments in zip(uchars, tokens, segments):
            assert len(_uchars) == sum(_segments) + 2
            assert len(_uchars) == len(_tokens)
                
        return uchars, tokens, segments

    def restore(self, uchars, segments):
        sent = []
        start = 1
        for segment in segments:
            sent.append(''.join(uchars[start:start+segment]))
            start += segment
        sent.append(uchars[-1])
        sent = self.segment_token.join(sent)
        sent = sent.replace('<\\n>', '\n')
        return sent
    
    @classmethod
    def _is_chinese_char(cls, uchar):
        """Checks whether uchar is the codepoint of a CJK character."""
        # This defines a "chinese character" as anything in the CJK Unicode block:
        #   https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
        #
        # Note that the CJK Unicode block is NOT all Japanese and Korean characters,
        # despite its name. The modern Korean Hangul alphabet is a different block,
        # as is Japanese Hiragana and Katakana. Those alphabets are used to write
        # space-separated words, so they are not treated specially and handled
        # like the all of the other languages.
        cp = ord(uchar)
        
        if ((cp >= 0x4E00 and cp <= 0x9FFF) or  #
            (cp >= 0x3400 and cp <= 0x4DBF) or  #
            (cp >= 0x20000 and cp <= 0x2A6DF) or  #
            (cp >= 0x2A700 and cp <= 0x2B73F) or  #
            (cp >= 0x2B740 and cp <= 0x2B81F) or  #
            (cp >= 0x2B820 and cp <= 0x2CEAF) or
            (cp >= 0xF900 and cp <= 0xFAFF) or  #
            (cp >= 0x2F800 and cp <= 0x2FA1F) or
            (uchar in '○')):  #
            return True
        else:
            return False
    
    def tokenize(self, uchar):
        '''
        Tokenize uchar
        '''
        #Check whether uchar is a Chinese character
        
        cp = ord(uchar)

        #Check whether uchar is a number
        #０１２３４５６７８９
        #0123456789
        #ⅢⅣⅠⅡⅤ
        #％．＋＞∶‰+㈨℃.
        if ((0xff10 <= cp <= 0xff19) or
              (0x0030 <= cp <= 0x0039) or
              (0x2160 <= cp <= 0x2179) or 
              (uchar in '％．＋＞∶‰+㈨℃.')):
            return self.number_token

        #Check whether uchar is an English character
        #ａ-ｚ
        #Ａ-Ｚ
        #A-Z
        #a-z
        #alpha, beta, gamma, ...
        #＆
        elif ((0xff41 <= cp <= 0xff5a) or 
            (0xff21 <= cp <= 0xff3a) or 
            (0x0041 <= cp <= 0x005A) or 
            (0x0061 <= cp <= 0x007A) or 
            (0x3B1 <= cp <= 0x3D0) or
            (uchar == '＆')):
            return self.english_token
        
        elif self._is_chinese_char(uchar):
            return uchar
          
        else:
          #It is a punctuation
            return self.punctuation_token
