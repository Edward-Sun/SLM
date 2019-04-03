#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random
import logging

import torch
from torch.utils import data

class InputDataset(data.Dataset):
    def __init__(self, 
                 input_files, 
                 tokenizer, 
                 is_training=False, 
                 batch_token_size=None, 
                 num_buckets=8):
        
        sent_uchars = []
        sent_tokens = []
        sent_segments = []
        
        line_count = 0
        for file in input_files:
            with open(file, 'r') as fin:
                for line in fin:
                    line_count += 1
                    uchars, tokens, segments = tokenizer.sent_tokenize(line)
                    sent_uchars.extend(uchars)
                    sent_tokens.extend(tokens)
                    sent_segments.extend(segments)
        
        for _uchars, _tokens, _segments in zip(sent_uchars, sent_tokens, sent_segments):
            assert len(_uchars) == sum(_segments) + 2
            assert len(_uchars) == len(_tokens)
        
        logging.info('#line: %d' % line_count)
        logging.info('#sentence: %d' % len(sent_tokens))
        logging.info('#token: %d' % sum(len(tokens) for tokens in sent_tokens))
        
        for c, _ in list(enumerate(zip(sent_uchars, sent_tokens, sent_segments)))[:10]:
            uchars, tokens, segments = _
            logging.info('##########Example %d##########' % c)
            logging.info('Characters: %s' % ' '.join(uchars))
            logging.info('Tokens: %s' % ' '.join(tokens))
            logging.info('Segments: %s' % ' '.join([str(segment) for segment in segments]))

        sent_ids = [[tokenizer.word2id(token) for token in sent] for sent in sent_tokens]
        
        data = list(zip(sent_uchars, sent_ids, sent_segments))
        
        self.is_training = is_training
        
        if is_training:
            data.sort(key=lambda x: len(x[1])) #ascending
            buckets = []
            bucket_size = len(data) // num_buckets + 1
            for i in range(0, len(data), bucket_size):
                buckets.append(tuple(data[i:i+bucket_size]))
            
            self.bucket_batch_size = tuple([max(1, batch_token_size // len(bucket[-1][1])) for bucket in buckets])
            self.buckets = tuple(buckets)
            self.num_buckets = num_buckets
            
            logging.info('Bucket batch sizes: %s' % ','.join([str(_) for _ in self.bucket_batch_size]))
        
        self.data = tuple(data)
            
    def __len__(self):
        if self.is_training:
            return 10000000000
        else:
            return len(self.data)

    def __getitem__(self, index):
        if self.is_training:
            random_bucket, bucket_batch_size = random.choice(list(zip(list(self.buckets),list(self.bucket_batch_size))))
            
            random_bucket = list(random_bucket)
                
            random.shuffle(random_bucket)
            
            ret = random_bucket[:bucket_batch_size]
            
            ids = [torch.LongTensor(ids) for uchars, ids, segments in ret]
            uchars = [uchars for uchars, ids, segments in ret]
            segments = [segments for uchars, ids, segments in ret]
            
            texts, lengths, collated_uchars, collated_segments, _ = self.padding_collate(list(zip(ids, uchars, segments)))
            
            return texts, lengths, collated_uchars, collated_segments
        
        else:
            uchars, ids, segments = self.data[index]
            ids = torch.LongTensor(ids)
            return ids, uchars, segments

    @staticmethod
    def padding_collate(batch):
        if len(batch) == 1:
            texts = batch[0][0]
            lengths = [text.size(0)]
            texts = texts.unsqueeze(0)
            restore_orders = [0]
            
        elif len(batch) > 1:
            for i in range(len(batch)):
                batch[i] = (batch[i][0], batch[i][1], batch[i][2], i)
                
            texts, lengths, collated_uchars, collated_segments, orders = zip(
                *[(instance[0], instance[0].size(0), instance[1], instance[2], instance[3]) 
                  for instance in sorted(batch, key=lambda x: x[0].size(0), reverse=True)])
            max_len = texts[0].size(0)
            texts = [torch.cat([s, torch.zeros(size=[max_len - s.size(0)], dtype=texts[0].dtype)], 0) 
                     if s.size(0) != max_len else s for s in texts]
            texts = torch.stack(texts, 0)
            
            restore_orders = []
            for i in range(len(batch)):
                restore_orders.append(orders.index(i))
                
        return texts, lengths, collated_uchars, collated_segments, restore_orders
    
    @staticmethod
    def single_collate(batch):
        return batch[0]
    
class OneShotIterator(object):
    def __init__(self, dataloader):
        self.iterator = self.one_shot_iterator(dataloader)
        self.step = 0
        
    def __next__(self):
        self.step += 1
        return next(self.iterator)
    
    @staticmethod
    def one_shot_iterator(dataloader):
        '''
        Transform a PyTorch Dataloader into python iterator
        '''
        while True:
            for data in dataloader:
                yield data