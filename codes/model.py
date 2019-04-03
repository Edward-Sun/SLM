#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SLM Model
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six
import copy
import json

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class SLMConfig(object):
    """Configuration for `SegmentalLM`."""

    def __init__(self,
                 vocab_size,
                 embedding_size=256,
                 hidden_size=256,
                 max_segment_length=4,
                 encoder_layer_number=1,
                 decoder_layer_number=1,
                 encoder_input_dropout_rate=0.0,
                 decoder_input_dropout_rate=0.0,
                 encoder_dropout_rate=0.0,
                 decoder_dropout_rate=0.0,
                 punc_id=2,
                 num_id=3,
                 eos_id=5,
                 eng_id=7):
        """
        Constructs SLMConfig.
        """
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.max_segment_length = max_segment_length
        self.encoder_layer_number = encoder_layer_number
        self.decoder_layer_number = decoder_layer_number
        self.encoder_input_dropout_rate = encoder_input_dropout_rate
        self.decoder_input_dropout_rate = decoder_input_dropout_rate
        self.encoder_dropout_rate = encoder_dropout_rate
        self.decoder_dropout_rate = decoder_dropout_rate
        self.eos_id = eos_id
        self.punc_id = punc_id
        self.eng_id = eng_id
        self.num_id = num_id

    @classmethod
    def from_dict(cls, json_object):
        """Constructs a `SLMConfig` from a Python dictionary of parameters."""
        config = SLMConfig(vocab_size=None)
        for (key, value) in six.iteritems(json_object):
            config.__dict__[key] = value
        return config

    @classmethod
    def from_json_file(cls, json_file):
        """Constructs a `SLMConfig` from a json file of parameters."""
        with open(json_file, "r") as reader:
            text = reader.read()
        return cls.from_dict(json.loads(text))

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)

    def to_json_file(self, json_file):
        """Serializes this instance to a JSON file."""
        with open(json_file, "w") as writer:
            writer.write(self.to_json_string())

class SegmentalLM(nn.Module):
    """
    SLM (Segmental Language Model)
    """
    
    def __init__(self,
                 config,
                 init_embedding = None):
        """
        Constructor for BertModel.
        """
        
        super(SegmentalLM, self).__init__()
        
        config = copy.deepcopy(config)

        self.config = config
        
        if init_embedding is not None:
            assert np.shape(init_embedding)[0] == config.vocab_size
            assert np.shape(init_embedding)[1] == config.embedding_size
            shard_embedding = nn.Parameter(torch.from_numpy(init_embedding).float())
        else:
            shard_embedding = torch.zeros(config.vocab_size, config.embedding_size)
            nn.init.uniform_(shard_embedding, a=-1.0, b=1.0)
            
        self.embedding = nn.Embedding.from_pretrained(shard_embedding)
        
        self.embedding2vocab = nn.Linear(config.embedding_size, config.vocab_size)
        #Weight Tying
        self.embedding2vocab.weight = self.embedding.weight
        
        self.context_encoder = ContextEncoder(
            input_size=config.embedding_size,
            hidden_size=config.hidden_size,
            layer_number=config.encoder_layer_number,
            dropout_rate=config.encoder_dropout_rate
        )
        
        self.segment_decoder = SegmentDecoder(
            hidden_size=config.hidden_size, 
            output_size=config.embedding_size,
            layer_number=config.decoder_layer_number,
            dropout_rate=config.decoder_dropout_rate
        )
        
        self.decoder_h_transformation = nn.Linear(
            config.hidden_size, 
            config.decoder_layer_number * config.hidden_size
        )
        
        self.encoder_input_dropout = nn.Dropout(p=config.encoder_input_dropout_rate)
        self.decoder_input_dropout = nn.Dropout(p=config.decoder_input_dropout_rate)
        
        self.start_of_segment = nn.Linear(
            config.hidden_size, 
            config.hidden_size
        )
        
    def forward(self, x, lengths, segments=None, mode='unsupervised'):
        if mode == 'supervised' and segments is None:
            raise ValueError('Supervised mode needs segmented text.')
        
        #input format: (seq_len, batch_size) 
        x = x.transpose(0, 1).contiguous()
        
        #transformed format: (seq_len, batch_size) 
        max_length = x.size(0)
        batch_size = x.size(1)
        
        loginf=1000000.0
        
        schedule = []
        
        max_length = max(lengths)
        
        #<BOS> is at the begin of sentence,
        #<PUNC> is at the end of sentece
        #sentence_length = true_length + 2 
        for j_start in range(1, max_length - 1):
            j_len = min(self.config.max_segment_length, (max_length-1) - j_start)
            j_end = j_start + j_len
            schedule.append((j_start, j_len, j_end))        
        
        inputs = self.embedding(x)
        if self.config.embedding_size != self.config.hidden_size:
            inputs = self.embedding2hidden(inputs)
        
        
        is_single = torch.eq(x, self.config.punc_id).type_as(inputs) * -loginf
        is_single = is_single + torch.eq(x, self.config.eng_id).type_as(inputs) * -loginf
        is_single = is_single + torch.eq(x, self.config.num_id).type_as(inputs) * -loginf
        
        
        if mode == 'supervised':
          is_single = torch.zeros_like(is_single)

        neg_inf_vector = torch.full_like(inputs[0,:,0], -loginf)

        #log_probability
        logpy = [[ neg_inf_vector
                  for i in range(self.config.max_segment_length)] 
                 for j in range(max_length - 1)]
        
        logpy[0][0] = torch.zeros_like(neg_inf_vector)
        
        #Context Encoder
        inputs = self.encoder_input_dropout(inputs)
        encoder_init_states = self.context_encoder.get_init_states(batch_size)
        encoder_output = self.context_encoder(inputs, lengths, encoder_init_states)
        
        #Hack, make context encoder and segment decoder have different learning rate
        encoder_output = encoder_output * 0.5

        for j_start, j_len, j_end in schedule:
            decoder_init = encoder_output[j_start-1, :, :]
            
            start_symbol = self.start_of_segment(decoder_init).unsqueeze(0)

            decoder_h_init = torch.tanh(self.decoder_h_transformation(decoder_init))
            decoder_h_init = decoder_h_init.view(batch_size, self.config.decoder_layer_number, -1)
            decoder_h_init = torch.transpose(decoder_h_init, 0, 1).contiguous()
            
            decoder_c_init = torch.zeros_like(decoder_h_init)
            decoder_init_states = (decoder_h_init, decoder_c_init)
            
            decoder_input = self.decoder_input_dropout(inputs[j_start:j_end, :, :])
            decoder_input = torch.cat([start_symbol, decoder_input], dim=0).contiguous()
            
            #Segment Decoder
            decoder_output = self.segment_decoder(decoder_input, decoder_init_states)
            decoder_output = self.embedding2vocab(decoder_output)
            decoder_logpy = F.log_softmax(decoder_output, dim=2)
            
            decoder_target = x[j_start:j_end, :]
            
            target_logpy = decoder_logpy[:-1, :, :].gather(dim=2, index=decoder_target.unsqueeze(-1)).squeeze(-1)
            
            tmp_logpy = torch.zeros_like(target_logpy[0])
            
            #j is a temporary j_end
            for j in range(j_start, j_end):
                tmp_logpy = tmp_logpy + target_logpy[j - j_start, :]
                if j > j_start:
                    tmp_logpy = tmp_logpy + is_single[j, :]
                if j == j_start + 1:
                    tmp_logpy = tmp_logpy + is_single[j_start, :]
                logpy[j_start][j - j_start] = tmp_logpy + decoder_logpy[j - j_start + 1, :, self.config.eos_id]
        
        if mode == 'unsupervised' or mode == 'supervised':
            
            #total_log_probability
            alpha = [neg_inf_vector for _ in range(max_length - 1)]

            #log probability for generate <bos> at beginning is 0
            alpha[0] = torch.zeros_like(neg_inf_vector)

            for j_end in range(1, max_length - 1):
                logprobs = []
                for j_start in range(max(0, j_end - self.config.max_segment_length), j_end):                    
                    logprobs.append(alpha[j_start] + logpy[j_start + 1][j_end - j_start - 1])
                alpha[j_end] =  torch.logsumexp(torch.stack(logprobs), dim=0)
            
            NLL_loss = 0.0
            total_length = 0
            
            alphas = torch.stack(alpha)
            index = (torch.LongTensor(lengths) - 2).view(1, -1)
            
            if alphas.is_cuda:
                index = index.cuda()
            
            NLL_loss = - torch.gather(input=alphas, dim=0, index=index)
            
            assert NLL_loss.view(-1).size(0) == batch_size
            
            total_length += sum(lengths) - 2 * batch_size

            normalized_NLL_loss = NLL_loss.sum() / float(total_length)
            
            if mode == 'supervised':
                # Get extra loss for supervised segmentation
                
                supervised_NLL_loss = 0.0
                total_length = 0
                
                for i in range(batch_size):
                    j_start = 1
                    for j_length in segments[i]:
                        if j_length <= self.config.max_segment_length:
                            supervised_NLL_loss = supervised_NLL_loss - logpy[j_start][j_length - 1][i]
                            total_length += j_length
                        j_start += j_length
                        
                normalized_supervised_NLL_loss = supervised_NLL_loss / float(total_length)
                
                normalized_NLL_loss = normalized_supervised_NLL_loss * 0.1 + normalized_NLL_loss
                
            return normalized_NLL_loss
          
        elif mode == 'decode':
            ret = []

            #<BOS> is at the begin of sentence,
            #<PUNC> is at the end of sentece
            #sentence_length = true_length + 2 
            
            for i in range(batch_size):
                alpha = [-loginf]*(lengths[i] - 1)
                prev = [-1]*(lengths[i] - 1)
                alpha[0] = 0.0
                for j_end in range(1, lengths[i] - 1):
                    for j_start in range(max(0, j_end - self.config.max_segment_length), j_end):
                        logprob = alpha[j_start] + logpy[j_start + 1][j_end - j_start - 1][i].item()
                        if logprob > alpha[j_end]:
                            alpha[j_end] = logprob
                            prev[j_end] = j_start
                
                j_end = lengths[i] - 2
                segment_lengths = []
                while j_end > 0:
                    prev_j = prev[j_end]
                    segment_lengths.append(j_end - prev_j)
                    j_end = prev_j

                segment_lengths = segment_lengths[::-1]

                ret.append(segment_lengths)

            return ret

        else:
            raise ValueError('Mode %s not supported' % mode)

class ContextEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, layer_number, dropout_rate):
        super(ContextEncoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.rnn = nn.LSTM(
            input_size=hidden_size, 
            hidden_size=hidden_size,
            num_layers=layer_number,
            dropout=dropout_rate
        )
        #num_layers * num_directions, batch, hidden_size
        self.h_init_state = nn.Parameter(torch.zeros(layer_number, 1, hidden_size))
        self.c_init_state = nn.Parameter(torch.zeros(layer_number, 1, hidden_size))
        if self.input_size != self.hidden_size:
            self.embedding2hidden = nn.Linear(config.input_size, config.hidden_size)
        
    def forward(self, rnn_input, lengths, init_states):
        if self.input_size != self.hidden_size:
            rnn_input = self.embedding2hidden(rnn_input)
        rnn_input = nn.utils.rnn.pack_padded_sequence(rnn_input, lengths)
        output, _ = self.rnn(rnn_input, init_states)
        output, _ = nn.utils.rnn.pad_packed_sequence(output)
        return output
        
    def get_init_states(self, batch_size):
        return (self.h_init_state.expand(-1, batch_size, -1).contiguous(),
                self.c_init_state.expand(-1, batch_size, -1).contiguous())
        
class SegmentDecoder(nn.Module):
    def __init__(self, hidden_size, output_size, layer_number, dropout_rate):
        super(SegmentDecoder, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.rnn = nn.LSTM(
            input_size=hidden_size, 
            hidden_size=hidden_size,
            num_layers=layer_number,
            dropout=dropout_rate
        )
        if self.hidden_size != self.output_size:
            self.hidden2embedding = nn.Linear(hidden_size, output_size)
        self.output_dropout = nn.Dropout(p=dropout_rate)
        
    def forward(self, rnn_input, init_states):
        output, _ = self.rnn(rnn_input, init_states)
        if self.hidden_size != self.output_size:
            output = self.hidden2embedding(output)
        output = self.output_dropout(output)
        
        return output
