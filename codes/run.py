#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SLM Training and Decoding
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import six
import logging
import os
import random
import subprocess

import numpy as np

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils import data

import model
from tokenization import CWSTokenizer
from dataloader import InputDataset, OneShotIterator

def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description="Training and decoding SLM (segmental language model)",
        usage="train.py [<args>] [-h | --help]"
    )
    
    parser.add_argument("--use_cuda", action='store_true', help="Whether to use gpu.")
    
    # mode
    parser.add_argument("--do_unsupervised", action='store_true', help="Whether to run unsupervised training.")
    
    parser.add_argument("--do_supervised", action='store_true', help="Whether to run supervised training.")
    
    parser.add_argument("--do_valid", action='store_true', help="Whether to do validation during training.")
    
    parser.add_argument("--do_predict", action='store_true', help="Whether to run prediction.")
    
    # general setting
    parser.add_argument("--unsegmented", type=str, nargs="+", help="Path of unsegmented input file")
    
    parser.add_argument("--segmented", type=str, nargs="+", help="Path of segmented input file")
    
    parser.add_argument("--predict_inputs", type=str, nargs='+', help="Path to prediction input file")
    parser.add_argument("--valid_inputs", type=str, nargs='+', help="Path to validation input file")
    parser.add_argument("--predict_output", type=str, help="Path to prediction result")
    parser.add_argument("--valid_output", type=str, help="Path to validation output file")
    
    parser.add_argument("--max_seq_length", type=int, default=32, help="The maximum input sequence length")
    
    parser.add_argument("--vocab_file", type=str, required=True, help="Path to vocabulary")
    parser.add_argument("--config_file", type=str, required=True, help="Path to SLM configuration file")
    
    parser.add_argument("--init_embedding_path", type=str, default=None, help="Path to init word embedding")
    parser.add_argument("--init_checkpoint", type=str, default=None, help="Path to init checkpoint")
    parser.add_argument("--save_path", type=str, help="Path to saving checkpoint")

    # training setting
    parser.add_argument("--gradient_clip", type=float, default=0.1)
    parser.add_argument("--supervised_lambda", type=float, default=1.0, 
                        help="supervised weight, total_loss = unsupervised_loss + supervised_lambda * supervised_loss")
    parser.add_argument("--sgd_learning_rate", type=float, default=16.0)
    parser.add_argument("--adam_learning_rate", type=float, default=0.005)
    
    parser.add_argument("--unsupervised_batch_size", type=int, default=6000)
    parser.add_argument("--supervised_batch_size", type=int, default=1000)
    parser.add_argument("--valid_batch_size", type=int, default=16)
    parser.add_argument("--predict_batch_size", type=int, default=16)
    
    # training step setting
    parser.add_argument("--save_every_steps", type=int, default=400)
    parser.add_argument("--log_every_steps", type=int, default=100)
    parser.add_argument("--warm_up_steps", type=int, default=800)
    parser.add_argument("--train_steps", type=int, default=4000)
    
    # other setting
    parser.add_argument("--cpu_num", type=int, default=4)
    parser.add_argument("--segment_token", type=str, default='  ', help="Segment token")
    parser.add_argument("--english_token", type=str, default='<ENG>', help="token for English characters")
    parser.add_argument("--number_token", type=str, default='<NUM>', help="token for numbers")
    parser.add_argument("--punctuation_token", type=str, default='<PUNC>', help="token for punctuations")
    parser.add_argument("--bos_token", type=str, default='<BOS>', help="token for begin of sentence")
    parser.add_argument("--eos_token", type=str, default='</s>', help="token for begin of sentence")
    
    return parser.parse_args(args)

def set_logger(args):
    '''
    Write logs to checkpoint and console
    '''

    if args.do_unsupervised or args.do_supervised:
        log_file = os.path.join(args.save_path or args.init_checkpoint, 'train.log')
    else:
        log_file = os.path.join(args.save_path or args.init_checkpoint, 'test.log')

    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S',
        filename=log_file,
        filemode='w'
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

def main(args):
    if not args.do_unsupervised and not args.do_supervised and not args.do_predict:
        raise ValueError("At least one of `do_unsupervised`, `do_supervised` or `do_predict' must be True.")
    
    if args.do_unsupervised and not args.unsegmented:
        raise ValueError("Unsupervised learning requires unsegmented data.")

    if args.do_supervised and not args.segmented:
        raise ValueError("Supervised learning requires segmented data.")
        
    if args.do_predict and not (args.predict_inputs or not args.predict_output or not args.init_checkpoint):
        raise ValueError("Predicion requires init_checkpoint, inputs, and output.")
        
    if (args.do_unsupervised or args.do_supervised) and args.save_path is None:
        raise ValueError('Where do you want to save your trained model?')
    
    if args.save_path and not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    
    set_logger(args)
    
    logging.info(str(args))
    
    tokenizer = CWSTokenizer(
        vocab_file=args.vocab_file,
        max_seq_length=args.max_seq_length,
        segment_token=args.segment_token,
        english_token=args.english_token,
        number_token=args.number_token,
        punctuation_token=args.punctuation_token,
        bos_token=args.bos_token,
        eos_token=args.eos_token
    )
    
    if args.init_embedding_path:
        logging.info('Loading init embedding from %s...' % args.init_embedding_path)
        init_embedding = np.load(args.init_embedding_path)
    else:
        logging.info('Ramdomly Initializing character embedding...')
        init_embedding = None
        
    slm_config = model.SLMConfig.from_json_file(args.config_file)
    
    logging.info('Config Info:\n%s' % slm_config.to_json_string())
    
    slm = model.SegmentalLM(
        config=slm_config,
        init_embedding=init_embedding
    )
    
    logging.info('Model Info:\n%s' % slm)
    
    if args.use_cuda:
        slm = slm.cuda()
    
    logs = []
    
    if args.do_unsupervised or args.do_supervised:
        
        if args.do_unsupervised:
            logging.info('Prepare unsupervised dataloader')
            unsupervsied_dataset = InputDataset(
                args.unsegmented, 
                tokenizer,
                is_training=True,
                batch_token_size=args.unsupervised_batch_size
            )
            unsupervised_dataloader = data.DataLoader(
                unsupervsied_dataset,  
                num_workers=args.cpu_num,
                batch_size=1, 
                shuffle=False,
                collate_fn=InputDataset.single_collate
            )
            unsupervised_data_iterator = OneShotIterator(unsupervised_dataloader)
            
        if args.do_supervised:
            logging.info('Prepare supervised dataloader')
            supervsied_dataset = InputDataset(
                args.segmented, 
                tokenizer,
                is_training=True,
                batch_token_size=args.supervised_batch_size
            )
            supervised_dataloader = data.DataLoader(
                supervsied_dataset, 
                num_workers=args.cpu_num,
                batch_size=1,
                shuffle=False,
                collate_fn=InputDataset.single_collate
            )
            supervised_data_iterator = OneShotIterator(supervised_dataloader)

        if args.do_valid:
            logging.info('Prepare validation dataloader')
            valid_dataset = InputDataset(args.valid_inputs, tokenizer)
            valid_dataloader = data.DataLoader(
                dataset=valid_dataset,
                shuffle=False,
                batch_size=args.valid_batch_size,
                num_workers=0,
                collate_fn=InputDataset.padding_collate
            )
            
        adam_optimizer = optim.Adam(slm.parameters(), lr=args.adam_learning_rate, betas=(0.9, 0.998))
        lr_lambda = lambda step: 1 if step < 0.8 * args.train_steps else 0.1
        scheduler = optim.lr_scheduler.LambdaLR(adam_optimizer, lr_lambda=lr_lambda)
        
        if args.init_checkpoint:
            # Restore model from checkpoint directory
            logging.info('Loading checkpoint %s...' % args.init_checkpoint)
            checkpoint = torch.load(os.path.join(args.init_checkpoint, 'best-checkpoint'))
            global_step = checkpoint['global_step']
            best_F_score = checkpoint['best_F_score']
            slm.load_state_dict(checkpoint['model_state_dict'])
            adam_optimizer.load_state_dict(checkpoint['adam_optimizer'])
        else:
            logging.info('Ramdomly Initializing SLM parameters...')
            global_step = 0
            best_F_score = 0
        
        for step in range(global_step):
            scheduler.step()
        
        for step in range(global_step, args.train_steps):
            scheduler.step()
            
            slm.train()
            slm.zero_grad()
            
            log = {}
            
            if args.do_unsupervised:
                x_batch, seq_len_batch, uchars_batch, segments_batch = next(unsupervised_data_iterator)
                if args.use_cuda:
                    x_batch = x_batch.cuda()
                loss = slm(x_batch, seq_len_batch, mode='unsupervised')
                log['unsupervised_loss'] = loss.item()
            
            elif args.do_supervised:
                x_batch, seq_len_batch, uchars_batch, segments_batch = next(supervised_data_iterator)
                if args.use_cuda:
                    x_batch = x_batch.cuda()
                loss = slm(x_batch, seq_len_batch, segments_batch, mode='supervised')
                log['supervised_loss'] = loss.item()

            logs.append(log)
            
            loss.backward()
            nn.utils.clip_grad_norm_(slm.parameters(), args.gradient_clip)
            
            if step > args.warm_up_steps:
                adam_optimizer.step()
            else:
                #do manually SGD
                for p in slm.parameters():
                    if p.grad is not None:
                        p.data.add_(-args.sgd_learning_rate, p.grad.data)

            if step % args.log_every_steps == 0:
                logging.info("global_step = %s" % step)
                if len(logs) > 0:
                    for key in logs[0]:
                        logging.info("%s = %f" % (key, sum([log[key] for log in logs])/len(logs)))
                else:
                    logging.info("Currently no metrics available")
                logs = []

            if (step % args.save_every_steps == 0) or (step == args.train_steps - 1):
                logging.info('Saving checkpoint %s...' % args.save_path)
                slm_config.to_json_file(os.path.join(args.save_path, 'config.json'))
                torch.save({
                    'global_step': step,
                    'best_F_score': best_F_score,
                    'model_state_dict': slm.state_dict(),
                    'adam_optimizer': adam_optimizer.state_dict()
                },
                    os.path.join(args.save_path, 'checkpoint')
                )

                if args.do_valid:
                    slm.eval()

                    with open(args.valid_output, 'w') as fout:
                        with torch.no_grad():
                            for x_batch, seq_len_batch, uchars_batch, segments_batch, restore_orders in valid_dataloader:
                                if args.use_cuda:
                                    x_batch = x_batch.cuda()
                                segments_batch = slm(x_batch, seq_len_batch, mode='decode')
                                for i in restore_orders:
                                    uchars, segments = uchars_batch[i], segments_batch[i]
                                    fout.write(tokenizer.restore(uchars, segments))
                    
                    eval_command = "bash run.sh valid %s" % ' '.join(args.save_path.split('/')[-1].rsplit('-'))
                    logging.info('Bash Command: %s' % eval_command)
                    out = subprocess.Popen(eval_command.split(' '), 
                                           stdout=subprocess.PIPE, 
                                           stderr=subprocess.STDOUT)
                    stdout, stderr = out.communicate()
                    stdout = stdout.decode("utf-8")
                    logging.info('Validation results:\n%s' % stdout)
                    
                    for line in stdout.split('\n'):
                        if line[:len('=== F MEASURE:')] == '=== F MEASURE:':
                            F_score = float(line.split('\t')[-1])
                    
                if (not args.do_valid) or (F_score > best_F_score):
                    best_F_score = F_score
                    logging.info('Overwriting best checkpoint....')
                    os.system('cp %s %s' % (os.path.join(args.save_path, 'checkpoint'), 
                                            os.path.join(args.save_path, 'best-checkpoint')))
                    
    if args.do_predict:

        logging.info('Prepare prediction dataloader')

        predict_dataset = InputDataset(args.predict_inputs, tokenizer)

        predict_dataloader = data.DataLoader(
            dataset=predict_dataset,
            shuffle=False,
            batch_size=args.predict_batch_size,
            num_workers=0,
            collate_fn=InputDataset.padding_collate
        )

        # Restore model from best checkpoint
        logging.info('Loading checkpoint %s...' % (args.init_checkpoint or args.save_path))
        checkpoint = torch.load(os.path.join(args.init_checkpoint or args.save_path, 'best-checkpoint'))
        step = checkpoint['global_step']
        slm.load_state_dict(checkpoint['model_state_dict'])

        logging.info('Global step of best-checkpoint: %s' % step)

        slm.eval()

        with open(args.predict_output, 'w') as fout:
            with torch.no_grad():
                for x_batch, seq_len_batch, uchars_batch, segments_batch, restore_orders in predict_dataloader:
                    if args.use_cuda:
                        x_batch = x_batch.cuda()
                    segments_batch = slm(x_batch, seq_len_batch, mode='decode')
                    for i in restore_orders:
                        uchars, segments = uchars_batch[i], segments_batch[i]
                        fout.write(tokenizer.restore(uchars, segments))

        eval_command = "bash run.sh eval %s" % ' '.join((
          args.init_checkpoint or args.save_path).split('/')[-1].rsplit('-'))
        logging.info('Bash Command: %s' % eval_command)
        out = subprocess.Popen(eval_command.split(' '), 
                               stdout=subprocess.PIPE, 
                               stderr=subprocess.STDOUT)
        stdout, stderr = out.communicate()
        stdout = stdout.decode("utf-8")
        logging.info('Test evaluation results:\n%s' % stdout)
                    
if __name__ == "__main__":
    main(parse_args())