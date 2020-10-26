# -*- encoding:utf8 -*-
#!/usr/bin/python3
import argparse
from NLPer.data import Corpus
from NLPer.datasets import ColumnCorpus
from NLPer.embeddings import XLNetEmbeddings, BertEmbeddings, StackedEmbeddings, FastTextEmbeddings
from NLPer.models.sequence_tagger_model import SequenceTagger
import NLPer
from NLPer.trainers import ModelTrainer
from transformers import  AdamW
import  torch
import numpy as np
from NLPer.training_utils import get_final_test_score
import os,sys

def args_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", default=None, type=str, help="the input data dir")

    parser.add_argument("--embedding_type", type=str, default='BERT',
                        help="BERT,FastText,stacked")
    parser.add_argument("--tag_type", type=str, default='ner',
                        help="BERT,FastText,stacked")
    parser.add_argument("--train_batch_size", default=32, type=int)
    parser.add_argument("--eval_batch_size", default=32, type=int)
    parser.add_argument("--checkpoint", default=False, type=str)
    parser.add_argument("--learning_rate", default=2e-5, type=float)
    parser.add_argument("--num_train_epochs", default=46, type=int)
    parser.add_argument("--seed", type=int, default=3306)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--tolerance', type=float, default=1e-5)
    parser.add_argument('--dropout', type=float, default=1e-5)
    parser.add_argument('--use_rnn', type=bool, default=True)
    parser.add_argument('--use_crf', type=bool, default=True)
    parser.add_argument('--rnn_layers', type=int, default=2)
    parser.add_argument('--rnn_hidden_dim', type=int, default=2)
    parser.add_argument('--use_rnn', type=bool, default=False)
    parser.add_argument('--transformer_layers', type=int, default=2)
    parser.add_argument('--transformer_headss', type=int, default=2)
    parser.add_argument("--output_dir", type=str, default=None)

    args = parser.parse_args()


    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    return args
if __name__ == "__main__":
    args_config = args_parser()
    columns = {0: 'text', 1:  'ner'}


    corpus: Corpus = ColumnCorpus(args_config.data_dir, columns,
                                  train_file='train.txt',
                                  test_file='test.txt',
                                  dev_file='valid.txt',
                                  )
    tag_dictionary = corpus.make_tag_dictionary(args_config.tag_type)
    if args_config.embedding_type== 'BERT':
        embedding_types =  BertEmbeddings('./pre-trained-models/chinese-bert_chinese_wwm_pytorch')

    else:
        embedding_types =  FastTextEmbeddings('./pre-trained-models/FastText/zh-wiki-fasttext-300d-1M')

    tagger: SequenceTagger = SequenceTagger(hidden_size=args_config.rnn_hidden_dim,
                                            embeddings=embedding_types,
                                            tag_dictionary=tag_dictionary,
                                            use_rnn=args_config.use_rnn,

                                            rnn_layers=args_config.rnn_layer,
                                            tag_type=args_config.tag_type,
                                            dropout=args_config.dropout,
                                            use_crf=args_config.usr_crf)


    trainer: ModelTrainer = ModelTrainer(
        tagger,
        corpus,
        optimizer=AdamW,
    )
    trainer.train(
        args_config.output_dir,
        max_epochs=args_config.num_train_epochs,
        evaluation_metric=NLPer.training_utils.EvaluationMetric.MACRO_F1_SCORE,
        patience=args_config.patience ,
        tolerance = args_config.tolerance,
        learning_rate=args_config.learning_rate,
        mini_batch_size=args_config.train_batch_size,
        eval_mini_batch_size = args_config.eval_batch_size,
        checkpoint=False)
    get_final_test_score( args_config['train_path'] )
    del tagger
    del trainer
