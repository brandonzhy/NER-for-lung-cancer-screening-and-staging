import os, csv
from torch.utils.data import Dataset, random_split
from typing import List, Dict, Union
import re
import logging
from pathlib import Path
from collections import Counter
import torch.utils.data.dataloader
from torch.utils.data.dataset import Subset, ConcatDataset
from NLPer.data import Sentence, Corpus, Token, BaseDataset
from NLPer.file_utils import cached_path



log = logging.getLogger("NLPer")


def convert_sent_to_feature(tokenizer, sent_text, max_sequence_length):
    tokens = []
    # input_type_ids = []
    tokens.append("[CLS]")
    # input_type_ids.append(0)
    for token in sent_text:
        tokens.append(token)
        # input_type_ids.append(0)
    tokens.append("[SEP]")
    # input_type_ids.append(0)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    # The mask has 1 for real tokens and 0 for padding tokens. Only real tokens are attended to.
    # print('length of input_ids is :',len(input_ids))
    # print('length of tokens is :', len(tokens))

    input_mask = [1] * len(input_ids)
    input_type_ids = [0] * len(input_ids)

    # Zero-pad up to the sequence length.
    while len(input_ids) < max_sequence_length:
        input_ids.append(0)
        input_mask.append(0)
        input_type_ids.append(0)
    return input_ids, input_mask, input_type_ids


class ColumnCorpus(Corpus):
    def __init__(
        self,
        data_folder: Union[str, Path],
        column_format: Dict[int, str],
        train_file=None,
        test_file=None,
        dev_file=None,

        tag_to_bioes=None,
        comment_symbol:str=None,
        in_memory: bool=True,
    ):
        """
        Instantiates a Corpus from CoNLL column-formatted task data such as CoNLL03 or CoNLL2000.

        :param data_folder: base folder with the task data
        :param column_format: a map specifying the column format
        :param train_file: the name of the train file
        :param test_file: the name of the test file
        :param dev_file: the name of the dev file, if None, dev data is sampled from train
        :param tag_to_bioes: whether to convert to BIOES tagging scheme
        :return: a Corpus with annotated train, dev and test data
        """

        if type(data_folder) == str:
            data_folder: Path = Path(data_folder)

        if train_file is not None:
            train_file = data_folder / train_file
        if test_file is not None:
            test_file = data_folder / test_file
        if dev_file is not None:
            dev_file = data_folder / dev_file

        # automatically identify train / test / dev files
        if train_file is None:
            for file in data_folder.iterdir():
                file_name = file.name
                if file_name.endswith(".gz"):
                    continue
                if "train" in file_name and not "54019" in file_name:
                    train_file = file
                if "dev" in file_name:
                    dev_file = file
                if "testa" in file_name:
                    dev_file = file
                if "testb" in file_name:
                    test_file = file

            # if no test file is found, take any file with 'test' in name
            if test_file is None:
                for file in data_folder.iterdir():
                    file_name = file.name
                    if file_name.endswith(".gz"):
                        continue
                    if "test" in file_name:
                        test_file = file

        log.info("Reading data from {}".format(data_folder))
        log.info("Train: {}".format(train_file))
        log.info("Dev: {}".format(dev_file))
        log.info("Test: {}".format(test_file))

        # get train data
        train = ColumnDataset(
            train_file,
            column_format,
            tag_to_bioes,
            comment_symbol=comment_symbol,
            in_memory=in_memory,
        )

        # read in test file if exists, otherwise sample 10% of train data as test dataset
        if test_file is not None:
            test = ColumnDataset(
                test_file,
                column_format,
                tag_to_bioes,
                comment_symbol=comment_symbol,
                in_memory=in_memory,
            )
        else:
            train_length = len(train)
            test_size: int = round(train_length / 10)
            splits = random_split(train, [train_length - test_size, test_size])
            train = splits[0]
            test = splits[1]

        # read in dev file if exists, otherwise sample 10% of train data as dev dataset
        if dev_file is not None:
            dev = ColumnDataset(
                dev_file,
                column_format,
                tag_to_bioes,
                comment_symbol=comment_symbol,
                in_memory=in_memory,
            )
        else:
            train_length = len(train)
            dev_size: int = round(train_length / 10)
            splits = random_split(train, [train_length - dev_size, dev_size])
            train = splits[0]
            dev = splits[1]

        super(ColumnCorpus, self).__init__(train, dev, test, name=data_folder.name)
    def get_all_tokens(self) -> List[str]:
        tokens = list(map((lambda s: s.tokens), self.pre_train))
        tokens = [token for sublist in tokens for token in sublist]
        return list(map((lambda t: t.text), tokens))
    def get_tokens_frequence(self):
        tokens_and_frequencies = Counter(self.get_all_tokens())
        return tokens_and_frequencies



class ColumnDataset(BaseDataset):
    def __init__(
        self,
        path_to_column_file: Path,
        column_name_map: Dict[int, str],
        tag_to_bioes: str = None,
        comment_symbol: str = None,
        in_memory: bool = True,
    ):
        assert path_to_column_file.exists()
        self.path_to_column_file = path_to_column_file
        self.tag_to_bioes = tag_to_bioes
        self.column_name_map = column_name_map
        self.comment_symbol = comment_symbol
        self.input_ids = None,
        self.attention_masks =None,
        self.input_type_ids =None
        self.max_length = 0
        # store either Sentence objects in memory, or only file offsets
        self.in_memory = in_memory
        if self.in_memory:
            self.sentences: List[Sentence] = []
        else:
            self.indices: List[int] = []

        self.total_sentence_count: int = 0

        # most data sets have the token text in the first column, if not, pass 'text' as column
        self.text_column: int = 0
        for column in self.column_name_map:
            if column_name_map[column] == "text":
                self.text_column = column

        # determine encoding of text file
        encoding = "utf-8"
        try:
            lines: List[str] = open(str(path_to_column_file), encoding="utf-8").read(
                10
            ).strip().split("\n")
        except:
            log.info(
                'UTF-8 can\'t read: {} ... using "latin-1" instead.'.format(
                    path_to_column_file
                )
            )
            encoding = "latin1"

        sentence: Sentence = Sentence()
        with open(str(self.path_to_column_file), encoding=encoding) as f:

            line = f.readline()
            position = 0

            while line:

                if self.comment_symbol is not None and line.startswith(comment_symbol):
                    line = f.readline()
                    continue
                # If it is a blank line, it means the end of a sentence, and add it to sentencess
                if line.isspace():
                    if len(sentence) > 0:
                        sentence.infer_space_after()
                        if self.in_memory:
                            if self.tag_to_bioes is not None:
                                sentence.convert_tag_scheme(
                                    tag_type=self.tag_to_bioes, target_scheme="iobes"
                                )
                            self.sentences.append(sentence)
                        else:
                            self.indices.append(position)
                            position = f.tell()
                        self.total_sentence_count += 1
                        self.max_length = len(sentence) if len(sentence) > self.max_length else self.max_length

                    sentence: Sentence = Sentence()

                else:
                    fields: List[str] = re.split("\s+", line)
                    token = Token(fields[self.text_column])
                    for column in column_name_map:
                        if len(fields) > column:
                            if column != self.text_column:
                                token.add_tag(
                                    self.column_name_map[column], fields[column]
                                )

                    sentence.add_token(token)

                line = f.readline()

        if len(sentence.tokens) > 0:
            sentence.infer_space_after()
            self.max_length = len(sentence.tokens) if len(sentence.tokens) > self.max_length else self.max_length
            if self.in_memory:
                self.sentences.append(sentence)

            else:
                self.indices.append(position)
            self.total_sentence_count += 1


    def is_in_memory(self) -> bool:
        return self.in_memory

    def __len__(self):
        return self.total_sentence_count

    def __getitem__(self, index: int = 0) -> Sentence:

        if self.in_memory:
            sentence = self.sentences[index]

        else:
            with open(str(self.path_to_column_file), encoding="utf-8") as file:
                file.seek(self.indices[index])
                line = file.readline()
                sentence: Sentence = Sentence()
                while line:
                    if self.comment_symbol is not None and line.startswith("#"):
                        line = file.readline()
                        continue

                    if line.strip().replace("﻿", "") == "":
                        if len(sentence) > 0:
                            sentence.infer_space_after()

                            if self.tag_to_bioes is not None:
                                sentence.convert_tag_scheme(
                                    tag_type=self.tag_to_bioes, target_scheme="iobes"
                                )
                            break
                    else:
                        fields: List[str] = re.split("\s+", line)
                        token = Token(fields[self.text_column])
                        for column in self.column_name_map:
                            if len(fields) > column:
                                if column != self.text_column:
                                    token.add_tag(
                                        self.column_name_map[column], fields[column]
                                    )

                        sentence.add_token(token)
                    line = file.readline()

        return sentence






class DataLoader(torch.utils.data.dataloader.DataLoader):
    def __init__(
        self,
        dataset,
        batch_size=1,
        shuffle=False,
        sampler=None,
        batch_sampler=None,
        num_workers=8,
        drop_last=False,
        timeout=0,
        worker_init_fn=None,
    ):

        # in certain cases, multi-CPU data loading makes no sense and slows
        # everything down. For this reason, we detect if a dataset is in-memory:
        # if so, num_workers is set to 0 for faster processing
        flair_dataset = dataset
        while True:
            if type(flair_dataset) is Subset:
                flair_dataset = flair_dataset.dataset
            elif type(flair_dataset) is ConcatDataset:
                flair_dataset = flair_dataset.datasets[0]
            else:
                break

        if type(flair_dataset) is list:
            num_workers = 0
        elif isinstance(flair_dataset, BaseDataset) and flair_dataset.is_in_memory():
            num_workers = 0

        super(DataLoader, self).__init__(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            collate_fn=list,
            drop_last=drop_last,
            timeout=timeout,
            worker_init_fn=worker_init_fn,
        )
