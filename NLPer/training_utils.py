import itertools
import random
import logging
from collections import defaultdict
from enum import Enum
from pathlib import Path
from typing import List
from NLPer.data import Dictionary, Sentence
from functools import reduce
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr, spearmanr
import os,re
import pandas as pd
from abc import abstractmethod


class Result(object):
    def __init__(
        self, main_score: float, log_header: str, log_line: str, detailed_results: str
    ):
        self.main_score: float = main_score
        self.log_header: str = log_header
        self.log_line: str = log_line
        self.detailed_results: str = detailed_results


class Metric(object):
    def __init__(self, name):
        self.name = name

        self._tps = defaultdict(int)
        self._fps = defaultdict(int)
        self._tns = defaultdict(int)
        self._fns = defaultdict(int)

    def add_tp(self, class_name):
        self._tps[class_name] += 1

    def add_tn(self, class_name):
        self._tns[class_name] += 1

    def add_fp(self, class_name):
        self._fps[class_name] += 1

    def add_fn(self, class_name):
        self._fns[class_name] += 1

    def get_tp(self, class_name=None):
        if class_name is None:
            return sum([self._tps[class_name] for class_name in self.get_classes()])
        return self._tps[class_name]

    def get_tn(self, class_name=None):
        if class_name is None:
            return sum([self._tns[class_name] for class_name in self.get_classes()])
        return self._tns[class_name]

    def get_fp(self, class_name=None):
        if class_name is None:
            return sum([self._fps[class_name] for class_name in self.get_classes()])
        return self._fps[class_name]

    def get_fn(self, class_name=None):
        if class_name is None:
            return sum([self._fns[class_name] for class_name in self.get_classes()])
        return self._fns[class_name]

    def precision(self, class_name=None):
        if self.get_tp(class_name) + self.get_fp(class_name) > 0:
            return round(
                self.get_tp(class_name)
                / (self.get_tp(class_name) + self.get_fp(class_name)),
                4,
            )
        return 0.0

    def recall(self, class_name=None):
        if self.get_tp(class_name) + self.get_fn(class_name) > 0:
            return round(
                self.get_tp(class_name)
                / (self.get_tp(class_name) + self.get_fn(class_name)),
                4,
            )
        return 0.0

    def f_score(self, class_name=None):
        if self.precision(class_name) + self.recall(class_name) > 0:
            return round(
                2
                * (self.precision(class_name) * self.recall(class_name))
                / (self.precision(class_name) + self.recall(class_name)),
                4,
            )
        return 0.0

    def accuracy(self, class_name=None):
        if (
            self.get_tp(class_name) + self.get_fp(class_name) + self.get_fn(class_name)
            > 0
        ):
            return round(
                (self.get_tp(class_name))
                / (
                    self.get_tp(class_name)
                    + self.get_fp(class_name)
                    + self.get_fn(class_name)
                ),
                4,
            )
        return 0.0

    def micro_avg_f_score(self):
        return self.f_score(None)

    def macro_avg_f_score(self):
        class_f_scores = [self.f_score(class_name) for class_name in self.get_classes()]
        if len(class_f_scores) == 0:
            return 0.0
        macro_f_score = sum(class_f_scores) / len(class_f_scores)
        return macro_f_score

    def micro_avg_accuracy(self):
        return self.accuracy(None)

    def macro_avg_accuracy(self):
        class_accuracy = [
            self.accuracy(class_name) for class_name in self.get_classes()
        ]

        if len(class_accuracy) > 0:
            return round(sum(class_accuracy) / len(class_accuracy), 4)

        return 0.0

    def get_classes(self) -> List:
        all_classes = set(
            itertools.chain(
                *[
                    list(keys)
                    for keys in [
                        self._tps.keys(),
                        self._fps.keys(),
                        self._tns.keys(),
                        self._fns.keys(),
                    ]
                ]
            )
        )
        all_classes = [
            class_name for class_name in all_classes if class_name is not None
        ]
        all_classes.sort()
        return all_classes

    def to_tsv(self):
        return "{}\t{}\t{}\t{}".format(
            self.precision(), self.recall(), self.accuracy(), self.micro_avg_f_score()
        )

    @staticmethod
    def tsv_header(prefix=None):
        if prefix:
            return "{0}_PRECISION\t{0}_RECALL\t{0}_ACCURACY\t{0}_F-SCORE".format(prefix)

        return "PRECISION\tRECALL\tACCURACY\tF-SCORE"

    @staticmethod
    def to_empty_tsv():
        return "\t_\t_\t_\t_"

    def __str__(self):
        all_classes = self.get_classes()
        all_classes = [None] + all_classes
        all_lines = [
            "{0:<10}\ttp: {1} - fp: {2} - fn: {3} - tn: {4} - precision: {5:.4f} - recall: {6:.4f} - accuracy: {7:.4f} - f1-score: {8:.4f}".format(
                self.name if class_name is None else class_name,
                self.get_tp(class_name),
                self.get_fp(class_name),
                self.get_fn(class_name),
                self.get_tn(class_name),
                self.precision(class_name),
                self.recall(class_name),
                self.accuracy(class_name),
                self.f_score(class_name),
            )
            for class_name in all_classes
        ]
        return "\n".join(all_lines)


class MetricRegression(object):
    def __init__(self, name):
        self.name = name

        self.true = []
        self.pred = []

    def mean_squared_error(self):
        return mean_squared_error(self.true, self.pred)

    def mean_absolute_error(self):
        return mean_absolute_error(self.true, self.pred)

    def pearsonr(self):
        return pearsonr(self.true, self.pred)[0]

    def spearmanr(self):
        return spearmanr(self.true, self.pred)[0]

    ## dummy return to fulfill trainer.train() needs
    def micro_avg_f_score(self):
        return self.mean_squared_error()

    def to_tsv(self):
        return "{}\t{}\t{}\t{}".format(
            self.mean_squared_error(),
            self.mean_absolute_error(),
            self.pearsonr(),
            self.spearmanr(),
        )

    @staticmethod
    def tsv_header(prefix=None):
        if prefix:
            return "{0}_MEAN_SQUARED_ERROR\t{0}_MEAN_ABSOLUTE_ERROR\t{0}_PEARSON\t{0}_SPEARMAN".format(
                prefix
            )

        return "MEAN_SQUARED_ERROR\tMEAN_ABSOLUTE_ERROR\tPEARSON\tSPEARMAN"

    @staticmethod
    def to_empty_tsv():
        return "\t_\t_\t_\t_"

    def __str__(self):
        line = "mean squared error: {0:.4f} - mean absolute error: {1:.4f} - pearson: {2:.4f} - spearman: {3:.4f}".format(
            self.mean_squared_error(),
            self.mean_absolute_error(),
            self.pearsonr(),
            self.spearmanr(),
        )
        return line


class EvaluationMetric(Enum):
    MICRO_ACCURACY = "micro-average accuracy"
    MICRO_F1_SCORE = "micro-average f1-score"
    MACRO_ACCURACY = "macro-average accuracy"
    MACRO_F1_SCORE = "macro-average f1-score"
    MEAN_SQUARED_ERROR = "mean squared error"


class WeightExtractor(object):
    def __init__(self, directory: Path, number_of_weights: int = 10):
        self.weights_file = init_output_file(directory, "weights.txt")
        self.weights_dict = defaultdict(lambda: defaultdict(lambda: list()))
        self.number_of_weights = number_of_weights

    def extract_weights(self, state_dict, iteration):
        for key in state_dict.keys():

            vec = state_dict[key]
            weights_to_watch = min(
                self.number_of_weights, reduce(lambda x, y: x * y, list(vec.size()))
            )

            if key not in self.weights_dict:
                self._init_weights_index(key, state_dict, weights_to_watch)

            for i in range(weights_to_watch):
                vec = state_dict[key]
                for index in self.weights_dict[key][i]:
                    vec = vec[index]

                value = vec.item()

                with open(self.weights_file, "a") as f:
                    f.write("{}\t{}\t{}\t{}\n".format(iteration, key, i, float(value)))

    def _init_weights_index(self, key, state_dict, weights_to_watch):
        indices = {}

        i = 0
        while len(indices) < weights_to_watch:
            vec = state_dict[key]
            cur_indices = []

            for x in range(len(vec.size())):
                index = random.randint(0, len(vec) - 1)
                vec = vec[index]
                cur_indices.append(index)

            if cur_indices not in list(indices.values()):
                indices[i] = cur_indices
                i += 1

        self.weights_dict[key] = indices


def init_output_file(base_path: Path, file_name: str) -> Path:
    """
    Creates a local file.
    :param base_path: the path to the directory
    :param file_name: the file name
    :return: the created file
    """
    base_path.mkdir(parents=True, exist_ok=True)

    file = base_path / file_name
    open(file, "w", encoding="utf-8").close()
    return file


def convert_labels_to_one_hot(
    label_list: List[List[str]], label_dict: Dictionary
) -> List[List[int]]:
    """
    Convert list of labels (strings) to a one hot list.
    :param label_list: list of labels
    :param label_dict: label dictionary
    :return: converted label list
    """
    return [
        [1 if l in labels else 0 for l in label_dict.get_items()]
        for labels in label_list
    ]


def log_line(log):
    log.info("-" * 100)


def add_file_handler(log, output_file):
    init_output_file(output_file.parents[0], output_file.name)
    fh = logging.FileHandler(output_file)
    fh.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)-15s %(message)s")
    fh.setFormatter(formatter)
    log.addHandler(fh)
    return fh


def store_embeddings(sentences: List[Sentence], storage_mode: str):

    # if memory mode option 'none' delete everything
    if storage_mode == "none":
        for sentence in sentences:
            sentence.clear_embeddings()

    # else delete only dynamic embeddings (otherwise autograd will keep everything in memory)
    else:
        # find out which ones are dynamic embeddings
        delete_keys = []
        for name, vector in sentences[0][0]._embeddings.items():
            if sentences[0][0]._embeddings[name].requires_grad:
                delete_keys.append(name)

        # find out which ones are dynamic embeddings
        for sentence in sentences:
            sentence.clear_embeddings(delete_keys)

    # memory management - option 1: send everything to CPU
    if storage_mode == "cpu":
        for sentence in sentences:
            sentence.to("cpu")


def get_tp_count(predict_tags, gold_tags):
    tp_count = 0
    for tag in predict_tags:
        if tag in gold_tags :
            tp_count += 1
    return tp_count

    datas = pd.read_csv(file_path, sep=' ', header=None,
                        names=['Original char', 'gold_tag', 'predict_tag', 'score'])
    predict_tags = list(datas['predict_tag'])
    gold_tags = list(datas['gold_tag'])
    output_file = os.path.split(file_path)[0] + '\evaluation_result.txt'


    len_predict = len(predict_tags)
    len_gold = len(gold_tags)

    entity_static = {
        'DRUG':  {'predict': [], 'gold': []},
        'DRUG_INGREDIENT': {'predict': [], 'gold': []},
        'DISEASE':  {'predict': [], 'gold': []},
        'SYMPTOM': {'predict': [], 'gold': []},
        'SYNDROME':  {'predict': [], 'gold': []},
        'DISEASE_GROUP': {'predict': [], 'gold': []},
        'FOOD':  {'predict': [], 'gold': []},
        'FOOD_GROUP':  {'predict': [], 'gold': []},
        'PERSON_GROUP':  {'predict': [], 'gold': []},
        'DRUG_GROUP':  {'predict': [], 'gold': []},
        'DRUG_DOSAGE':  {'predict': [], 'gold': []},
        'DRUG_TASTE':  {'predict': [], 'gold': []},
        'DRUG_EFFICACY':  {'predict': [], 'gold': []}
    }
    entity_lst = list(entity_static.keys())
    pred_start = 0
    pred_end = 1
    while pred_end < len_predict :
        while pred_end < len_predict and  predict_tags[pred_end] == predict_tags[pred_end - 1]:
            pred_end += 1
        entity = predict_tags[pred_end - 1]
        if entity in entity_lst:
            entity_static[entity]['predict'].append([pred_start,pred_end])
        pred_start = pred_end
        pred_end += 1
    print('entity_static = ',entity_static)
    gold_start = 0
    gold_end = 1
    while gold_end < len_gold :
        while gold_end < len_gold and  gold_tags[gold_end] == gold_tags[gold_end - 1]:
            gold_end += 1
        entity = gold_tags[gold_end - 1]
        if entity in entity_lst:
            entity_static[entity]['gold'].append([gold_start,gold_end])
        gold_start = gold_end
        gold_end += 1
    print('entity_static = ', entity_static)
    with open(output_file, 'w', encoding='utf8') as f:
        for entity in entity_lst:
            entity_static[entity]['tp'] = get_tp_count(entity_static[entity]['predict'],entity_static[entity]['gold'])
            entity_static[entity]['count_pre'] = len(entity_static[entity]['predict'])
            entity_static[entity]['count_gold'] = len(entity_static[entity]['gold'])
            entity_static[entity]['precision_precise'] = entity_static[entity]['tp'] / entity_static[entity]['count_pre'] if   entity_static[entity]['count_pre'] != 0  else 0
            entity_static[entity]['recall_precise'] = entity_static[entity]['tp'] / entity_static[entity]['count_gold'] if   entity_static[entity]['count_gold'] != 0  else 0
            entity_static[entity]['f1_precise'] = 2*entity_static[entity]['precision_precise']* entity_static[entity]['recall_precise']/ (entity_static[entity]['precision_precise'] + entity_static[entity]['recall_precise'])if   entity_static[entity]['precision_precise'] + entity_static[entity]['recall_precise'] != 0  else 0
            f.write(entity + '*-*'
                + 'tp_precise:%.4f' % entity_static[entity]['tp']
                # + '-*-' + 'tp_blurred:%.4f' % entity_static[entity]['tp_blurred']
                + '-*-' + 'count_pre:%.4f' % entity_static[entity]['count_pre']
                + '-*-' + 'count_gold:%.4f' % entity_static[entity]['count_gold']
                # + '-*-' + 'precision_blurred:%.4f' % entity_static[entity]['precision_blurred']
                # + '-*-' + 'recall_blurred:%.4f' % entity_static[entity]['recall_blurred']
                # + '-*-' + 'f1_blurred:%.4f' % entity_static[entity]['f1_blurred']
                + '-*-' + 'precision_precise:%.4f' % entity_static[entity]['precision_precise']
                + '-*-' + 'recall_precise:%.4f' % entity_static[entity]['recall_precise']
                + '-*-' + 'f1_precise:%.4f' % entity_static[entity]['f1_precise']
                + '\n'
                )
        return entity_static

def get_entity_statics(file_path):
    datas = pd.read_csv(file_path, sep=' ', header=None,
                        names=['Original char', 'gold_tag', 'predict_tag', 'score'])
    predict_tags = list(datas['predict_tag'])
    gold_tags = list(datas['gold_tag'])
    output_file = os.path.join(os.path.split(file_path)[0] , 'evaluation_result.txt')
    entity_static = {
        '疾病和诊断':  {'count_pre': 0, 'tp_precise': 0, 'tp_blurred': 0, 'fn': 0, 'count_gold': 0},
        '影像检查': {'count_pre': 0, 'tp_precise': 0, 'tp_blurred': 0, 'fn': 0, 'count_gold': 0},
        '实验室检验': {'count_pre': 0, 'tp_precise': 0, 'tp_blurred': 0, 'fn': 0, 'count_gold': 0},
        '手术':  {'count_pre': 0, 'tp_precise': 0, 'tp_blurred': 0, 'fn': 0, 'count_gold': 0},
        '药物': {'count_pre': 0, 'tp_precise': 0, 'tp_blurred': 0, 'fn': 0, 'count_gold': 0},
        '解剖部位': {'count_pre': 0, 'tp_precise': 0, 'tp_blurred': 0, 'fn': 0, 'count_gold': 0},

    }
 
    entity_lst = list(entity_static.keys())

    len_predict = len(predict_tags)
    len_gold = len(gold_tags)
 
    index_predict = 0
    while index_predict < len_predict:
        entity_predict = predict_tags[index_predict]
        if "B-" in entity_predict:
            start_index = index_predict
            entity = entity_predict.split('-')[1]
            # entity = entity_predict.split('_')[0]
            if entity in entity_lst:
                index_predict += 1
                entity_predict = predict_tags[index_predict]

                while index_predict < len_predict - 1 and entity_predict != 'Others' and 'B-' not in entity_predict:
                    index_predict += 1
                    entity_predict = predict_tags[index_predict]
                end_index = index_predict
                entity_static[entity]['count_pre'] += 1
                entity_static[entity][entity + str(entity_static[entity]['count_pre']) + '_pre'] = [start_index,
                                                                                                    end_index]
                if 'B-' in entity_predict:
                    index_predict -= 1

        index_predict += 1

    index_gold = 0
    while index_gold < len_gold:
        entity_gold = gold_tags[index_gold]
        if 'B-' in entity_gold:
            start_index = index_gold
            entity = entity_gold.split('-')[1]
            if entity in entity_lst:

                index_gold += 1
                entity_gold = gold_tags[index_gold]

                while index_gold < len_gold - 1 and entity_gold != 'Others' and 'B-' not in entity_gold:
                    index_gold += 1
                    entity_gold = gold_tags[index_gold]
                end_index = index_gold
                entity_static[entity]['count_gold'] += 1
                entity_static[entity][entity + str(entity_static[entity]['count_gold']) + '_gold'] = [start_index,
                                                                                                      end_index]
                if 'B-' in entity_gold:
                    index_gold -= 1

        index_gold += 1

    with open(output_file, 'w', encoding='utf8') as f:
        for entity in entity_lst:
            key_pre = list(entity_static[entity].keys())[5:entity_static[entity]['count_pre'] + 5]
            key_gold = list(entity_static[entity].keys())[
                       entity_static[entity]['count_pre'] + 5:entity_static[entity]['count_pre'] + 5 +
                                                              entity_static[entity]['count_gold']]
            index_pres = [entity_static[entity][key] for key in key_pre]
            index_golds = [entity_static[entity][key] for key in key_gold]
            # count_min = min(entity_static['Location']['count_pre'], entity_static['Location']['count_gold'])

            for index_pre in index_pres:
                if index_pre in index_golds:
                    entity_static[entity]['tp_precise'] += 1
                for index_gold in index_golds:
                    if not (index_pre[1] < index_gold[0] or index_gold[1] < index_pre[0]):
                        #                     if not(index_pre[1] < index_gold[0] or index_gold[1] < index_pre[0] ):
                        entity_static[entity]['tp_blurred'] += 1
                        break

            entity_static[entity]['tp_blurred'] = min(entity_static[entity]['tp_blurred'],
                                                      entity_static[entity]['count_pre'],
                                                      entity_static[entity]['count_gold'])
            # print('entity= ',entity)

            precision_precise = entity_static[entity]['tp_precise'] / entity_static[entity]['count_pre'] if \
            entity_static[entity]['count_pre'] != 0 else 0
            precision_blurred = entity_static[entity]['tp_blurred'] / entity_static[entity]['count_pre'] if \
            entity_static[entity]['count_pre'] != 0 else 0

            recall_blurred = entity_static[entity]['tp_blurred'] / entity_static[entity]['count_gold'] if  entity_static[entity]['count_gold']!=0 else 0
            recall_precise = entity_static[entity]['tp_precise'] / entity_static[entity]['count_gold'] if  entity_static[entity]['count_gold']!=0 else 0

            f1_precise = 0 if precision_precise == 0 and recall_precise == 0 else 2 * precision_precise * recall_precise / (
                    precision_precise + recall_precise)
            f1_blurred = 0 if precision_blurred == 0 and recall_blurred == 0 else 2 * precision_blurred * recall_blurred / (
                    precision_blurred + recall_blurred)

            entity_static[entity]['precision_precise'] = precision_precise
            entity_static[entity]['recall_precise'] = recall_precise
            entity_static[entity]['f1_precise'] = f1_precise

            entity_static[entity]['precision_blurred'] = precision_blurred
            entity_static[entity]['recall_blurred'] = recall_blurred
            entity_static[entity]['f1_blurred'] = f1_blurred

            f.write(entity + '*-*'
                    + 'tp_precise:%.4f' % entity_static[entity]['tp_precise']
                    + '-*-' + 'tp_blurred:%.4f' % entity_static[entity]['tp_blurred']
                    + '-*-' + 'count_pre:%.4f' % entity_static[entity]['count_pre']
                    + '-*-' + 'count_gold:%.4f' % entity_static[entity]['count_gold']
                    + '-*-' + 'precision_blurred:%.4f' % entity_static[entity]['precision_blurred']
                    + '-*-' + 'recall_blurred:%.4f' % entity_static[entity]['recall_blurred']
                    + '-*-' + 'f1_blurred:%.4f' % entity_static[entity]['f1_blurred']
                    + '-*-' + 'precision_precise:%.4f' % entity_static[entity]['precision_precise']
                    + '-*-' + 'recall_precise:%.4f' % entity_static[entity]['recall_precise']
                    + '-*-' + 'f1_precise:%.4f' % entity_static[entity]['f1_precise']
                    + '\n'
                    )

    return entity_static


def getPRF_from_trainlog(filename, doc_type='log'):
    PRF_lst = []
    item_lst = []

    with open(filename, 'r', encoding='utf8')as f:
        if doc_type == 'log':
            data_lines = f.readlines()
            for i in range(len(data_lines)):
                if 'MICRO_AVG' in data_lines[i]:
                    #                 print(i)
                    #                 end=i+17
                    data_res = data_lines[i: i + 16]
                    break

            for data_line in data_res[2:]:
                print(data_line)
                data_line = re.sub(r'[ ]{2,}', '\t', data_line)
                item, res = data_line.split('\t')
                res_splst = res.split('-')
                precision = float(res_splst[4].split(':')[-1].strip())
                recall = float(res_splst[5].split(':')[-1].strip())
                f1 = float(res_splst[-1].split(':')[-1].strip())
                PRF_lst.append([precision, recall, f1])
                item_lst.append(item)
            df = pd.DataFrame(PRF_lst,
                              columns=['precision', 'recall', 'f1'], index=item_lst)
            df.to_excel(os.path.join(os.path.split(filename)[0] , 'res.xlsx'), encoding='utf8')

        elif doc_type == 'txt':
            data_lines = f.readlines()

            for line in data_lines:
                line = re.sub('\n', '', line)
                entity = line.split('*-*')[0]
                print(entity)
                item_lst.append(entity)
                data = line.split('*-*')[1]
                data_splite = data.split('-*-')
                data_line = [float(x.split(':')[1]) for x in data_splite]
                PRF_lst.append(data_line)

            df = pd.DataFrame(
                PRF_lst,
                columns=[
                    'tp_precise',
                    'tp_blurred',
                    'count_pre',
                    'count_gold',
                    'precision_blurred',
                    'recall_blurred',
                    'f1_blurred',
                    'precision_precise',
                    'recall_precise',
                    'f1_precise'
                ],
                index=item_lst)
            name =os.path.split(os.path.split(filename)[0])[1]
            df.to_excel(os.path.join(os.path.split(filename)[0],'res.xlsx'), encoding='utf8')

    PRF_lst = []
    item_lst = []

    with open(filename, 'r', encoding='gbk')as f:
        if doc_type == 'log':
            data_lines = f.readlines()
            for i in range(len(data_lines)):
                if 'MICRO_AVG' in data_lines[i]:
                    #                 print(i)
                    #                 end=i+17
                    data_res = data_lines[i: i + 16]
                    break

            for data_line in data_res[2:]:
                print(data_line)
                data_line = re.sub(r'[ ]{2,}', '\t', data_line)
                item, res = data_line.split('\t')
                res_splst = res.split('-')
                precision = float(res_splst[4].split(':')[-1].strip())
                recall = float(res_splst[5].split(':')[-1].strip())
                f1 = float(res_splst[-1].split(':')[-1].strip())
                PRF_lst.append([precision, recall, f1])
                item_lst.append(item)
            df = pd.DataFrame(PRF_lst,
                              columns=['precision', 'recall', 'f1'], index=item_lst)
            df.to_excel(os.path.split(filename)[0] + '/' + 'res.xlsx', encoding='utf8')

        elif doc_type == 'txt':
            data_lines = f.readlines()

            for line in data_lines:
                line = re.sub('\n', '', line)
                entity = line.split('*-*')[0]
                print(entity)
                item_lst.append(entity)
                data = line.split('*-*')[1]
                data_splite = data.split('-*-')
                data_line = [float(x.split(':')[1]) for x in data_splite]
                PRF_lst.append(data_line)

            df = pd.DataFrame(
                PRF_lst,
                columns=[
                    'tp_precise',
                    'count_pre',
                    'count_gold',
                    'precision_precise',
                    'recall_precise',
                    'f1_precise'
                ],
                index=item_lst)
            name =os.path.split(os.path.split(filename)[0])[1]
            df.to_excel(os.path.join(os.path.split(filename)[0],'res.xlsx'), encoding='utf8')

def get_final_test_score(file_dir):
    test_tsv_dir = os.path.join(file_dir, 'test.tsv')
    test_txt_dir = os.path.join(file_dir, 'evaluation_result.txt')
    get_entity_statics(test_tsv_dir)
    getPRF_from_trainlog(test_txt_dir, doc_type='txt')


if __name__ == "__main__":
    get_final_test_score(r'./output/bert_layers2_head8_dim128_dv64_dk64_lr5e-5_batch80seed1')