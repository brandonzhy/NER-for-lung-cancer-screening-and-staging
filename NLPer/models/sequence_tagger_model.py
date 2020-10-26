import warnings
import logging
from pathlib import Path

import torch.nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import NLPer.nn
import torch
import math
from NLPer.data import Dictionary, Sentence, Token, Label
from NLPer.datasets import DataLoader
from NLPer.embeddings import TokenEmbeddings
from NLPer.file_utils import cached_path
from collections import  Counter
from typing import List, Tuple, Union
from NLPer.training_utils import Metric, Result, store_embeddings
from sklearn.feature_extraction.text import  TfidfVectorizer
from tqdm import tqdm
from tabulate import tabulate
import numpy as np
log = logging.getLogger("NLPer")

START_TAG: str = "<START>"
STOP_TAG: str = "<STOP>"


def to_scalar(var):
    return var.view(-1).detach().tolist()[0]


def argmax(vec):
    _, idx = torch.max(vec, 1)
    return to_scalar(idx)


def log_sum_exp(vec):
    max_score = vec[0, argmax(vec)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score + torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))


def argmax_batch(vecs):
    _, idx = torch.max(vecs, 1)
    return idx


def log_sum_exp_batch(vecs):
    maxi = torch.max(vecs, 1)[0]
    maxi_bc = maxi[:, None].repeat(1, vecs.shape[1])
    recti_ = torch.log(torch.sum(torch.exp(vecs - maxi_bc), 1))
    return maxi + recti_


def pad_tensors(tensor_list):
    ml = max([x.shape[0] for x in tensor_list])
    shape = [len(tensor_list), ml] + list(tensor_list[0].shape[1:])
    template = torch.zeros(*shape, dtype=torch.long, device=NLPer.device)
    lens_ = [x.shape[0] for x in tensor_list]
    for i, tensor in enumerate(tensor_list):
        template[i, : lens_[i]] = tensor

    return template, lens_

def get_attn_pad_mask(seq_q, seq_k):
    batch_size, len_q = seq_q.shape[0:2]
    batch_size, len_k = seq_k.shape[0:2]
    # eq(zero) is PAD token
    pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)  # batch_size x 1 x len_k(=len_q), one is masking
    return pad_attn_mask.expand(batch_size, len_q, len_k)  # batch_size x len_q x len_k

def get_tfodf_vocab(sentences):
    sentence_texts = [" ".join([token.text for token in sentence.tokens]) for sentence in sentences]
    tf_idf_vocab = {}
    index = 0
    for line in sentence_texts:
        for word in line.split():
            if word not in tf_idf_vocab:
                tf_idf_vocab[word] = index
                index += 1
    return tf_idf_vocab,sentence_texts


class PositionalEncoding(torch.nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = torch.nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0., max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0., d_model, 2) *
                             -(math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # batch size is on the second dim
        x = x +self.pe[:x.size(0),:,:x.size(2)].clone().detach()
        return self.dropout(x)
class ScaledDotProductAttention(torch.nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V,d_k):
        scores = torch.matmul(Q, K.transpose(-1, -2).contiguous()) /torch.sqrt(torch.tensor(d_k,dtype=torch.float32)) # scores : [batch_size x n_heads x len_q(=len_k) x len_k(=len_q)]
        attn = torch.nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)
        return context, attn

class MultiHeadAttention(torch.nn.Module):
    def __init__(self,embedding_dim, d_k,d_v ,n_heads):
        super(MultiHeadAttention, self).__init__()
        self.n_heads=n_heads[0] if type(n_heads)==tuple  else n_heads
        self.d_k=d_k[0] if type(d_k)==tuple  else d_k
        self.d_v=d_v[0] if type(d_v)==tuple  else d_v
        self.embedding_dim= embedding_dim[0] if type(embedding_dim)==tuple  else embedding_dim
        self.W_Q = torch.nn.Linear(embedding_dim, d_k * self.n_heads)
        self.W_K = torch.nn.Linear(embedding_dim, d_k * self.n_heads)
        self.W_V = torch.nn.Linear(embedding_dim, d_v * self.n_heads)

    def forward(self, Q, K, V):

        residual, batch_size = Q, Q.size(1)
        q_s = self.W_Q(Q).view( -1,batch_size, self.n_heads, self.d_k).transpose(1,2).contiguous()  # q_s: [batch_size x n_heads x len_q x d_k]
        k_s = self.W_K(K).view( -1,batch_size, self.n_heads, self.d_k).transpose(1,2).contiguous() # k_s: [batch_size x n_heads x len_k x d_k]
        v_s = self.W_V(V).view( -1,batch_size, self.n_heads, self.d_v).transpose(1,2).contiguous()  # v_s: [batch_size x n_heads x len_k x d_v]

        context, attn = ScaledDotProductAttention()(q_s, k_s, v_s,self.d_k)
        context = context.transpose(1, 2).contiguous().view( -1,batch_size, self.n_heads * self.d_v) # context: [batch_size x len_q x n_heads * d_v]
        output = torch.nn.Linear(self.n_heads * self.d_v, self.embedding_dim)(context)
        return torch.nn.LayerNorm(self.embedding_dim)(output + residual), attn # output: [batch_size x len_q x embedding_dim]



class SequenceTagger(NLPer.nn.Model):
    def __init__(
        self,
        hidden_size: int,
        embeddings: TokenEmbeddings,
        tag_dictionary: Dictionary,
        tag_type: str,
        use_crf: bool = True,
        use_rnn: bool = True,
        pre_train:bool=False,
        n_heads: int = 0 ,
        transformer_layers:int =1,
        rnn_layers: int = 1,
        dropout: float = 0.0,
        train_initial_hidden_state: bool = False,
        add_position:bool=False,
        pickle_module: str = "pickle",
    ):

        super(SequenceTagger, self).__init__()

        self.use_rnn = use_rnn
        self.hidden_size = hidden_size
        self.use_crf: bool = use_crf
        self.rnn_layers: int = rnn_layers
        self.use_pre_train:bool=False
        self.trained_epochs: int = 0

        self.n_heads=n_heads[0] if type(n_heads)==tuple else n_heads
        self.transformer_layers =transformer_layers
        self.embeddings=embeddings[0]  if type(embeddings)==tuple else embeddings


        # set the dictionaries
        self.tag_dictionary: Dictionary = tag_dictionary
        self.tag_type: str = tag_type
        self.tagset_size: int = len(tag_dictionary)

        # initialize the network architecture
        self.nlayers: int = rnn_layers
        self.hidden_word = None

        # dropouts
        self.use_dropout: float = dropout
        self.add_position=add_position
        self.pickle_module = pickle_module

        if dropout > 0.0:
            self.dropout = torch.nn.Dropout(dropout)


        # rnn_input_dim: int = self.embeddings.embedding_length
        rnn_input_dim: int = self.embeddings.embedding_length
        self.relearn_embeddings: bool = True

        if self.relearn_embeddings:
               self.embedding2nn = torch.nn.Linear(rnn_input_dim, rnn_input_dim)

        self.train_initial_hidden_state = train_initial_hidden_state
        self.bidirectional = True
        self.rnn_type = "LSTM"

        if self.n_heads :
            if self.use_rnn:
                tramsformer_dim = 2*self.hidden_size
            else:
                tramsformer_dim = self.embeddings.embedding_length
            encoder_layer = torch.nn.TransformerEncoderLayer(d_model=tramsformer_dim,
                                                             nhead=self.transformer_layers, dim_feedforward=self.hidden_size,
                                                             dropout=0.1, activation='relu')
            self.self_attn = torch.nn.TransformerEncoder(encoder_layer=encoder_layer, num_layers=self.layers,
                                                         norm=None)
            self.linear = torch.nn.Linear(self.embeddings.embedding_length, len(tag_dictionary))

        if self.add_position:
            self.pos=PositionalEncoding(self.embeddings.embedding_length,self.dropout,512)
        #  for pre_training,the output of linear  layer must be embedding_length
        if pre_train:
            if self.use_rnn:
                self.pretrain_linear=torch.nn.Linear(
                    hidden_size * 2,self.embeddings.embedding_length
                )
            else:
                self.pretrain_linear = torch.nn.Linear(
                    self.embeddings.embedding_length, self.embeddings.embedding_length
                )
        # bidirectional LSTM on top of embedding layer

        if self.use_rnn:
            num_directions = 2 if self.bidirectional else 1

            if self.rnn_type in ["LSTM", "GRU"]:

                self.rnn = getattr(torch.nn, self.rnn_type)(
                    rnn_input_dim,
                    hidden_size,
                    num_layers=self.nlayers,
                    dropout=0.0 if self.nlayers == 1 else 0.5,
                    bidirectional=True,
                )
                # Create initial hidden state and initialize it
                if self.train_initial_hidden_state:
                    self.hs_initializer = torch.nn.init.xavier_normal_

                    self.lstm_init_h = Parameter(
                        torch.randn(self.nlayers * num_directions, self.hidden_size),
                        requires_grad=True,
                    )

                    self.lstm_init_c = Parameter(
                        torch.randn(self.nlayers * num_directions, self.hidden_size),
                        requires_grad=True,
                    )


            # final linear map to tag space
            self.linear = torch.nn.Linear(
                hidden_size * num_directions, len(tag_dictionary)
            )

        else:
            self.linear = torch.nn.Linear(
                self.embeddings.embedding_length, len(tag_dictionary)
            )

        if self.use_crf:
            self.transitions = torch.nn.Parameter(
                torch.randn(self.tagset_size, self.tagset_size)
            )
            self.transitions.detach()[
                self.tag_dictionary.get_idx_for_item(START_TAG), :
            ] = -10000
            self.transitions.detach()[
                :, self.tag_dictionary.get_idx_for_item(STOP_TAG)
            ] = -10000

        self.to(NLPer.device)

    def _get_state_dict(self):
        model_state = {
            "state_dict": self.state_dict(),
            "embeddings": self.embeddings,
            "hidden_size": self.hidden_size,
            "train_initial_hidden_state": self.train_initial_hidden_state,
            "tag_dictionary": self.tag_dictionary,
            "tag_type": self.tag_type,
            "use_crf": self.use_crf,
            "use_rnn": self.use_rnn,
            "n_heads": self.n_heads,
            "transformer_layers":self.transformer_layers,
            "rnn_layers": self.rnn_layers
        }
        return model_state

    def get_sentences_embeddings(self,sentences):

        self.embeddings.embed(sentences)

        lengths: List[int] = [len(sentence.tokens) for sentence in sentences]
        longest_token_sequence_in_batch: int = max(lengths)

        # initialize zero-padded word embeddings tensor
        sentence_tensor = torch.zeros(
            [
                len(sentences),
                longest_token_sequence_in_batch,
                self.embeddings.embedding_length,
            ],
            dtype=torch.float,
            device=NLPer.device,
        )

        for s_id, sentence in enumerate(sentences):
            # fill values with word embeddings
            sentence_tensor[s_id][: len(sentence)] = torch.cat(
                [token.get_embedding().unsqueeze(0) for token in sentence], 0
            )

        # TODO: this can only be removed once the implementations of word_dropout and locked_dropout have a batch_first mode
        sentence_tensor = sentence_tensor.transpose_(0, 1).contiguous()
        return sentence_tensor,lengths

    def _init_model_with_state_dict(state):

        use_dropout = 0.0 if not "use_dropout" in state.keys() else state["use_dropout"]
       
        train_initial_hidden_state = (
            False
            if not "train_initial_hidden_state" in state.keys()
            else state["train_initial_hidden_state"]
        )

        model = SequenceTagger(
            hidden_size=state["hidden_size"],
            embeddings=state["embeddings"],
            tag_dictionary=state["tag_dictionary"],
            tag_type=state["tag_type"],
            use_crf=state["use_crf"],
            use_rnn=state["use_rnn"],
            rnn_layers=state["rnn_layers"],
            n_heads=state["n_heads"],
            transformer_layers=state['transformer_layers'],
            dropout=use_dropout,
            train_initial_hidden_state=train_initial_hidden_state,
        )
        model.load_state_dict(state["state_dict"],strict=False)
        return model

    def evaluate(
        self,
        data_loader: DataLoader,
        out_path: Path = None,
        embeddings_storage_mode: str = "cpu",
        fine_tune :bool = False,
        tokenizer =None
    ) -> (Result, float):
        excep_list=['Hills_B','Hills_I','Esophagus_B','Esophagus_I','Others','<unk>','ChestWall_B','ChestWall_I']
        with torch.no_grad():
            eval_loss = 0

            batch_no: int = 0

            metric = Metric("Evaluation")

            lines: List[str] = []
            for batch in data_loader:
                batch_no += 1

                with torch.no_grad():
                    features = self.forward(batch)
                    loss = self._calculate_loss(features, batch)
                    tags, _ = self._obtain_labels(features, batch)

                eval_loss += loss

                for (sentence, sent_tags) in zip(batch, tags):
                    for (token, tag) in zip(sentence.tokens, sent_tags):
                        token: Token = token
                        token.add_tag_label("predicted", tag)

                        # append both to file for evaluation
                        eval_line = "{} {} {} {}\n".format(
                            token.text,
                            token.get_tag(self.tag_type).value,
                            tag.value,
                            tag.score,
                        )
                        lines.append(eval_line)
                    lines.append("\n")
                index =0
                count_rec = 0
                len_data = len(lines)
                while index < len_data and lines[index]!='\n':
                    line_split = lines[index].split(' ')
                    tag_predict = line_split[-2]
                    if 0 < index < len_data-1 and tag_predict =='Others' and lines[index - 1]!='\n'   and '_I' in lines[index - 1].split(' ')[-2]   and lines[index + 1]!='\n'  and '_I' in lines[index + 1].split(' ')[-2]:
                        if lines[index + 1].split(' ')[-2] == lines[index - 1].split(' ')[-2]:
                            lines[index]= "{} {} {} {}\n".format(
                                line_split[0],
                                line_split[1],
                                lines[index + 1].split(' ')[-2],
                                line_split[-1]
                        )

                        else:
                            tag = '_B'+ str(lines[index + 1].split(' ')[-2]).split('-')[-1]
                            lines[index] = "{} {} {} {}\n".format(
                                line_split[0],
                                line_split[1],
                                tag,
                                line_split[-1]
                            )
                        count_rec += 1
                    elif 0 < index < len_data-1 and '_I' in tag_predict and lines[index - 1]!='\n' and lines[index - 1].split(' ')[-2] == 'Others':
                        tag = '_B' + str(tag_predict).split('-')[-1]
                        lines[index] = "{} {} {} {}\n".format(
                            line_split[0],
                            line_split[1],
                            tag,
                            line_split[-1]
                        )
                        count_rec += 1
                    elif 0 < index < len_data-1 and '_B' in tag_predict and  lines[index + 1]!='\n'  and  '_B' in lines[index + 1].split(' ')[-2]:
                        tag = 'Others'
                        lines[index] = "{} {} {} {}\n".format(
                            line_split[0],
                            line_split[1],
                            tag,
                            line_split[-1]
                        )
                        count_rec += 1
                    elif 0 < index < len_data-1 and tag_predict =='Others' and  lines[index + 1]!='\n'  and   '_I' in lines[index + 1].split(' ')[-2] :

                        tag = '_B'+ lines[index + 1].split(' ')[-2][-1]
                        lines[index] = "{} {} {} {}\n".format(
                            line_split[0],
                            line_split[1],
                            tag,
                            line_split[-1]
                        )
                        count_rec +=1



                    index+=1
                for sentence in batch:
                    # make list of gold tags

                    gold_tags = [
                        tag.tag for tag in sentence.get_spans(self.tag_type)
                    ]
                    # make list of predicted tags
                    predicted_tags = [
                        tag.tag for tag in sentence.get_spans("predicted")
                    ]
                    predicted_tags_rec = []
                    index = 0

                    # assert  len(gold_tags)==len(predicted_tags)
                    # check for true positives, false positives and false negatives

                    len_gold = len(gold_tags)
                    len_predict = len(predicted_tags)
                    if len_gold >= len_predict:
                        length = len_predict
                    else:
                        length = len_gold
                    if len(gold_tags)==len(predicted_tags):

                        for index_predict in range(length):
                            tag = predicted_tags[index_predict]
                            if tag not in excep_list:
                                if tag == gold_tags[index_predict]:
                                    metric.add_tp(tag)
                                else:
                                    metric.add_fp(tag)

                        for index_gold in range(length):
                            gold_tag = gold_tags[index_gold]
                            if gold_tag not in excep_list:
                                if gold_tag != predicted_tags[index_gold]:
                                    metric.add_fn(gold_tag)


                store_embeddings(batch, embeddings_storage_mode)

            eval_loss /= batch_no

            if out_path is not None:
                with open(out_path, "w", encoding="utf-8") as outfile:
                    outfile.write("".join(lines))

            detailed_result = (
                f"\nMICRO_AVG: acc {metric.micro_avg_accuracy()} - f1-score {metric.micro_avg_f_score()}"
                f"\nMACRO_AVG: acc {metric.macro_avg_accuracy()} - f1-score {metric.macro_avg_f_score()}"
            )
            for class_name in metric.get_classes():
                detailed_result += (
                    f"\n{class_name:<10} tp: {metric.get_tp(class_name)} - fp: {metric.get_fp(class_name)} - "
                    f"fn: {metric.get_fn(class_name)} - tn: {metric.get_tn(class_name)} - precision: "
                    f"{metric.precision(class_name):.4f} - recall: {metric.recall(class_name):.4f} - "
                    f"accuracy: {metric.accuracy(class_name):.4f} - f1-score: "
                    f"{metric.f_score(class_name):.4f}"
                )

            result = Result(
                main_score=metric.macro_avg_f_score(),
                log_line=f"{metric.precision()}\t{metric.recall()}\t{metric.macro_avg_f_score()}",
                log_header="PRECISION\tRECALL\tF1",
                detailed_results=detailed_result,
            )

            return result, eval_loss



    def forward(self, sentences: List[Sentence]):

        self.zero_grad()
        sentence_tensor, lengths = self.get_sentences_embeddings(sentences)
      
        if self.add_position:
            sentence_tensor = self.pos(sentence_tensor)

        # --------------------------------------------------------------------
        # FF PART
        # --------------------------------------------------------------------
        

        if self.use_dropout > 0.0:
            sentence_tensor = self.dropout(sentence_tensor)

        if self.relearn_embeddings:
            sentence_tensor = self.embedding2nn(sentence_tensor)

        if self.use_rnn:

            packed = torch.nn.utils.rnn.pack_padded_sequence(
                sentence_tensor, lengths, enforce_sorted=False
            )

            # if initial hidden state is trainable, use this state
            if self.train_initial_hidden_state:
                initial_hidden_state = [
                    self.lstm_init_h.unsqueeze(1).repeat(1, len(sentences), 1),
                    self.lstm_init_c.unsqueeze(1).repeat(1, len(sentences), 1),
                ]
                rnn_output, hidden = self.rnn(packed, initial_hidden_state)
            else:
                rnn_output, hidden = self.rnn(packed)

            sentence_tensor, output_lengths = torch.nn.utils.rnn.pad_packed_sequence(
                rnn_output, batch_first=True
            )
            sentence_tensor = torch.nn.LayerNorm(sentence_tensor.shape).to(NLPer.device)(sentence_tensor)
            if self.use_dropout > 0.0:
                sentence_tensor = self.dropout(sentence_tensor)


            if self.n_heads > 0 and not self.use_pre_train:

                sentence_tensor, _ = self.self_attn(sentence_tensor, sentence_tensor, sentence_tensor)
        else:
            if self.n_heads > 0 and not self.use_pre_train:

                sentence_tensor,_ = self.self_attn(sentence_tensor,sentence_tensor,sentence_tensor)
            # transpose to batch_first mode
            sentence_tensor = sentence_tensor.transpose_(0, 1).contiguous()

        if not self.use_pre_train:
            features = self.linear(sentence_tensor)
        else:
            features = self.pretrain_linear(sentence_tensor)
        return features


    def forward_loss(
        self, data_points: Union[List[Sentence], Sentence], sort=True
    ) -> torch.tensor:
        features = self.forward(data_points)
        return self._calculate_loss(features, data_points)

    def predict(
        self,
        sentences: Union[List[Sentence], Sentence],
        mini_batch_size=32,
        embedding_storage_mode="none",
        verbose=False,
    ) -> List[Sentence]:
        with torch.no_grad():
            if isinstance(sentences, Sentence):
                sentences = [sentences]

            filtered_sentences = self._filter_empty_sentences(sentences)

           
            store_embeddings(filtered_sentences, "none")

           
            filtered_sentences.sort(key=lambda x: len(x), reverse=True)

          
            batches = [
                filtered_sentences[x : x + mini_batch_size]
                for x in range(0, len(filtered_sentences), mini_batch_size)
            ]

           
            if verbose:
                batches = tqdm(batches)

            for i, batch in enumerate(batches):

                if verbose:
                    batches.set_description(f"Inferencing on batch {i}")

                with torch.no_grad():
                    feature = self.forward(batch)
                    tags, all_tags = self._obtain_labels(feature, batch)

                for (sentence, sent_tags, sent_all_tags) in zip(batch, tags, all_tags):
                    for (token, tag, token_all_tags) in zip(
                        sentence.tokens, sent_tags, sent_all_tags
                    ):
                        token.add_tag_label(self.tag_type, tag)
                        token.add_tags_proba_dist(self.tag_type, token_all_tags)

             
                store_embeddings(batch, storage_mode=embedding_storage_mode)

            return sentences
    def get_all_tokens(self,sentences) -> List[str]:
        tokens = list(map((lambda s: s.tokens), sentences))
        tokens = [token for sublist in tokens for token in sublist]
        return list(map((lambda t: t.text), tokens))
    def get_tokens_frequence(self,sentences):
        tokens_and_frequencies = Counter(self.get_all_tokens(sentences))
        return tokens_and_frequencies
    def pre_train(self, sentences:List[Sentence]):
        criterian=torch.nn.MSELoss(reduction='meean')
        self.zero_grad()
        sentence_tensor, lengths = self.get_sentences_embeddings(sentences)
        self.use_crf=False
        output=torch.tanh(self.forward(sentences))
        tf_idf_dic,sentence_text=get_tfodf_vocab(sentences)

        trainsformer=TfidfVectorizer(sentence_text,vocabulary=tf_idf_dic,token_pattern=r"(?u)\b\w+\b")
        tf_idf_vec=trainsformer.fit_transform(sentence_text).toarray()
        weighted_embs=[]
        embedding_dim=sentence_tensor.shape[2]
        for doc in range(sentence_tensor.shape[1]):
            embed=np.zeros([sentence_tensor.shape[0],embedding_dim])
            for word in range(sentence_tensor.shape[0]):
                token=sentences[doc].get_token(word)
                if token:
                    tf_inx=tf_idf_dic[token.text]
                    embed[word]=sentence_tensor[word,doc].numpy()*np.array(tf_idf_vec[doc,tf_inx])

            weighted_embs.append(embed)
        weighted_embs=np.array(weighted_embs)
        weighted_embs = np.clip(weighted_embs,-1,1)
        weighted_embs=torch.from_numpy(weighted_embs)


        loss=self._calculate_pre_train_loss(output,weighted_embs,sentences)
    
        return loss
      


    def _score_sentence(self, feats, tags, lens_):
        start = torch.tensor(
            [self.tag_dictionary.get_idx_for_item(START_TAG)], device=NLPer.device
        )
        start = start[None, :].repeat(tags.shape[0], 1)

        stop = torch.tensor(
            [self.tag_dictionary.get_idx_for_item(STOP_TAG)], device=NLPer.device
        )
        stop = stop[None, :].repeat(tags.shape[0], 1)

        pad_start_tags = torch.cat([start, tags], 1)
        pad_stop_tags = torch.cat([tags, stop], 1)

        for i in range(len(lens_)):
            pad_stop_tags[i, lens_[i] :] = self.tag_dictionary.get_idx_for_item(
                STOP_TAG
            )

        score = torch.FloatTensor(feats.shape[0]).to(NLPer.device)

        for i in range(feats.shape[0]):
            r = torch.LongTensor(range(lens_[i])).to(NLPer.device)

            score[i] = torch.sum(
                self.transitions[
                    pad_stop_tags[i, : lens_[i] + 1], pad_start_tags[i, : lens_[i] + 1]
                ]
            ) + torch.sum(feats[i, r, tags[i, : lens_[i]]])

        return score

    def _calculate_loss(
        self, features: torch.tensor, sentences: List[Sentence]
    ) -> float:

        lengths: List[int] = [len(sentence.tokens) for sentence in sentences]

        tag_list: List = []
        for s_id, sentence in enumerate(sentences):
            # get the tags in this sentence
            tag_idx: List[int] = [
                self.tag_dictionary.get_idx_for_item(token.get_tag(self.tag_type).value)
                for token in sentence
            ]
            # add tags as tensor
            tag = torch.tensor(tag_idx, device=NLPer.device)
            tag_list.append(tag)

        if self.use_crf:
            # pad tags if using batch-CRF decoder
            tags, _ = pad_tensors(tag_list)

            forward_score = self._forward_alg(features, lengths)
            gold_score = self._score_sentence(features, tags, lengths)

            score = forward_score - gold_score

            return score.mean()

        else:
            score = 0
            for sentence_feats, sentence_tags, sentence_length in zip(
                features, tag_list, lengths
            ):
                sentence_feats = sentence_feats[:sentence_length]

                score += torch.nn.functional.cross_entropy(
                    sentence_feats, sentence_tags
                )
            score /= len(features)
            return score
    def _calculate_pre_train_loss(
        self, features: torch.tensor, targets:torch.tensor,sentences
    ) -> float:

        lengths: List[int] = [len(sentence.tokens) for sentence in sentences]

        if self.use_crf:
            # pad tags if using batch-CRF decoder
            tags, _ = pad_tensors(targets)

            forward_score = self._forward_alg(features, lengths)
            gold_score = self._score_sentence(features, tags, lengths)

            score = forward_score - gold_score

            return score.mean()

        else:
            score = 0
            criterian=torch.nn.MSELoss(reduction='sum')
            for sentence_feats, sentence_tags, sentence_length in zip(features, targets, lengths ):
                sentence_feats = sentence_feats[:sentence_length]
                sentence_tags=sentence_tags[:sentence_length].float()
                # print('sentence_feats.shape',sentence_feats.shape,'sentence_tags.shape',sentence_tags.shape)
                score += criterian(sentence_feats, sentence_tags)
            score /= len(features)
            return score

    def _obtain_labels(
        self, feature, sentences
    ) -> (List[List[Label]], List[List[List[Label]]]):
    
        lengths: List[int] = [len(sentence.tokens) for sentence in sentences]

        tags = []
        all_tags = []
        for feats, length in zip(feature, lengths):
            if self.use_crf:
                confidences, tag_seq, scores = self._viterbi_decode(feats[:length])
            else:
                tag_seq = []
                confidences = []
                scores = []
                for backscore in feats[:length]:
                    softmax = F.softmax(backscore, dim=0)
                    _, idx = torch.max(backscore, 0)
                    prediction = idx.item()
                    tag_seq.append(prediction)
                    confidences.append(softmax[prediction].item())
                    scores.append(softmax.tolist())

            tags.append(
                [
                    Label(self.tag_dictionary.get_item_for_index(tag), conf)
                    for conf, tag in zip(confidences, tag_seq)
                ]
            )

            all_tags.append(
                [
                    [
                        Label(self.tag_dictionary.get_item_for_index(score_id), score)
                        for score_id, score in enumerate(score_dist)
                    ]
                    for score_dist in scores
                ]
            )

        return tags, all_tags

    def _viterbi_decode(self, feats):
        backpointers = []
        backscores = []
        scores = []

        init_vvars = (
            torch.FloatTensor(1, self.tagset_size).to(NLPer.device).fill_(-10000.0)
        )
        init_vvars[0][self.tag_dictionary.get_idx_for_item(START_TAG)] = 0
        forward_var = init_vvars

        for feat in feats:
            next_tag_var = (
                forward_var.view(1, -1).expand(self.tagset_size, self.tagset_size)
                + self.transitions
            )
            _, bptrs_t = torch.max(next_tag_var, dim=1)
            viterbivars_t = next_tag_var[range(len(bptrs_t)), bptrs_t]
            forward_var = viterbivars_t + feat
            backscores.append(forward_var)
            backpointers.append(bptrs_t)

        terminal_var = (
            forward_var
            + self.transitions[self.tag_dictionary.get_idx_for_item(STOP_TAG)]
        )
        terminal_var.detach()[self.tag_dictionary.get_idx_for_item(STOP_TAG)] = -10000.0
        terminal_var.detach()[
            self.tag_dictionary.get_idx_for_item(START_TAG)
        ] = -10000.0
        best_tag_id = argmax(terminal_var.unsqueeze(0))

        best_path = [best_tag_id]

        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)

        best_scores = []
        for backscore in backscores:
            softmax = F.softmax(backscore, dim=0)
            _, idx = torch.max(backscore, 0)
            prediction = idx.item()
            best_scores.append(softmax[prediction].item())
            scores.append([elem.item() for elem in softmax.flatten()])

        start = best_path.pop()
        assert start == self.tag_dictionary.get_idx_for_item(START_TAG)
        best_path.reverse()

        for index, (tag_id, tag_scores) in enumerate(zip(best_path, scores)):
            if type(tag_id) != int and tag_id.item() != np.argmax(tag_scores):
                swap_index_score = np.argmax(tag_scores)
                scores[index][tag_id.item()], scores[index][swap_index_score] = (
                    scores[index][swap_index_score],
                    scores[index][tag_id.item()],
                )
            elif type(tag_id) == int and tag_id != np.argmax(tag_scores):
                swap_index_score = np.argmax(tag_scores)
                scores[index][tag_id], scores[index][swap_index_score] = (
                    scores[index][swap_index_score],
                    scores[index][tag_id],
                )

        return best_scores, best_path, scores

    def _forward_alg(self, feats, lens_):

        init_alphas = torch.FloatTensor(self.tagset_size).fill_(-10000.0)
        init_alphas[self.tag_dictionary.get_idx_for_item(START_TAG)] = 0.0
        forward_var = torch.zeros(
            feats.shape[0],
            feats.shape[1] + 1,
            feats.shape[2],
            dtype=torch.float,
            device=NLPer.device,
        )

        forward_var[:, 0, :] = init_alphas[None, :].repeat(feats.shape[0], 1)

        transitions = self.transitions.view(
            1, self.transitions.shape[0], self.transitions.shape[1]
        ).repeat(feats.shape[0], 1, 1)

        for i in range(feats.shape[1]):
            emit_score = feats[:, i, :]

            tag_var = (
                emit_score[:, :, None].repeat(1, 1, transitions.shape[2])
                + transitions
                + forward_var[:, i, :][:, :, None]
                .repeat(1, 1, transitions.shape[2])
                .transpose(2, 1).contiguous()
            )

            max_tag_var, _ = torch.max(tag_var, dim=2)

            tag_var = tag_var - max_tag_var[:, :, None].repeat(
                1, 1, transitions.shape[2]
            )

            agg_ = torch.log(torch.sum(torch.exp(tag_var), dim=2))

            cloned = forward_var.clone()
            cloned[:, i + 1, :] = max_tag_var + agg_

            forward_var = cloned

        forward_var = forward_var[range(forward_var.shape[0]), lens_, :]

        # print( 'forward_var.shape = ,forward_var[0] = ',forward_var.shape,forward_var[0])

        terminal_var = forward_var + self.transitions[
            self.tag_dictionary.get_idx_for_item(STOP_TAG)
        ][None, :].repeat(forward_var.shape[0], 1)

        alpha = log_sum_exp_batch(terminal_var)

        return alpha

    @staticmethod
    def _filter_empty_sentences(sentences: List[Sentence]) -> List[Sentence]:
        filtered_sentences = [sentence for sentence in sentences if sentence.tokens]
        if len(sentences) != len(filtered_sentences):
            log.warning(
                "Ignore {} sentence(s) with no tokens.".format(
                    len(sentences) - len(filtered_sentences)
                )
            )
        return filtered_sentences

    def get_transition_matrix(self):
        data = []
        for to_idx, row in enumerate(self.transitions):
            for from_idx, column in enumerate(row):
                row = [
                    self.tag_dictionary.get_item_for_index(from_idx),
                    self.tag_dictionary.get_item_for_index(to_idx),
                    column.item(),
                ]
                data.append(row)
            data.append(["----"])
        print(tabulate(data, headers=["FROM", "TO", "SCORE"]))
