
import logging
from pathlib import Path
from typing import  Union
import time
import torch
from torch.optim.lr_scheduler import  LambdaLR
from torch.optim.sgd import SGD
from torch.utils.data.dataset import ConcatDataset
from NLPer.data import Corpus
import NLPer
import NLPer.nn

from NLPer.datasets import DataLoader

from NLPer.training_utils import (
    init_output_file,
    WeightExtractor,
    log_line,
    add_file_handler,
    Result,
    store_embeddings,
)
from NLPer.datasets import convert_sent_to_feature
log = logging.getLogger("NLPer")


def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
    """ Create a schedule with a learning rate that decreases linearly after
    linearly increasing during a warmup period.
    """
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps)))

    return LambdaLR(optimizer, lr_lambda, last_epoch)

class ModelTrainer:
    def __init__(
        self,
        model: NLPer.nn.Model,
        corpus: Corpus,
        optimizer: torch.optim.Optimizer = SGD,
        epoch: int = 0,
        loss: float = 10000.0,
        optimizer_state: dict = None,
        best_score :float=0.0,
        scheduler_state: dict = None,
        use_tensorboard: bool = False,
    ):


        """
        :param model: model for training
        :param corpus: corpus containing training,dev and test data
        :param optimizer: optimizer for updating parameters
        :param epoch: record the current epoch
        :param loss: record the current training loss.
        :param optimizer_state: record optimizer's state
        :param scheduler_state: record scheduler's state
        :param tensorboard: If True, use tensorboard

        """

        self.model: NLPer.nn.Model = model
        self.corpus: Corpus = corpus
        self.optimizer = optimizer
        self.epoch: int = epoch
        self.loss: float = loss
        self.scheduler_state: dict = scheduler_state
        self.optimizer_state: dict = optimizer_state
        self.use_tensorboard: bool = use_tensorboard
        self.best_score = best_score



    def train(
        self,
        base_path: Union[Path, str],
        learning_rate: float = 0.1,
        mini_batch_size: int = 32,
        eval_mini_batch_size: int = None,
        max_epochs: int = 100,
        anneal_factor: float = 0.5,
        patience: int = 5,
        tolerance: int =0.0001,
        train_with_dev: bool = False,
        monitor_train: bool = False,
        monitor_test: bool = False,
        embeddings_storage_mode: str = "cpu",
        pre_training: bool = False,
        checkpoint: bool = False,
        save_final_model: bool = True,
        anneal_with_restarts: bool = False,
        shuffle: bool = True,
        param_selection_mode: bool = False,
        num_workers: int = 4,
        sampler=None,
        **kwargs,
    ) -> dict:
        """
        :param base_path: Main path to which all output during training is logged and models are saved
        :param learning_rate: Initial learning rate
        :param mini_batch_size: Size of mini-batches during training
        :param eval_mini_batch_size: Size of mini-batches during evaluation
        :param max_epochs: Maximum number of epochs to train. Terminates training if this number is surpassed.
        :param anneal_factor: The factor by which the learning rate is annealed
        :param patience: Patience is the number of epochs with no improvement the Trainer waits
         until annealing the learning rate
        :param train_with_dev: If True, training is performed using both train+dev data
        :param embeddings_storage_mode: One of 'none' (all embeddings are deleted and freshly recomputed),
        'cpu' (embeddings are stored on CPU) or 'gpu' (embeddings are stored on GPU)
        :param checkpoint: If True, a full checkpoint is saved at end of each epoch
        :param save_final_model: If True, final model is saved
        :param anneal_with_restarts: If True, the last best model is restored when annealing the learning rate
        :param shuffle: If True, data is shuffled during training
        :param param_selection_mode: If True, testing is performed against dev data. Use this mode when doing
        parameter selection.
        :param num_workers: Number of workers in your data loader.
        :param sampler: You can pass a data sampler here for special sampling of data.
        :param kwargs: Other arguments for the Optimizer
        :return:
        """

        if self.use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter

                writer = SummaryWriter()
            except:
                log_line(log)
                log.warning(
                    "ATTENTION! PyTorch >= 1.1.0 and pillow are required for TensorBoard support!"
                )
                log_line(log)
                self.use_tensorboard = False
                pass


        if eval_mini_batch_size is None:
            eval_mini_batch_size = mini_batch_size

        # cast string to Path
        if type(base_path) is str:
            base_path = Path(base_path)

        if pre_training:
            print('pre_training')
            self.pre_train(base_path,learning_rate,mini_batch_size,max_epochs=20)

        patience_std = 0
        log_handler = add_file_handler(log, base_path / "training.log")

        log_line(log)
        log.info(f'Corpus: "{self.corpus}"')
        log_line(log)
        log.info("Parameters:")
        log.info(f' - learning_rate: "{learning_rate}"')
        log.info(f' - mini_batch_size: "{mini_batch_size}"')
        log.info(f' - patience: "{patience}"')
        log.info(f' - anneal_factor: "{anneal_factor}"')
        log.info(f' - max_epochs: "{max_epochs}"')
        log.info(f' - shuffle: "{shuffle}"')
        log.info(f' - train_with_dev: "{train_with_dev}"')
        log_line(log)
        log.info(f'Model training base path: "{base_path}"')
        log_line(log)
        log.info(f"Device: {NLPer.device}")
        log_line(log)
        log.info(f"Embeddings storage mode: {embeddings_storage_mode}")

        # determine what splits (train, dev, test) to evaluate and log
        log_train = True if monitor_train else False
        log_test = (
            True
            if (not param_selection_mode and self.corpus.test and monitor_test)
            else False
        )
        log_dev = True if not train_with_dev else False

        # prepare loss logging file and set up header
        loss_txt = init_output_file(base_path, "loss.tsv")

        weight_extractor = WeightExtractor(base_path)

        optimizer = self.optimizer(
            self.model.parameters(),
            lr=learning_rate,
        )
        if self.optimizer_state is not None:
            optimizer.load_state_dict(self.optimizer_state)
        print(optimizer)


        # minimize training loss if training with dev data, else maximize dev score
        anneal_mode = "min" if train_with_dev else "max"

        scheduler=get_linear_schedule_with_warmup(optimizer,
                                                  num_warmup_steps=0,
                                                  num_training_steps=len(self.corpus.train)  )

        if self.scheduler_state is not None:
            scheduler.load_state_dict(self.scheduler_state)

        train_data = self.corpus.train


        # if training also uses dev data, include in training set
        if train_with_dev:
            train_data = ConcatDataset([self.corpus.train, self.corpus.dev])

        if sampler is not None:
            sampler = sampler(train_data)
            shuffle = False

        dev_score_history = []
        dev_loss_history = []
        train_loss_history = []

        try:
            previous_learning_rate = learning_rate
            self.best_score = 0
            for epoch in range(0 + self.epoch, max_epochs + self.epoch):
                log_line(log)

                # get new learning rate
                for group in optimizer.param_groups:
                    if group["lr"] > 0 :
                        learning_rate = group["lr"]

                # reload last best model if annealing with restarts is enabled
                if (
                    learning_rate != previous_learning_rate
                    and anneal_with_restarts
                    and (base_path / "best-model.pt").exists()
                ):
                    log.info("resetting to best model")
                    self.model.load(base_path / "best-model.pt")

                previous_learning_rate = learning_rate

                batch_loader = DataLoader(
                    train_data,
                    batch_size=mini_batch_size,
                    shuffle=shuffle,
                    num_workers=num_workers,
                    sampler=sampler,
                )

                self.model.train()

                train_loss: float = 0

                seen_batches = 0
                total_number_of_batches = len(batch_loader)

                modulo = max(1, int(total_number_of_batches / 10))

                # process mini-batches
                batch_time = 0
                for batch_no, batch in enumerate(batch_loader):
                    start_time = time.time()
     
                    if self.fine_tune:
                        input_ids = []
                        attention_masks = []
                        input_type_ids = []
                        for sent in batch:
                            sent_text = ''.join([token.text for token in sent])

                            input_id, input_mask, input_type_id = convert_sent_to_feature(
                                self.tokenizer,
                                sent_text,
                                max_sequence_length=128
                            )
                            input_ids.append(input_id)
                            attention_masks.append(input_mask)
                            input_type_ids.append(input_type_id)
                        batch = (torch.tensor(input_ids),torch.tensor(attention_masks),torch.tensor(input_type_ids),batch)

                    loss = self.model.forward_loss(batch)

                    optimizer.zero_grad()
                    # Backward

                    loss.backward()

                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)

                    # update parameters
                    optimizer.step()
                    seen_batches += 1
                    train_loss += loss.item()

                  
                    if not self.fine_tune:
                        store_embeddings(batch, embeddings_storage_mode)

                    batch_time += time.time() - start_time
                    if batch_no % modulo == 0:
                        log.info(
                            f"epoch {epoch + 1} - iter {batch_no}/{total_number_of_batches} - loss "
                            f"{train_loss / seen_batches:.8f} - samples/sec: {mini_batch_size * modulo / batch_time:.2f}"
                        )
                        batch_time = 0
                        iteration = epoch * total_number_of_batches + batch_no
                        if not param_selection_mode:
                            weight_extractor.extract_weights(
                                self.model.state_dict(), iteration
                            )

                train_loss /= seen_batches

                self.model.eval()

                log_line(log)
                log.info(
                    f"EPOCH {epoch + 1} done: loss {train_loss:.4f} - lr {learning_rate:.2e}"
                )

                if self.use_tensorboard:
                    writer.add_scalar("train_loss", train_loss, epoch + 1)

                # anneal against train loss if training with dev, otherwise anneal against dev score
                current_score = train_loss

                # evaluate on train / dev / test split depending on training settings
                result_line: str = ""

                if log_train:
                    train_eval_result, train_loss = self.model.evaluate(
                        DataLoader(
                            self.corpus.train,
                            batch_size=eval_mini_batch_size,
                            num_workers=num_workers

                        ),
                        out_path=base_path / "train.tsv",
                        embeddings_storage_mode=embeddings_storage_mode,
                    )
                    result_line += f"\t{train_eval_result.log_line}"

                    # depending on memory mode, embeddings are moved to CPU, GPU or deleted
                    if not self.fine_tune:
                        store_embeddings(self.corpus.train, embeddings_storage_mode)

                if log_dev:

                    dev_data =self.corpus.dev

                    dev_eval_result, dev_loss = self.model.evaluate(
                        DataLoader(
                            dev_data,
                            batch_size=eval_mini_batch_size,
                            num_workers=num_workers,
                        ),
                        out_path=base_path / "dev.tsv",
                        embeddings_storage_mode=embeddings_storage_mode,
                    )
                    result_line += f"\t{dev_loss}\t{dev_eval_result.log_line}"
                    log.info(
                        f"DEV : loss {dev_loss} - score {dev_eval_result.main_score}"
                    )
                    # calculate scores using dev data if available
                    # append dev score to score history
                    dev_score_history.append(dev_eval_result.main_score)
                    dev_loss_history.append(dev_loss)

                    current_score = dev_eval_result.main_score
                    # self.best_score = max(current_score,self.best_score)
                    if current_score >self.best_score:
                        self.best_score = current_score
                        patience_std = 0
                        if (
                                not train_with_dev
                                and not param_selection_mode
                   #             and current_score == self.best_score
                        ):
                            self.model.save(base_path / "best-model.pt")
                    elif (self.best_score == current_score)  or (self.best_score - current_score > tolerance) :
                        patience_std += 1
                        log.info('patience_std ={} ,best score ={}  ,curren_score ={}  '.format(patience_std,self.best_score,current_score))

                    if patience_std > patience:
                        break

                    # depending on memory mode, embeddings are moved to CPU, GPU or deleted
                    if not self.fine_tune:
                        store_embeddings(dev_data, embeddings_storage_mode)

                    if self.use_tensorboard:
                        writer.add_scalar("dev_loss", dev_loss, epoch + 1)
                        writer.add_scalar(
                            "dev_score", dev_eval_result.main_score, epoch + 1
                        )

                if log_test:

                    test_data = self.corpus.test
                    test_eval_result, test_loss = self.model.evaluate(
                        DataLoader(
                            test_data,
                            batch_size=eval_mini_batch_size,
                            num_workers=num_workers,
                        ),
                        base_path / "test.tsv",
                        embeddings_storage_mode=embeddings_storage_mode,
                    )
                    result_line += f"\t{test_loss}\t{test_eval_result.log_line}"
                    log.info(
                        f"TEST : loss {test_loss} - score {test_eval_result.main_score}"
                    )

                    # depending on memory mode, embeddings are moved to CPU, GPU or deleted
                    if not self.fine_tune:
                        store_embeddings(test_data, embeddings_storage_mode)

                    if self.use_tensorboard:
                        writer.add_scalar("test_loss", test_loss, epoch + 1)
                        writer.add_scalar(
                            "test_score", test_eval_result.main_score, epoch + 1
                        )

                # determine learning rate annealing through scheduler

                scheduler.step()
                
                train_loss_history.append(train_loss)
              

                # if checkpoint is enable, save model at each epoch
                if checkpoint and not param_selection_mode:
                    self.model.save_checkpoint(
                        base_path / "checkpoint.pt",
                        optimizer.state_dict(),
                        scheduler.state_dict(),
                        epoch + 1,
                        train_loss,
                    )

            # if we do not use dev data for model selection, save final model
            if save_final_model and not param_selection_mode:
                self.model.save(base_path / "final-model.pt")

        except KeyboardInterrupt:
            log_line(log)
            log.info("Exiting from training early.")

            if self.use_tensorboard:
                writer.close()

            if not param_selection_mode:
                log.info("Saving model ...")
                self.model.save(base_path / "final-model.pt")
                log.info("Done.")

        # test best model if test data is present
        if self.corpus.test:
            final_score = self.final_test(base_path, eval_mini_batch_size, num_workers)
        else:
            final_score = 0
            log.info("Test data not provided setting final score to 0")

        log.removeHandler(log_handler)

        if self.use_tensorboard:
            writer.close()

        return {
            "test_score": final_score,
            "dev_score_history": dev_score_history,
            "train_loss_history": train_loss_history,
            "dev_loss_history": dev_loss_history,
        }

    def final_test(
        self, base_path: Path, eval_mini_batch_size: int, num_workers: int = 8
    ):

        log_line(log)
        log.info("Testing using best model ...")

        self.model.eval()

        if (base_path / "best-model.pt").exists():
            self.model = self.model.load(base_path / "best-model.pt")



        test_data = self.corpus.test

        test_results, test_loss = self.model.evaluate(
            DataLoader(
                test_data,
                batch_size=eval_mini_batch_size,
                num_workers=num_workers,
            ),
            out_path=base_path / "test.tsv",
            embeddings_storage_mode="none",
        )

        test_results: Result = test_results
        log.info(test_results.log_line)
        log.info(test_results.detailed_results)
        log_line(log)


        # get and return the final test score of best model
        final_score = test_results.main_score

        return final_score

    @classmethod
    def load_from_checkpoint(
    checkpoint, corpus: Corpus, optimizer: torch.optim.Optimizer = SGD
    ):
        return ModelTrainer(
            checkpoint["model"],
            corpus,
            optimizer,
            epoch=checkpoint["epoch"],
            loss=checkpoint["loss"],
            optimizer_state=checkpoint["optimizer_state_dict"],
            scheduler_state=checkpoint["scheduler_state_dict"],
        )


