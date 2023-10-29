import os
import datetime
import torch

from typing import Union, Callable
from pathlib import Path
from operator import itemgetter
from trainer.tensorboard_visualizer import TensorBoardVisualizer
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau
from .utils import AverageMeter

# Generic class for training loop.
# Pass in the model, optimizer, learning rate scheduler, and the number of epochs.

class Trainer:

    def __init__(
        self,
        model: torch.nn.Module, # model (nn.Module): torch model which will be train.
        loader_train: torch.utils.data.DataLoader, # loader (torch.utils.DataLoader): dataset loader.
        loader_test: torch.utils.data.DataLoader, # loader (torch.utils.DataLoader): dataset loader.
        loss_fn: Callable, # loss_fn (callable): loss function.
        metric_fn: Callable, # metric_fn (callable): evaluation metric function.
        optimizer: torch.optim.Optimizer, # optimizer (torch.optim.Optimizer): Optimizer.
        lr_scheduler: Callable, # lr_scheduler (torch.optim.LrScheduler): Learning Rate scheduler.
        device: Union[torch.device, str] = "cuda", # device (str): Specifies device at which samples will be uploaded.
        model_saving_frequency: int = 1, # model_saving_frequency (int): frequency of model state savings per epochs.
        save_dir: Union[str, Path] = "Project/TrainingModel/checkpoints", # save_dir (str): directory to save model states.
        model_name_prefix: str = "model", # model_name_prefix (str): prefix which will be add to the model name.
        data_getter: Callable = itemgetter(0), # data_getter (Callable): function object to extract input data from the sample prepared by dataloader.
        target_getter: Callable = itemgetter(1), # target_getter (Callable): function object to extract target data from the sample prepared by dataloader.
        stage_progress: bool = True, # stage_progress (bool): if True then progress bar will be show.
        visualizer: Union[TensorBoardVisualizer, None] = None, # visualizer (Visualizer): shows metrics values (various backends are possible).
        get_key_metric: Callable = itemgetter("top1"), # get_key_metric (Callable): function object to extract key metric from the metric dictionary.
    ):
        self.model = model
        self.loader_train = loader_train
        self.loader_test = loader_test
        self.loss_fn = loss_fn
        self.metric_fn = metric_fn
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.device = device
        self.model_saving_frequency = model_saving_frequency
        self.save_dir = save_dir
        self.model_name_prefix = model_name_prefix
        self.stage_progress = stage_progress
        self.data_getter = data_getter
        self.target_getter = target_getter
        self.hooks = {}
        self.visualizer = visualizer
        self.get_key_metric = get_key_metric
        self.metrics = {"epoch": [], "train_loss": [], "train_accuracy":[], "test_loss": [], "test_accuracy": []}

    def fit(self, epochs):

        best_loss = float("inf")

        for epoch in range(1, epochs+1):

            output_train = self.train_loop(
                model=self.model,
                loader=self.loader_train,
                loss_fn=self.loss_fn,
                metric_fn=self.metric_fn,
                optimizer=self.optimizer,
                device=self.device,
                prefix="[{}/{}]".format(epoch, epochs),
                stage_progress=self.stage_progress,
                data_getter=self.data_getter,
                target_getter=self.target_getter,
                get_key_metric=self.get_key_metric
            )

            output_test = self.test_loop(
                model=self.model,
                loader=self.loader_test,
                loss_fn=self.loss_fn,
                metric_fn=self.metric_fn,
                device=self.device,
                prefix="[{}/{}]".format(epoch, epochs),
                stage_progress=self.stage_progress,
                data_getter=self.data_getter,
                target_getter=self.target_getter,
                get_key_metric=self.get_key_metric
            )

            if self.visualizer:
                self.visualizer.update_charts(
                    output_train['accuracy'], output_train['loss'], output_test['accuracy'], output_test['loss'],
                    self.optimizer.param_groups[0]['lr'], epoch,
                    self.model, self.loader_train
                )

            self.metrics['epoch'].append(epoch)
            self.metrics['train_loss'].append(output_train['loss'])
            self.metrics['test_loss'].append(output_test['loss'])
            self.metrics['test_accuracy'].append(output_test['accuracy'])

            if self.lr_scheduler is not None:
                if isinstance(self.lr_scheduler, ReduceLROnPlateau):
                    self.lr_scheduler.step(output_train['loss'])
                else:
                    self.lr_scheduler.step()

            if (epoch + 1) % self.model_saving_frequency == 0:
                os.makedirs(self.save_dir, exist_ok=True)
                torch.save(
                    self.model,
                    os.path.join(self.save_dir, self.model_name_prefix) + str(datetime.datetime.now()) + '.pt'
                )
            if output_test['loss'] < best_loss:
                best_loss = output_test['loss']
                os.makedirs(self.save_dir, exist_ok=True)
                torch.save(
                    self.model,
                    os.path.join(self.save_dir, self.model_name_prefix) + '_best.pt'
                )

        return self.metrics


    # Return a dictionary of output metrics with keys:
    #     loss: average loss.
    def train_loop(
        self,
        model, # model (nn.Module): torch model which will be train.
        loader, # loader (torch.utils.DataLoader): dataset loader.
        loss_fn, # loss_fn (callable): loss function.
        metric_fn, # metric_fn (callable): evaluation metric function.
        optimizer, # optimizer (torch.optim.Optimizer): Optimizer.
        device, # device (str): Specifies device at which samples will be uploaded.
        data_getter=itemgetter(0), # data_getter (Callable): function object to extract input data from the sample prepared by dataloader.
        target_getter=itemgetter(1), # target_getter (Callable): function object to extract target data from the sample prepared by dataloader.
        iterator_type=tqdm, # iterator_type (iterator): type of the iterator.
        prefix="", # prefix (string): prefix which will be add to the description string.
        stage_progress=False, # stage_progress (bool): if True then progress bar will be show.
        get_key_metric=itemgetter("accuracy") # get_key_metric (Callable): function object to extract key metric from the metric dictionary.
    ):

        model = model.train()
        iterator = iterator_type(loader, disable=not stage_progress, dynamic_ncols=True)
        loss_avg = AverageMeter()
        for i, sample in enumerate(iterator):
            optimizer.zero_grad()
            inputs = data_getter(sample).to(device)
            targets = target_getter(sample).to(device)
            predicts = model(inputs)
            loss = loss_fn(predicts, targets)
            loss.backward()
            optimizer.step()
            loss_avg.update(loss.item())
            status = "{0}[Train] Loss: {2:.3f} LR:  {4:.5f}".format(
                prefix, i, loss_avg.avg, loss_avg.val, optimizer.param_groups[0]["lr"]
            )
            if get_key_metric is not None:
                status = status + " Acc: {0:.3f}".format(get_key_metric(metric_fn.get_metric_value()))
            iterator.set_description(status)
        
        output = {"accuracy": metric_fn.get_metric_value(), "loss": loss_avg.avg}

        self.model = model

        return output


    # Return a dictionary of output metrics with keys:
    #     loss: average loss.
    #     metric: output metric.
    def test_loop(
        self,
        model, # model (nn.Module): torch model which will be train.
        loader, # loader (torch.utils.DataLoader): dataset loader.
        loss_fn, # loss_fn (callable): loss function.
        metric_fn, # metric_fn (callable): evaluation metric function.
        device, # device (str): Specifies device at which samples will be uploaded.
        data_getter=itemgetter(0), # data_getter (Callable): function object to extract input data from the sample prepared by dataloader.
        target_getter=itemgetter(1), # target_getter (Callable): function object to extract target data from the sample prepared by dataloader.
        iterator_type=tqdm, # iterator_type (iterator): type of the iterator.
        prefix="", # prefix (string): prefix which will be add to the description string.
        stage_progress=False, # stage_progress (bool): if True then progress bar will be show.
        get_key_metric=itemgetter("accuracy") # get_key_metric (Callable): function object to extract key metric from the metric dictionary.
    ):
        
        model = model.eval()
        iterator = iterator_type(loader, disable=not stage_progress, dynamic_ncols=True)
        loss_avg = AverageMeter()
        metric_fn.reset()
        for i, sample in enumerate(iterator):
            inputs = data_getter(sample).to(device)
            targets = target_getter(sample).to(device)
            with torch.no_grad():
                predict = model(inputs)
                loss = loss_fn(predict, targets)
            loss_avg.update(loss.item())
            predict = predict.softmax(dim=1).detach()
            metric_fn.update_value(predict, targets)
            status = "{0}[Test] Loss: {2:.3f}".format(prefix, i, loss_avg.avg)
            if get_key_metric is not None:
                status = status + " Acc: {0:.3f}".format(get_key_metric(metric_fn.get_metric_value()))
            iterator.set_description(status)
            
        output = {"accuracy": metric_fn.get_metric_value(), "loss": loss_avg.avg}
        return output
