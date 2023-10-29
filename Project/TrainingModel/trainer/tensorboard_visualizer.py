import torch
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt

from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import confusion_matrix
from datetime import datetime


class TensorBoardVisualizer():
    def __init__(self):
        current_time = datetime.now().strftime("%B_%d_%Y_%I_%M%p")
        log_dir = f"Project/TrainingModel/runs/{current_time}"
        self._writer = SummaryWriter(log_dir=log_dir)


    def create_confusion_matrix(self, net, loader):
        y_pred = [] # save prediction
        y_true = [] # save ground truth

        # iterate over data
        for inputs, labels in loader:
            inputs = inputs.to('cuda')
            output = net(inputs)

            output = (torch.max(torch.exp(output), 1)[1]).data.cpu().numpy()
            y_pred.extend(output)  # save prediction

            labels = labels.data.cpu().numpy()
            y_true.extend(labels)  # save ground truth

        # constant for classes
        classes = (
            "command_pose",
            "volume_control",
            "next",
            "previous",
            "play_pause",
            "none"
        )

        # Build confusion matrix
        cf_matrix = confusion_matrix(y_true, y_pred, normalize='true')
        df_cm = pd.DataFrame(
            cf_matrix, index=[i for i in classes],
            columns=[i for i in classes]
        )
        plt.figure(figsize=(12, 7))
        return sn.heatmap(df_cm, annot=True).get_figure()

    def update_charts(self, train_metric, train_loss, test_metric, test_loss, learning_rate, epoch, model, loader):
        # Train Accuracy
        for metric_key, metric_value in train_metric.items():
            self._writer.add_scalar("data/train_accuracy:{}".format(metric_key), metric_value, epoch)

        # Test Accuracy
        for test_metric_key, test_metric_value in test_metric.items():
            self._writer.add_scalar("data/test_accuracy:{}".format(test_metric_key), test_metric_value, epoch)

        if train_loss is not None:
            self._writer.add_scalar("data/train_loss", train_loss, epoch)
        if test_loss is not None:
            self._writer.add_scalar("data/test_loss", test_loss, epoch)

        self._writer.add_scalar("data/learning_rate", learning_rate, epoch)

        current_time = datetime.now().strftime("%B_%d_%Y_%I_%M%p")
        self._writer.add_figure(f"{current_time}", self.create_confusion_matrix(model, loader))

    def close_tensorboard(self):
        self._writer.close()
