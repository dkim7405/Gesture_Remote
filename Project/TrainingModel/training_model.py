from operator import itemgetter

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from torch.optim.lr_scheduler import MultiStepLR

from trainer import Trainer, configuration
from trainer.utils import setup_system, patch_configs
from trainer.metrics import AccuracyEstimator
from trainer.tensorboard_visualizer import TensorBoardVisualizer
from torch.utils.data import Dataset, DataLoader


class GestureDataset(Dataset):
    def __init__(self):

        self.data_loaded = np.concatenate([
            np.load(configuration.DatasetConfig.command_pose_path),
            np.load(configuration.DatasetConfig.volume_up_path),
            # np.load(configuration.DatasetConfig.volume_down_path),
            # np.load(configuration.DatasetConfig.next_path),
            # np.load(configuration.DatasetConfig.previous_path),
            # np.load(configuration.DatasetConfig.play_pause_path)
        ])

        self.data_points = self.data_loaded[:, :-1].astype(np.float32)
        self.data_labels = self.data_loaded[:, -1].astype(np.int64) 

    def __len__(self):
        return len(self.data_loaded)
    
    def __getitem__(self, idx):
        return self.data_points[idx], self.data_labels[idx]


def get_data(batch_size, num_workers):

    dataset = GestureDataset()

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_set, test_set = torch.utils.data.random_split(dataset, [train_size, test_size])
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    return train_loader, test_loader


class GestureDetector(nn.Module):
    def __init__(self):
        
        super().__init__()

        self.head = nn.Sequential(
            nn.Linear(in_features=99, out_features=32),
            nn.Linear(in_features=32, out_features=2)
        )
        
    def forward(self, x):
        x = self.head(x)
        return x


class Experiment:
    def __init__(
        self,
        system_config: configuration.SystemConfig = configuration.SystemConfig(),
        dataloader_config: configuration.DataloaderConfig = configuration.DataloaderConfig(),
        optimizer_config: configuration.OptimizerConfig = configuration.OptimizerConfig()
    ):
        self.loader_train, self.loader_test = get_data(
            batch_size=dataloader_config.batch_size,
            num_workers=dataloader_config.num_workers,
        )

        setup_system(system_config)

        self.model = GestureDetector()
        self.loss_fn = nn.CrossEntropyLoss()
        self.metric_fn = AccuracyEstimator(topk=(1, ))
        self.optimizer = optim.SGD(
            self.model.parameters(),
            lr=optimizer_config.learning_rate,
            weight_decay=optimizer_config.weight_decay,
            momentum=optimizer_config.momentum
        )

        self.lr_scheduler = MultiStepLR(
            self.optimizer, milestones=optimizer_config.lr_step_milestones, gamma=optimizer_config.lr_gamma
        )

        self.visualizer = TensorBoardVisualizer()

    def run(self, trainer_config: configuration.TrainerConfig) -> dict:

        device = torch.device(trainer_config.device)
        self.model = self.model.to(device)
        self.loss_fn = self.loss_fn.to(device)

        model_trainer = Trainer(
            model=self.model,
            loader_train=self.loader_train,
            loader_test=self.loader_test,
            loss_fn=self.loss_fn,
            metric_fn=self.metric_fn,
            optimizer=self.optimizer,
            lr_scheduler=self.lr_scheduler,
            device=device,
            data_getter=itemgetter(0),
            target_getter=itemgetter(1),
            stage_progress=trainer_config.progress_bar,
            get_key_metric=itemgetter("top1"),
            visualizer=self.visualizer,
            model_saving_frequency=trainer_config.model_saving_frequency,
            save_dir=trainer_config.model_dir
        )

        self.metrics = model_trainer.fit(trainer_config.epoch_num)
        return self.metrics

if __name__ == "__main__":
    dataloader_config, trainer_config = patch_configs()
    experiment = Experiment(dataloader_config=dataloader_config)
    results = experiment.run(trainer_config)
