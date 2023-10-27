# # <font style="color:blue">Configurations</font>

from typing import Iterable
from dataclasses import dataclass

@dataclass
class SystemConfig:
    seed: int = 42  # seed number to set the state of all random number generators
    cudnn_benchmark_enabled: bool = True  # enable CuDNN benchmark for the sake of performance
    cudnn_deterministic: bool = True  # make cudnn deterministic (reproducible training), radom is consistent

@dataclass
class DatasetConfig:

    command_pose_path: str = "HandRemote/TrainModel/data/command_pose.npy"
    volume_up_path: str = "HandRemote/TrainModel/data/volume_up.npy"
    volume_down_path: str = "HandRemote/TrainModel/data/volume_down.npy"
    next_path: str = "HandRemote/TrainModel/data/next.npy"
    previous_path: str = "HandRemote/TrainModel/data/previous.npy"
    play_pause_path: str = "HandRemote/TrainModel/data/play_pause.npy"

    num_classes: int = 2 # number of classes in the dataset

@dataclass
class DataloaderConfig:
    batch_size: int = 128
    num_workers: int = 4

@dataclass
class OptimizerConfig:
    learning_rate: float = 0.01
    momentum: float = 0.95 # SGD momentum (How much to reuse the previous update directionm percent) 
    weight_decay: float = 0.0001 # SGD weight decay
    lr_step_milestones: Iterable = (100,200,300,400,500) # epoch to start changing learning rate
    lr_gamma: float = 0.1 # learning rate multiplier when reaching milestone

@dataclass
class TrainerConfig:
    model_dir: str = "checkpoints"  # directory to save model states
    model_saving_frequency: int = 50  # frequency of model state savings per epochs
    device: str = "gpu"  # device to use for training.
    epoch_num: int = 5  # number of times the whole dataset will be passed through the network
    progress_bar: bool = True  # enable progress bar visualization during train process
