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

    command_pose_path: str = "Project/TrainingModel/data/command_pose.npy"
    volume_control_path: str = "Project/TrainingModel/data/volume_control.npy"
    next_path: str = "Project/TrainingModel/data/next.npy"
    previous_path: str = "Project/TrainingModel/data/previous.npy"
    play_pause_path: str = "Project/TrainingModel/data/play_pause.npy"
    none_path: str = "Project/TrainingModel/data/none.npy"

    num_classes: int = 6 # number of classes in the dataset

@dataclass
class DataloaderConfig:
    batch_size: int = 256
    num_workers: int = 1 

@dataclass
class OptimizerConfig:
    learning_rate: float = 0.002 # learning rate of SGD
    momentum: float = 0.95 # SGD momentum (How much to reuse the previous update directionm percent) 
    weight_decay: float = 0.0001 # SGD weight decay
    lr_step_milestones: Iterable = (20, 25, 30, 35, 40, 45) # epoch to start changing learning rate
    lr_gamma: float = 0.7 # learning rate multiplier when reaching milestone

@dataclass
class TrainerConfig:
    model_dir: str = "checkpoints"  # directory to save model states
    model_saving_frequency: int = 50  # frequency of model state savings per epochs
    device: str = "gpu"  # device to use for training.
    epoch_num: int = 50  # number of times the whole dataset will be passed through the network
    progress_bar: bool = True  # enable progress bar visualization during train process
