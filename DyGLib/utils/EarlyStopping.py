import os
import json
import torch
import torch.nn as nn
import logging


class EarlyStopping(object):

    def __init__(self, patience: int, save_model_folder: str, save_model_name: str = None,
                 logger: logging.Logger = None, model_name: str = None, save_model_file: str = 'model.pt',
                 config: dict = None):
        """
        Early stop strategy.
        :param patience: int, max patience
        :param save_model_folder: str, save model folder
        :param save_model_name: str, save model name (legacy path <folder>/<name>.pkl)
        :param logger: Logger
        :param model_name: str, model name
        :param save_model_file: str, file name for the best model, used when save_model_name is None
        :param config: dict, configuration written next to the best model as config.json
        """
        self.patience = patience
        self.counter = 0
        self.best_metrics = {}
        self.best_epoch = 0
        self.early_stop = False
        self.logger = logger
        self.model_name = model_name
        if save_model_name is not None:
            self.save_model_path = os.path.join(save_model_folder, f"{save_model_name}.pkl")
        else:
            self.save_model_path = os.path.join(save_model_folder, save_model_file)
        self.config_path = os.path.join(os.path.dirname(self.save_model_path), 'config.json')
        self.config = config
        if self.model_name in ['JODIE', 'DyRep', 'TGN']:
            # path to additionally save the nonparametric data (e.g., tensors) in memory-based models (e.g., JODIE, DyRep, TGN)
            self.save_model_nonparametric_data_path = os.path.join(os.path.dirname(self.save_model_path),
                                                                   f"{os.path.splitext(os.path.basename(self.save_model_path))[0]}_nonparametric_data.pkl")

    def step(self, metrics: list, model: nn.Module, epoch: int = None):
        """
        execute the early stop strategy for each evaluation process
        :param metrics: list, list of metrics, each element is a tuple (str, float, boolean) -> (metric_name, metric_value, whether higher means better)
        :param model: nn.Module
        :param epoch: int, epoch of the current evaluation step
        :return:
        """
        metrics_compare_results = []
        for metric_tuple in metrics:
            metric_name, metric_value, higher_better = metric_tuple[0], metric_tuple[1], metric_tuple[2]

            if higher_better:
                if self.best_metrics.get(metric_name) is None or metric_value >= self.best_metrics.get(metric_name):
                    metrics_compare_results.append(True)
                else:
                    metrics_compare_results.append(False)
            else:
                if self.best_metrics.get(metric_name) is None or metric_value <= self.best_metrics.get(metric_name):
                    metrics_compare_results.append(True)
                else:
                    metrics_compare_results.append(False)
        # all the computed metrics are better than the best metrics
        if torch.all(torch.tensor(metrics_compare_results)):
            for metric_tuple in metrics:
                metric_name, metric_value = metric_tuple[0], metric_tuple[1]
                self.best_metrics[metric_name] = metric_value
            if epoch is not None:
                self.best_epoch = epoch
            self.save_checkpoint(model)
            self.counter = 0
        # metrics are not better at the epoch
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

        return self.early_stop

    def save_checkpoint(self, model: nn.Module):
        """
        saves model at self.save_model_path together with a config.json (when a config was provided)
        :param model: nn.Module
        :return:
        """
        os.makedirs(os.path.dirname(self.save_model_path), exist_ok=True)
        self.logger.info(f"save model {self.save_model_path}")
        torch.save(model.state_dict(), self.save_model_path)
        if self.model_name in ['JODIE', 'DyRep', 'TGN']:
            torch.save(model[0].memory_bank.node_raw_messages, self.save_model_nonparametric_data_path)
        if self.config is not None:
            cfg = dict(self.config)
            cfg.update({
                "best_epoch": int(self.best_epoch),
                "model_file": os.path.basename(self.save_model_path),
                "config_file": "config.json",
            })
            for metric_name, metric_value in self.best_metrics.items():
                cfg[f"best_{metric_name}"] = float(metric_value) if isinstance(metric_value, (int, float)) else str(metric_value)
            with open(self.config_path, 'w') as f:
                json.dump(cfg, f, indent=2, default=str)

    def load_checkpoint(self, model: nn.Module, map_location: str = None):
        """
        load model at self.save_model_path
        :param model: nn.Module
        :param map_location: str, how to remap the storage locations
        :return:
        """
        self.logger.info(f"load model {self.save_model_path}")
        model.load_state_dict(torch.load(self.save_model_path, map_location=map_location))
        if self.model_name in ['JODIE', 'DyRep', 'TGN']:
            model[0].memory_bank.node_raw_messages = torch.load(self.save_model_nonparametric_data_path, map_location=map_location)
