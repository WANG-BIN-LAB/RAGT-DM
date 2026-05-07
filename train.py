import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, classification_report
from utils import accuracy, TotalMeter, count_params, isfloat, continus_mixup_data

class Train:
    def __init__(self, cfg, model, optimizers, lr_schedulers, dataloaders, logger, device):
        self.cfg = cfg
        self.logger = logger
        self.model = model
        self.device = device
        self.logger.info(f'#model params: {count_params(self.model)}')
        # Dataloader
        self.train_dataloader, self.test_dataloader = dataloaders
        # Training config
        self.epochs = cfg.training.epochs
        self.total_steps = cfg.total_steps
        self.optimizers = optimizers
        self.lr_schedulers = lr_schedulers
        self.loss_fn = nn.CrossEntropyLoss(reduction='mean')
        # Metric meters
        self.init_meters()

    def init_meters(self):
        """Initialize metric meters"""
        self.train_loss = TotalMeter()
        self.test_loss = TotalMeter()
        self.train_accuracy = TotalMeter()
        self.test_accuracy = TotalMeter()

    def reset_meters(self):
        """Reset metric meters before each epoch"""
        for meter in [self.train_loss, self.test_loss, self.train_accuracy, self.test_accuracy]:
            meter.reset()

    def train_per_epoch(self, optimizer, lr_scheduler):
        """Training process for one epoch"""
        self.model.train()
        self.current_step += 1
        for data in self.train_dataloader:
            time_series, node_feature, label, site, age, sex = data
            label = label.float()
            # Update learning rate (fixed lr here)
            lr_scheduler.update(optimizer=optimizer, step=self.current_step)
            # Move data to device
            time_series = time_series.to(self.device)
            node_feature = node_feature.to(self.device)
            label = label.to(self.device)
            site = site.to(self.device)
            age = age.to(self.device)
            sex = sex.to(self.device)
            # Mixup augmentation
            if self.cfg.preprocess.continus:
                time_series, node_feature, label, site, age, sex = continus_mixup_data(
                    time_series, node_feature, y=label, site=site, age=age, sex=sex,
                    alpha=1.0, device=self.device
                )
            # Forward and loss
            predict, _ = self.model(time_series, node_feature, site, age, sex)
            loss = self.loss_fn(predict, label)
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # Update metrics
            self.train_loss.update_with_weight(loss.item(), label.shape[0])
            top1 = accuracy(predict, label[:, 1])[0]
            self.train_accuracy.update_with_weight(top1, label.shape[0])

    def test_per_epoch(self, dataloader, loss_meter, acc_meter):
        """Evaluation process for one epoch"""
        labels = []
        result = []
        self.model.eval()
        with torch.no_grad():
            for data in dataloader:
                time_series, node_feature, label, site, age, sex = data
                time_series = time_series.to(self.device)
                node_feature = node_feature.to(self.device)
                label = label.to(self.device)
                site = site.to(self.device)
                age = age.to(self.device)
                sex = sex.to(self.device)
                # Forward
                output, _ = self.model(time_series, node_feature, site, age, sex)
                label = label.float()
                # Loss and accuracy
                loss = self.loss_fn(output, label)
                loss_meter.update_with_weight(loss.item(), label.shape[0])
                top1 = accuracy(output, label[:, 1])[0]
                acc_meter.update_with_weight(top1, label.shape[0])
                # Collect results for AUC/F1
                result += F.softmax(output, dim=1)[:, 1].tolist()
                labels += label[:, 1].tolist()
        # Calculate metrics
        auc = roc_auc_score(labels, result)
        result = np.array(result)
        labels = np.array(labels)
        result[result > 0.5] = 1
        result[result <= 0.5] = 0
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, result, average='binary', zero_division=0
        )
        report = classification_report(labels, result, output_dict=True, zero_division=0)
        class_recall = [0, 0]
        for k in report:
            if isfloat(k):
                class_recall[int(float(k))] = report[k]['recall']
        return [auc, precision, recall, f1] + class_recall

    def train(self):
        """Main training loop"""
        training_process = []
        self.current_step = 0
        best_test_acc = 0.0
        best_results = None
        use_cuda = self.device.type == 'cuda'
        epoch_times = []
        mem_records = []

        for epoch in range(self.epochs):
            epoch_start = time.time()
            self.reset_meters()
            # Train one epoch
            self.train_per_epoch(self.optimizers[0], self.lr_schedulers[0])
            # Record GPU memory
            if use_cuda:
                mem = torch.cuda.max_memory_allocated() / (1024 * 1024)
                mem_records.append(mem)
                torch.cuda.reset_max_memory_allocated()
            # Evaluate
            _ = self.test_per_epoch(self.train_dataloader, self.train_loss, self.train_accuracy)
            test_res = self.test_per_epoch(self.test_dataloader, self.test_loss, self.test_accuracy)
            # Record time
            epoch_time = time.time() - epoch_start
            epoch_times.append(epoch_time)
            # Log info
            self.logger.info(
                f'Epoch[{epoch}/{self.epochs}] | Train Loss:{self.train_loss.avg:.3f} | '
                f'Train Acc:{self.train_accuracy.avg:.3f}% | Test Acc:{self.test_accuracy.avg:.3f}% | '
                f'Test Sen:{test_res[-1]:.4f} | Test Spe:{test_res[-2]:.4f} | F1:{test_res[-4]:.4f} | '
                f'AUC:{test_res[0]:.4f} | Time:{epoch_time:.2f}s'
            )
            # Save best model
            current_acc = self.test_accuracy.avg
            if current_acc > best_test_acc:
                best_test_acc = current_acc
                best_results = {
                    "train_accuracy": self.train_accuracy.avg,
                    "test_accuracy": current_acc,
                    "test_auc": test_res[0],
                    "test_sensitivity": test_res[-1],
                    "test_specificity": test_res[-2],
                    "micro_f1": test_res[-4],
                    "micro_recall": test_res[-5],
                    "micro_precision": test_res[-6],
                }
        return best_results