"""
Model Trainer for KineticAugment Evaluation.

Simple trainer for SLR evaluation experiments with early stopping.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from kinetic_augment.evaluation.models.base import (
    BaseSLRClassifier,
    check_torch_available,
    TORCH_AVAILABLE,
)

if TORCH_AVAILABLE:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader


class SLRTrainer:
    """
    Simple trainer for SLR classification models.

    Features:
    - Training loop with validation
    - Early stopping
    - Learning rate scheduling
    - Metrics logging

    Example:
        >>> trainer = SLRTrainer(
        ...     model=LSTMClassifier(num_classes=100),
        ...     train_loader=train_loader,
        ...     val_loader=val_loader,
        ... )
        >>> results = trainer.train(epochs=50, patience=10)
    """

    def __init__(
        self,
        model: BaseSLRClassifier,
        train_loader: 'DataLoader',
        val_loader: 'DataLoader',
        optimizer: Optional['torch.optim.Optimizer'] = None,
        criterion: Optional['nn.Module'] = None,
        device: str = 'cpu',
        scheduler: Optional[Any] = None,
    ):
        """
        Initialize trainer.

        Args:
            model: SLR classifier model
            train_loader: Training data loader
            val_loader: Validation data loader
            optimizer: Optimizer (default: AdamW)
            criterion: Loss function (default: CrossEntropyLoss)
            device: Device to train on ('cpu' or 'cuda')
            scheduler: Optional learning rate scheduler
        """
        check_torch_available()

        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device

        self.optimizer = optimizer or torch.optim.AdamW(
            model.parameters(),
            lr=1e-3,
            weight_decay=0.01,
        )
        self.criterion = criterion or nn.CrossEntropyLoss()
        self.scheduler = scheduler

        # Training history
        self.history: Dict[str, List[float]] = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
        }

    def train(
        self,
        epochs: int = 50,
        patience: int = 10,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """
        Train the model.

        Args:
            epochs: Maximum number of epochs
            patience: Early stopping patience
            verbose: Print progress

        Returns:
            Dictionary with training history and best metrics
        """
        best_val_acc = 0.0
        best_epoch = 0
        patience_counter = 0
        best_state = None

        for epoch in range(epochs):
            # Training
            train_loss, train_acc = self._train_epoch()
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)

            # Validation
            val_metrics = self.evaluate(self.val_loader)
            val_loss = val_metrics['loss']
            val_acc = val_metrics['accuracy']
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)

            # Learning rate scheduling
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()

            # Check for improvement
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch = epoch
                patience_counter = 0
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
            else:
                patience_counter += 1

            if verbose:
                print(
                    f"Epoch {epoch+1:3d}/{epochs}: "
                    f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, "
                    f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}"
                )

            # Early stopping
            if patience_counter >= patience:
                if verbose:
                    print(f"Early stopping at epoch {epoch+1}")
                break

        # Restore best model
        if best_state is not None:
            self.model.load_state_dict(best_state)

        return {
            'history': self.history,
            'best_val_acc': best_val_acc,
            'best_epoch': best_epoch,
            'epochs_trained': epoch + 1,
        }

    def _train_epoch(self) -> Tuple[float, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for batch in self.train_loader:
            # Handle different batch formats
            if len(batch) == 3:
                x, lengths, labels = batch
            else:
                x, labels = batch
                lengths = None

            x = self._to_device(x)
            labels = labels.to(self.device)
            if lengths is not None:
                lengths = lengths.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            logits = self.model(x, lengths)
            loss = self.criterion(logits, labels)

            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            # Metrics
            total_loss += loss.item() * labels.size(0)
            preds = torch.argmax(logits, dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        return total_loss / total, correct / total

    def evaluate(self, loader: 'DataLoader') -> Dict[str, float]:
        """
        Evaluate model on a data loader.

        Args:
            loader: Data loader to evaluate on

        Returns:
            Dictionary with loss, accuracy, top5_accuracy
        """
        self.model.eval()
        total_loss = 0.0
        correct = 0
        correct_top5 = 0
        total = 0

        with torch.no_grad():
            for batch in loader:
                if len(batch) == 3:
                    x, lengths, labels = batch
                else:
                    x, labels = batch
                    lengths = None

                x = self._to_device(x)
                labels = labels.to(self.device)
                if lengths is not None:
                    lengths = lengths.to(self.device)

                logits = self.model(x, lengths)
                loss = self.criterion(logits, labels)

                total_loss += loss.item() * labels.size(0)

                # Top-1 accuracy
                preds = torch.argmax(logits, dim=-1)
                correct += (preds == labels).sum().item()

                # Top-5 accuracy
                num_classes = logits.size(-1)
                k = min(5, num_classes)
                _, top_k = torch.topk(logits, k, dim=-1)
                correct_top5 += (top_k == labels.unsqueeze(1)).any(dim=1).sum().item()

                total += labels.size(0)

        return {
            'loss': total_loss / total,
            'accuracy': correct / total,
            'top5_accuracy': correct_top5 / total,
        }

    def predict(self, loader: 'DataLoader') -> Tuple[np.ndarray, np.ndarray]:
        """
        Get predictions for a data loader.

        Args:
            loader: Data loader

        Returns:
            Tuple of (predictions, labels) as numpy arrays
        """
        self.model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in loader:
                if len(batch) == 3:
                    x, lengths, labels = batch
                else:
                    x, labels = batch
                    lengths = None

                x = self._to_device(x)
                if lengths is not None:
                    lengths = lengths.to(self.device)

                preds = self.model.predict(x, lengths)
                all_preds.append(preds.cpu().numpy())
                all_labels.append(labels.numpy())

        return np.concatenate(all_preds), np.concatenate(all_labels)

    def _to_device(self, value):
        """Move tensor batches recursively while retaining legacy tensor inputs."""
        if isinstance(value, dict):
            return {key: self._to_device(item) for key, item in value.items()}
        if torch.is_tensor(value):
            return value.to(self.device)
        return value

    def get_confusion_matrix(self, loader: 'DataLoader') -> np.ndarray:
        """
        Compute confusion matrix.

        Args:
            loader: Data loader

        Returns:
            Confusion matrix (num_classes, num_classes)
        """
        preds, labels = self.predict(loader)
        num_classes = self.model.num_classes

        conf_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
        for p, l in zip(preds, labels):
            conf_matrix[l, p] += 1

        return conf_matrix
