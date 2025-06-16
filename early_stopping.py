from typing import Callable, Literal
import torch


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self,
                 patience: int = 7,
                 threshold = 0,
                 threshold_mode: Literal['rel', 'abs'] = 'abs',
                 mode: Literal['min', 'max'] = 'min',
                 verbose: bool = False,
                 path: str = 'checkpoint.pt',
                 trace_func: Callable = print
                 ):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.threshold_mode = threshold_mode
        self.counter = 0
        self.best_val_score = None
        self.early_stop = False
        self.threshold = threshold
        self.mode = {'min': 1, 'max': -1}[mode]
        self.path = path
        self.trace_func = trace_func

    def _significant_improvement(self, val_score) -> bool:
        if self.threshold_mode == 'abs':
            return self.mode * (self.best_val_score - val_score) > self.threshold
        else:
            return self.mode * (self.best_val_score - val_score) / self.best_val_score > self.threshold

    def __call__(self, val_score, model):
        # Check if validation loss is nan
        if torch.isnan(torch.tensor(val_score, dtype=torch.float)):
            self.trace_func("Validation score is NaN. Ignoring this epoch.")
            return

        if self.best_val_score is None:
            self.best_val_score = val_score
            self.save_checkpoint(model)
        elif self._significant_improvement(val_score):
            if self.verbose:
                self.trace_func(f'Validation score {"decreased" if self.mode == 1 else "increased"} ({self.best_val_score:.6f} --> {val_score:.6f}).  Saving model...')
            self.best_val_score = val_score
            self.save_checkpoint(model)
            self.counter = 0  # Reset counter since improvement occurred
        else:
            # No significant improvement
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
                return True
        return False

    def save_checkpoint(self, model: torch.nn.Module):
        '''Saves model when validation loss decreases.'''
        torch.save(model.state_dict(), self.path)