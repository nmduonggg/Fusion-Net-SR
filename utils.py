import os
import torch
import shutil
import scipy.stats as stats

def save_args(__file__, args):
    shutil.copy(os.path.basename(__file__), args.cv_dir)
    with open(args.cv_dir+'/args.txt','w') as f:
        f.write(str(args))

def confidence_interval_mean(estimated_mean, estimated_stddev, sample_size, confidence_level):
    """
    Calculate the confidence interval for the population mean.
    
    Parameters:
        estimated_mean (float): The estimated mean of the sample.
        estimated_stddev (float): The estimated standard deviation of the sample.
        sample_size (int): The size of the sample.
        confidence_level (float): The desired confidence level (e.g., 0.95 for 95% confidence).
    
    Returns:
        tuple: A tuple containing the lower and upper bounds of the confidence interval.
    """
    # Calculate the critical value based on the confidence level
    z_critical = stats.norm.ppf((1 + confidence_level) / 2)  # Two-tailed test
    
    # Calculate the margin of error
    margin_of_error = z_critical * (abs(estimated_stddev) / (sample_size ** 0.5))
    
    # Calculate the confidence interval
    lower_bound = estimated_mean - margin_of_error
    upper_bound = estimated_mean + margin_of_error
    
    return lower_bound, upper_bound

class LrScheduler:
    def __init__(self, optimizer, base_lr, lr_decay_ratio, epoch_step):
        self.base_lr = base_lr
        self.lr_decay_ratio = lr_decay_ratio
        self.epoch_step = epoch_step
        self.optimizer = optimizer

    def adjust_learning_rate(self, epoch):
        """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
        lr = self.base_lr * (self.lr_decay_ratio ** (epoch // self.epoch_step))
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
            if epoch%self.epoch_step==0:
                print('[INFO] Setting learning_rate to %.2E'%lr)