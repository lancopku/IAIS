import torch


def get_tsa_threshold(schedule, global_step, num_train_steps):
    training_progress = torch.tensor(global_step, dtype=torch.float) / torch.tensor(num_train_steps, dtype=torch.float)
    if schedule == "linear_schedule":
        threshold = training_progress
    elif schedule == "exp_schedule":
        scale = 5
        threshold = torch.exp((training_progress - 1) * scale)
        # [exp(-5), exp(0)] = [1e-2, 1]
    elif schedule == "log_schedule":
        scale = 5
        # [1 - exp(0), 1 - exp(-5)] = [0, 0.99]
        threshold = 1 - torch.exp((-training_progress) * scale)
    else:
        raise ValueError('schedule must in [linear_schedule, exp_schedule, log_schedule]')
    return threshold