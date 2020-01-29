import math
import sys
import torch
import time
from ..utils import utils
from ..utils.metric_logger import MetricLogger, SmoothedValue


def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq, writer):
    model.train()
    metric_logger = MetricLogger(delimiter="  ", writer=writer)
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    epoch_results = {}

    lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1. / 1000
        warmup_iters = min(1000, len(data_loader) - 1)

        lr_scheduler = utils.warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)

    for images, targets in metric_logger.log_every(data_loader, print_freq, epoch, header):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)

        losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        loss_value = losses_reduced.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training:\n{}".format(loss_value, loss_dict))
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])


@torch.no_grad()
def evaluate(model, data_loader, device, writer, epoch):
    n_threads = torch.get_num_threads()
    # FIXME remove this and make paste_masks_in_image run on the GPU
    torch.set_num_threads(1)
    cpu_device = torch.device("cpu")
    model.eval()
    metric_logger = MetricLogger(delimiter="  ", writer=writer)
    header = 'Test:'

    total, correct = 0, 0

    for image, targets in metric_logger.log_every(data_loader, 100, epoch=epoch, header=header):
        image = list(img.to(device) for img in image)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        targets_labels = torch.as_tensor([int(1 in target["labels"]) for target in targets], dtype=torch.int8)

        torch.cuda.synchronize()
        model_time = time.time()
        outputs = model(image)

        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
        outputs_labels = torch.as_tensor([int(1 in output["labels"]) for output in outputs], dtype=torch.int8)
        model_time = time.time() - model_time

        total += len(image)
        correct += (targets_labels == outputs_labels).sum().item()

        res = {target["image_id"].item(): output for target, output in zip(targets, outputs)}
        evaluator_time = time.time()
        evaluator_time = time.time() - evaluator_time
        metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)

    print("Test accuracy :", correct/total)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    torch.set_num_threads(n_threads)
    writer.add_scalar("Accuracy/eval", correct/total, epoch)
