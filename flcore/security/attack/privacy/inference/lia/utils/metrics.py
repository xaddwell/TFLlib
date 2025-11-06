import numpy as np
from sklearn.metrics import accuracy_score

def get_label_stats(gt_label, num_classes):
    from collections import Counter
    LabelCounter = dict(Counter(gt_label))
    labels = list(sorted(LabelCounter.keys()))
    existences = [1 if i in labels else 0 for i in range(num_classes)]
    num_instances = [LabelCounter[i] if i in labels else 0 for i in range(num_classes)]
    num_instances_nonzero = [item[1] for item in sorted(LabelCounter.items(), key=lambda x: x[0])]
    return labels, existences, num_instances, num_instances_nonzero

def _print_util(string, logger):
    print(string)
    if logger is not None:
        logger.log(round = -1, identity=f"Server-Attacker", action="Measurement", message=string)


def lacc(gt_label, num_classes, rec_instances, rec_labels, num_images, simplified, logger=None):
    if len(rec_labels) == 0:
        _print_util('Our Attack failed to Recovered labels', logger=logger)
        return None 
    for i in range(len(rec_labels)):
        rec_labels[i] = int(rec_labels[i])
    labels, existences, num_instances, num_instances_nonzero = get_label_stats(gt_label, num_classes)
    rec_existences = [1 if i in rec_labels else 0 for i in range(num_classes)]
    # Calculate Class-wise Acc, Instance-wise Acc and Recall
    leacc = 1.0 if simplified else accuracy_score(existences, rec_existences)
    lnacc = accuracy_score(num_instances_nonzero if simplified else num_instances, list(rec_instances))
    irec = sum([rec_instances[i] if rec_instances[i] <= num_instances_nonzero[i] else num_instances_nonzero[i] for i in
                range(len(labels))]) / num_images if simplified else sum(
        [rec_instances[i] if rec_instances[i] <= num_instances[i] else num_instances[i] for i in labels]) / num_images
    rec_instances_nonzero = rec_instances if simplified else rec_instances[rec_labels]
    # Print results
    _print_util('Ground-truth Labels: ' + ','.join(str(l) for l in labels), logger=logger)
    _print_util('Ground-truth Num of Instances: ' + ','.join(str(num_instances[l]) for l in labels), logger=logger)
    _print_util('Our Recovered Labels: ' + ','.join(str(l) for l in rec_labels) + ' | LeAcc: %.3f' % leacc,
            logger=logger)
    prefix = 'Our Recovered Num of Instances by Simplified Method: ' if simplified else 'Our Recovered Num of Instances: '
    _print_util(prefix + ','.join(str(l) for l in list(rec_instances_nonzero)) +
            ' | LnAcc: %.3f | IRec: %.3f' % (
                lnacc, irec),
            logger=logger)
    metrics = [leacc, lnacc, irec]
    return metrics