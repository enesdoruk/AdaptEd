import torch
import numpy as np

from utils import set_model_mode


def tester(encoder, classifier, discriminator, source_test_loader, target_test_loader, training_mode):
    encoder.cuda()
    classifier.cuda()
    set_model_mode('eval', [encoder, classifier])

    source_correct = 0
    target_correct = 0

    for batch_idx, (source_data, target_data) in enumerate(zip(source_test_loader, target_test_loader)):
        # Process source and target data
        source_image, source_label = process_data(source_data, expand_channels=True)
        target_image, target_label = process_data(target_data)

        # Compute source and target predictions
        source_pred = compute_output(encoder, classifier, source_image, alpha=None)
        target_pred = compute_output(encoder, classifier, target_image, alpha=None)

        # Update correct counts
        source_correct += source_pred.eq(source_label.data.view_as(source_pred)).sum().item()
        target_correct += target_pred.eq(target_label.data.view_as(target_pred)).sum().item()

       
    source_dataset_len = len(source_test_loader.dataset)
    target_dataset_len = len(target_test_loader.dataset)

    accuracies = {
        "Source": {
            "correct": source_correct,
            "total": source_dataset_len,
            "accuracy": calculate_accuracy(source_correct, source_dataset_len)
        },
        "Target": {
            "correct": target_correct,
            "total": target_dataset_len,
            "accuracy": calculate_accuracy(target_correct, target_dataset_len)
        }
    }


    print_accuracy(training_mode, accuracies)
    
    return calculate_accuracy(source_correct, source_dataset_len), calculate_accuracy(target_correct, target_dataset_len)


def process_data(data, expand_channels=False):
    images, labels = data
    images, labels = images.cuda(), labels.cuda()
    if expand_channels:
        images = images.repeat(1, 3, 1, 1)  # Repeat channels to convert to 3-channel images
    return images, labels


def compute_output(encoder, classifier, images, alpha=None):
    features, dom_feats = encoder(images)
    outputs = classifier(features)  # Category classifier
    preds = outputs.data.max(1, keepdim=True)[1]
    return preds


def calculate_accuracy(correct, total):
    return 100. * correct / total


def print_accuracy(training_mode, accuracies):
    print(f"Test Results on {training_mode}:")
    for key, value in accuracies.items():
        print(f"{key} Accuracy: {value['correct']}/{value['total']} ({value['accuracy']:.2f}%)")
