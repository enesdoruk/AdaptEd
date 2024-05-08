import torch
import numpy as np
import cv2
import wandb

from utils import set_model_mode

def tester(encoder, classifier, discriminator, source_test_loader, target_test_loader, epoch, training_mode):
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
        source_pred = compute_output(encoder, classifier, source_image, epoch, trg_img=target_image, vis=False)
    
        target_pred = compute_output(encoder, classifier, target_image, epoch, trg_img=None, vis=False)

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


def compute_output(encoder, classifier, images, epoch, trg_img=None, vis=False):
    features = encoder(images)[0]
    outputs = classifier(features)  # Category classifier
    preds = outputs.data.max(1, keepdim=True)[1]

    if vis:
        if trg_img is not None:
            gradients = encoder.get_activations_gradient()
            pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])
            activations = encoder.get_activations(encoder, trg_img).detach()
            for i in range(activations.size(1)):
                activations[:, i, :, :] *= pooled_gradients[i]
            heatmap = torch.mean(activations, dim=1).squeeze()
            heatmap = heatmap.detach().cpu()
            heatmap = np.maximum(heatmap, 0)
            heatmap /= torch.max(heatmap)
            
            for i in range(heatmap.size(0)):
                try:
                    heatmap = cv2.resize(heatmap[i].detach().cpu().numpy(), (256, 256))
                    heatmap = np.uint8(255 * heatmap)
                    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
                    img = np.uint8(255 * cv2.resize(trg_img[i].detach().cpu().numpy().transpose(1,2,0), (256,256)))
                    superimposed_img = heatmap * 0.4 + img
                    images = wandb.Image(superimposed_img, caption=f"Image {i}, Epoch: {epoch}")
                    wandb.log({"GradCam": images})
                except:
                    heatmap = cv2.resize(heatmap[i], (256, 256))
                    heatmap = np.uint8(255 * heatmap)
                    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
                    img = np.uint8(255 * cv2.resize(trg_img[i].cpu().numpy().transpose(1,2,0), (256,256))) 
                    superimposed_img = heatmap * 0.4 + img
                    images = wandb.Image(superimposed_img, caption=f"Image {i}, Epoch: {epoch}")
                    wandb.log({"GradCam": images})
     
        
    return preds


def calculate_accuracy(correct, total):
    return 100. * correct / total


def print_accuracy(training_mode, accuracies):
    print(f"Test Results on {training_mode}:")
    for key, value in accuracies.items():
        print(f"{key} Accuracy: {value['correct']}/{value['total']} ({value['accuracy']:.2f}%)")
