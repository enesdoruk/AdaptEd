import torch
import numpy as np
import utils
import torch.optim as optim
import torch.nn as nn
import test
import mnist
import mnistm
from utils import (visualize, set_model_mode, save_model, \
                    split_image_to_patches)
import params
import wandb
from dom_distance import calc_distance_dom



# Source : 0, Target :1
source_test_loader = mnist.mnist_test_loader
target_test_loader = mnistm.mnistm_test_loader


def source_only(encoder, classifier, source_train_loader, target_train_loader): 
    print("Training with only the source dataset")

    classifier_criterion = nn.CrossEntropyLoss().cuda()
    optimizer = optim.SGD(
        list(encoder.parameters()) +
        list(classifier.parameters()),
        lr=0.01, momentum=0.9)

    for epoch in range(params.epochs):
        print(f"Epoch: {epoch}")
        set_model_mode('train', [encoder, classifier])

        start_steps = epoch * len(source_train_loader)
        total_steps = params.epochs * len(target_train_loader)
        cls_loss_epoch = []
        dist_loss_epoch = []

        for batch_idx, (source_data, target_data) in enumerate(zip(source_train_loader, target_train_loader)):
            source_image, source_label = source_data
            target_image, target_label = target_data

            p = float(batch_idx + start_steps) / total_steps

            source_image = torch.cat((source_image, source_image, source_image), 1)  # MNIST convert to 3 channel
            source_image, source_label = source_image.cuda(), source_label.cuda()  # 32
            target_image, target_label = target_image.cuda(), target_label.cuda()  # 32

            # source_patches = split_image_to_patches(source_image, patch_size=params.patch_size)
            # target_patches = split_image_to_patches(target_image, patch_size=params.patch_size)
            
            optimizer = utils.optimizer_scheduler(optimizer=optimizer, p=p)
            optimizer.zero_grad()

            source_feature, dom_feats_s = encoder(source_image)
            _, dom_feats_t = encoder(target_image)
            
            distance_loss = calc_distance_dom(dom_feats_s, dom_feats_t)

            # Classification loss
            class_pred = classifier(source_feature)
            class_loss = classifier_criterion(class_pred, source_label)

            total_loss = class_loss + (distance_loss * params.lambda_mmd)
            total_loss.backward()
            optimizer.step()
            if (batch_idx + 1) % 100 == 0:
                total_processed = batch_idx * len(source_image)
                total_dataset = len(source_train_loader.dataset)
                percentage_completed = 100. * batch_idx / len(source_train_loader)
                print(f'[{total_processed}/{total_dataset} ({percentage_completed:.0f}%)]\tClassification Loss: {class_loss.item():.4f}\tMMD Loss: {distance_loss.item()*params.lambda_mmd:.4f}')
            
            cls_loss_epoch.append(class_loss.item())
            dist_loss_epoch.append(distance_loss.item() * params.lambda_mmd)

        source_acc, target_acc = test.tester(encoder, classifier, None, source_test_loader, target_test_loader, training_mode='Source_only')
        wandb.log({"Target Accuracy": target_acc})
        wandb.log({"Source Accuracy": source_acc})
        wandb.log({"Train Loss": np.mean(cls_loss_epoch)})
        wandb.log({"Distance Loss": np.mean(dist_loss_epoch)})
        
        
    save_model(encoder, classifier, None, 'Source-only')
    visualize(encoder, 'Source-only')
