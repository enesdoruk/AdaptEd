import torch
import numpy as np
import utils
import torch.optim as optim
import torch.nn as nn
import test
import mnist
import mnistm
from utils import save_model
from utils import visualize
from utils import set_model_mode
import params
import wandb

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

        for batch_idx, (source_data, target_data) in enumerate(zip(source_train_loader, target_train_loader)):
            source_image, source_label = source_data
            p = float(batch_idx + start_steps) / total_steps

            source_image = torch.cat((source_image, source_image, source_image), 1)  # MNIST convert to 3 channel
            source_image, source_label = source_image.cuda(), source_label.cuda()  # 32

            optimizer = utils.optimizer_scheduler(optimizer=optimizer, p=p)
            optimizer.zero_grad()

            source_feature = encoder(source_image)

            # Classification loss
            class_pred = classifier(source_feature)
            class_loss = classifier_criterion(class_pred, source_label)

            class_loss.backward()
            optimizer.step()
            if (batch_idx + 1) % 100 == 0:
                total_processed = batch_idx * len(source_image)
                total_dataset = len(source_train_loader.dataset)
                percentage_completed = 100. * batch_idx / len(source_train_loader)
                print(f'[{total_processed}/{total_dataset} ({percentage_completed:.0f}%)]\tClassification Loss: {class_loss.item():.4f}')
            
            cls_loss_epoch.append(class_loss.item())

        source_acc, target_acc = test.tester(encoder, classifier, None, source_test_loader, target_test_loader, training_mode='Source_only')
        wandb.log({"Target Accuracy": target_acc})
        wandb.log({"Source Accuracy": source_acc})
        wandb.log({"Train Loss": np.mean(cls_loss_epoch)})
        
        
    save_model(encoder, classifier, None, 'Source-only')
    visualize(encoder, 'Source-only')
