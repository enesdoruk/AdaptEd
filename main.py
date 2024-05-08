import torch
import train
import mnist
import mnistm
import model
import wandb


def main():
    wandb.init(project='AdaptEd', name='local_adapt', sync_tensorboard=True)
    
    source_train_loader = mnist.mnist_train_loader
    target_train_loader = mnistm.mnistm_train_loader

    if torch.cuda.is_available():
        encoder = model.Extractor().cuda()
        classifier = model.Classifier().cuda()
        discriminator = model.Discriminator(in_channels=512).cuda()

        train.source_only(encoder, classifier, discriminator, source_train_loader, target_train_loader)
    else:
        print("No GPUs available.")


if __name__ == "__main__":
    main()
