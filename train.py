import torch
import torch.nn as nn
import torch.optim as optim
import config
from utils import save_some_examples, save_checkpoint, load_checkpoint, run_inference
from dataset import MapDataset
from generator_model import Generator
from discriminator_model import Discriminator
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse

torch.backends.cudnn.benchmark = True

def parse_arguments():
    parser = argparse.ArgumentParser(description="GAN Training Script")
    parser.add_argument('--c', action='store_true', help="Flag to continue training.")
    parser.add_argument('--i', action='store_true', help="Flag to run inference.")
    
    return parser.parse_args()

def train_fn(disc, gen, loader, opt_disc, opt_gen, l1, bce, g_scaler, d_scaler):
    loop = tqdm(loader, leave=True)

    for idx, (x,y) in enumerate(loop):
        x, y = x.to(config.DEVICE), y.to(config.DEVICE)

        # Train Discriminator
        with torch.cuda.amp.autocast():
            y_fake = gen(x) # Generates fake samples with the generator gen using the real data x.
            D_real = disc(x,y) # Discriminator outputs a prediction for the real samples x, y.
            D_fake = disc(x, y_fake.detach()) # Discriminator outputs a prediction for the fake samples. .detach() is used to prevent gradients from being calculated for the generator during the discriminator training.
            
            # D_real_loss and D_fake_loss are the losses for real and fake predictions, respectively, using Binary CrossEntropy (bce) loss.
            D_real_loss = bce(D_real, torch.ones_like(D_real)) # example: bce([0.9, 0.8, 0.7], [1, 1, 1])
            D_fake_loss = bce(D_fake, torch.zeros_like(D_fake))
            D_loss = (D_real_loss + D_fake_loss) / 2 # The total discriminator loss

        disc.zero_grad() # Gradients of the discriminator parameters are set to zero.
        d_scaler.scale(D_loss).backward() # computes the gradient of the loss with respect to model parameters.
        d_scaler.step(opt_disc) # The optimizer opt_disc updates the discriminator weights.
        d_scaler.update() # Updates the gradient scaler.

        # without gradientScaler it would be:
        # disc.zero_grad() OR opt_disc.zero_grad()
        # D_loss.backward()
        # opt_disc.step()

        # Train Generator
        with torch.cuda.amp.autocast():
            D_fake = disc(x, y_fake) # Discriminator outputs a prediction for the fake samples without detached gradients since we want to update the generator.
            G_fake_loss = bce(D_fake, torch.ones_like(D_fake)) # The generator tries to minimize the binary cross-entropy loss between D_fake and real labels to fool the discriminator.
            L1 = l1(y_fake, y) * config.L1_LAMBDA #  L1 loss between the generated samples and the real samples, scaled by config.L1_LAMBDA.
            G_loss = G_fake_loss + L1 # The total generator loss is the sum of the GAN loss and the L1 loss.

        # Zeroing the generator gradients, computing the gradients, updating the generator parameters, and updating the gradient scaler respectively.
        opt_gen.zero_grad()
        g_scaler.scale(G_loss).backward()
        g_scaler.step(opt_gen)
        g_scaler.update()

        if idx % 10 == 0:
            loop.set_postfix(
                D_real=torch.sigmoid(D_real).mean().item(),
                D_fake=torch.sigmoid(D_fake).mean().item(),
            )

def main():

    print("DEVICE:" + config.DEVICE)

    args = parse_arguments()

    disc = Discriminator(in_channels=3).to(config.DEVICE)
    gen = Generator(in_channels=3, features=64).to(config.DEVICE)
    opt_disc = optim.Adam(disc.parameters(), lr=config.LEARNING_RATE, betas=(0.5, 0.999))
    opt_gen = optim.Adam(gen.parameters(), lr=config.LEARNING_RATE, betas=(0.5, 0.999))
    BCE = nn.BCEWithLogitsLoss() # standard GAN loss
    L1_LOSS = nn.L1Loss()

    train_dataset = MapDataset(root_dir=config.TRAIN_DIR)
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=config.NUM_WORKERS)

    # Gradient Scaling is used to prevent underflow in the gradients during mixed-precision training 
    # (training that uses both float16 and float32 data types to utilize lesser memory and computational resources). 
    g_scaler = torch.cuda.amp.GradScaler()  
    d_scaler = torch.cuda.amp.GradScaler()

    val_dataset = MapDataset(root_dir=config.VAL_DIR)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)


    if args.i:
        load_checkpoint(config.CHECKPOINT_GEN, gen, opt_gen, config.LEARNING_RATE)
        run_inference(gen, val_loader, folder="results")
    else: # train

        if args.c:
            load_checkpoint(config.CHECKPOINT_GEN, gen, opt_gen, config.LEARNING_RATE)
            load_checkpoint(config.CHECKPOINT_DISC, disc, opt_disc, config.LEARNING_RATE)

        for epoch in range(config.NUM_EPOCHS):
            train_fn(disc, gen, train_loader, opt_disc, opt_gen, L1_LOSS, BCE, g_scaler, d_scaler)

            if config.SAVE_MODEL and epoch%5==0:
                save_checkpoint(gen, opt_gen, filename=config.CHECKPOINT_GEN)
                save_checkpoint(disc, opt_disc, filename=config.CHECKPOINT_DISC)
            
            save_some_examples(gen, val_loader, epoch, folder="evaluation")

if __name__ == "__main__":
    main()
