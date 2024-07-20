import GAN
from trainer import Trainer
from dataset import Dataset
from tensorboardX import SummaryWriter

from pytorch_fid import fid_score
from torchvision.utils import make_grid
from torchvision.utils import save_image

import torch
import torch.optim as optim
import os
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--do_train', action='store_true')
    parser.add_argument('--no_cuda', action='store_true')
    parser.add_argument('--latent_dim', default=16, type=int)
    parser.add_argument('--generator_hidden_dim', default=16, type=int)
    parser.add_argument('--discriminator_hidden_dim', default=16, type=int)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--num_training_steps', default=5000, type=int)
    parser.add_argument('--logging_steps', type=int, default=10)
    parser.add_argument('--saving_steps', type=int, default=1000)
    parser.add_argument('--learning_rate', default=0.0002, type=float)
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
    parser.add_argument('--data_dir', default='../data', type=str, help='The path of the data directory')
    parser.add_argument('--ckpt_dir', default='results', type=str, help='The path of the checkpoint directory')
    parser.add_argument('--log_dir', default='./runs', type=str)
    parser.add_argument('--interpolation_batch', default=5, type=int)
    parser.add_argument('--interpolation_K', default=10, type=int)
    args = parser.parse_args()

    config = 'z-{}-{}_batch-{}_num-train-steps-{}'.format(args.latent_dim, args.generator_hidden_dim, args.batch_size, args.num_training_steps)
    args.ckpt_dir = os.path.join(args.ckpt_dir, config)
    args.log_dir = os.path.join(args.log_dir, config)
    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')

    dataset = Dataset(args.batch_size, args.data_dir)
    netG = GAN.get_generator(1, args.latent_dim, args.generator_hidden_dim, device)
    netD = GAN.get_discriminator(1, args.discriminator_hidden_dim, device)
    tb_writer = SummaryWriter(args.log_dir)

    if args.do_train:
        optimG = optim.Adam(netG.parameters(), lr=args.learning_rate, betas=(args.beta1, 0.999))
        optimD = optim.Adam(netD.parameters(), lr=args.learning_rate, betas=(args.beta1, 0.999))
        trainer = Trainer(device, netG, netD, optimG, optimD, dataset, args.ckpt_dir, tb_writer)
        trainer.train(args.num_training_steps, args.logging_steps, args.saving_steps)

    restore_ckpt_path = os.path.join(args.ckpt_dir, str(max(int(step) for step in os.listdir(args.ckpt_dir))))
    netG.restore(restore_ckpt_path)

    num_samples = 3000
    real_imgs = None
    real_dl = iter(dataset.training_loader)
    while real_imgs is None or real_imgs.size(0) < num_samples:
        imgs = next(real_dl)
        if real_imgs is None:
            real_imgs = imgs[0]
        else:
            real_imgs = torch.cat((real_imgs, imgs[0]), 0)
    real_imgs = real_imgs[:num_samples].expand(-1, 3, -1, -1) * 0.5 + 0.5
    
    # interpolation
    # Reference: yangjx21
    # I tried to implement this in a 'PyTorch way'. However I failed.
    with torch.no_grad():
        z_1 = torch.randn(args.interpolation_batch, netG.latent_dim, 1, 1, device=device)
        z_2 = torch.randn(args.interpolation_batch, netG.latent_dim, 1, 1, device=device)
        final_imgs = []
        for i in range(z_1.size(0)):
            interpolated_imgs = []
            for i_over_K in torch.linspace(0, 1, steps=args.interpolation_K):
                z = z_1[i] + i_over_K * (z_2[i] - z_1[i])
                z = z.unsqueeze(0)  # add batch dimension
                img = netG(z)
                interpolated_imgs.append(img)
            
            row = torch.cat(interpolated_imgs, 0)
            final_imgs.append(row)
        final_imgs = torch.cat(final_imgs, 0)
        final_imgs = make_grid(final_imgs, nrow=10) * 0.5 + 0.5
        save_image(final_imgs, os.path.join(args.ckpt_dir, "interpolation.png"))
        
    # extrapolation
    with torch.no_grad():
        z_1 = torch.randn(args.interpolation_batch, netG.latent_dim, 1, 1, device=device)
        z_2 = torch.randn(args.interpolation_batch, netG.latent_dim, 1, 1, device=device)
        final_imgs = []
        for i in range(z_1.size(0)):
            interpolated_imgs = []
            for i_over_K in torch.linspace(1, 2, steps=args.interpolation_K):
                z = z_1[i] + i_over_K * (z_2[i] - z_1[i])
                z = z.unsqueeze(0)  # add batch dimension
                img = netG(z)
                interpolated_imgs.append(img)
            
            row = torch.cat(interpolated_imgs, 0)
            final_imgs.append(row)
        final_imgs = torch.cat(final_imgs, 0)
        final_imgs = make_grid(final_imgs, nrow=10) * 0.5 + 0.5
        save_image(final_imgs, os.path.join(args.ckpt_dir, "extrapolation.png"))
        
        
    # Generate samples
    with torch.no_grad():
        samples = netG.forward(torch.randn(100, netG.latent_dim, 1, 1, device=device))
        sample_imgs = make_grid(samples, nrow=10) * 0.5 + 0.5
        save_image(sample_imgs, os.path.join(args.ckpt_dir, "samples.png"))
            

    with torch.no_grad():
        samples = None
        while samples is None or samples.size(0) < num_samples:
            imgs = netG.forward(torch.randn(args.batch_size, netG.latent_dim, 1, 1, device=device))
            if samples is None:
                samples = imgs
            else:
                samples = torch.cat((samples, imgs), 0)
    samples = samples[:num_samples].expand(-1, 3, -1, -1) * 0.5 + 0.5
    samples = samples.cpu()

    fid = fid_score.calculate_fid_given_images(real_imgs, samples, args.batch_size, device)
    tb_writer.add_scalar('fid', fid)
    print("FID score: {:.3f}".format(fid), flush=True)