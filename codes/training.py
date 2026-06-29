"""
training.py - Train the steganography model
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import numpy as np

# Import from utils
from utils import (
    StegoEncoder, StegoDecoder, RobustDataset,
    SSIMLoss, GradientLoss, FrequencyLoss,
    dwt_init, idwt_init, QuantizationLayer
)


def train(args):
    """Main training function"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on: {device}")

    # Dataset
    dataset = RobustDataset(args.data_root)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True
    )

    # Models
    netG = StegoEncoder().to(device)
    netD = StegoDecoder().to(device)
    params = list(netG.parameters()) + list(netD.parameters())

    optimizer = optim.AdamW(params, lr=args.learning_rate, weight_decay=1e-5)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2)

    # Loss functions
    criterion_bce = nn.BCEWithLogitsLoss()
    criterion_mse = nn.MSELoss()
    criterion_l1 = nn.L1Loss()
    criterion_ssim = SSIMLoss().to(device)
    criterion_grad = GradientLoss().to(device)
    criterion_freq = FrequencyLoss().to(device)

    print(">>> Starting Training...\n")

    best_score = float('inf')
    os.makedirs(args.save_dir, exist_ok=True)

    def get_phase_config(epoch):
        """Curriculum learning phases"""
        if epoch < 10:
            return {
                'phase': 'WARMUP', 'alpha': 0.20,
                'w_secret': 100, 'w_mse': 1, 'w_l1': 0,
                'w_ssim': 0, 'w_grad': 0, 'w_freq': 0
            }
        elif epoch < 25:
            return {
                'phase': 'INTRODUCE', 'alpha': 0.12,
                'w_secret': 80, 'w_mse': 20, 'w_l1': 5,
                'w_ssim': 10, 'w_grad': 2, 'w_freq': 5
            }
        elif epoch < 50:
            return {
                'phase': 'BALANCE', 'alpha': 0.08,
                'w_secret': 60, 'w_mse': 50, 'w_l1': 10,
                'w_ssim': 30, 'w_grad': 5, 'w_freq': 15
            }
        elif epoch < 75:
            return {
                'phase': 'QUALITY', 'alpha': 0.06,
                'w_secret': 40, 'w_mse': 100, 'w_l1': 20,
                'w_ssim': 50, 'w_grad': 10, 'w_freq': 25
            }
        else:
            return {
                'phase': 'POLISH', 'alpha': 0.05,
                'w_secret': 30, 'w_mse': 200, 'w_l1': 30,
                'w_ssim': 80, 'w_grad': 15, 'w_freq': 40
            }

    for epoch in range(args.epochs):
        cfg = get_phase_config(epoch)
        total_loss = 0
        metrics = {'ber': 0, 'psnr': 0, 'ssim': 0}

        netG.train()
        netD.train()

        for cover, secret in loader:
            cover, secret = cover.to(device), secret.to(device)

            # DWT decomposition
            cover_dwt = dwt_init(cover)
            middle_bands = cover_dwt[:, 3:9]

            # Generate and embed residuals
            residuals = netG(middle_bands, secret)
            stego_dwt = cover_dwt.clone()
            stego_dwt[:, 3:6] = cover_dwt[:, 3:6] + cfg['alpha'] * residuals[:, 0:3]
            stego_dwt[:, 6:9] = cover_dwt[:, 6:9] + cfg['alpha'] * residuals[:, 3:6]

            # Reconstruct image
            stego_img = torch.clamp(idwt_init(stego_dwt), -1, 1)

            # Decode secret
            rec_logits = netD(dwt_init(stego_img))

            # Compute losses
            loss_secret = criterion_bce(rec_logits, secret)
            loss_mse = criterion_mse(stego_img, cover)
            loss_l1 = criterion_l1(stego_img, cover)
            loss_ssim = criterion_ssim(stego_img, cover)
            loss_grad = criterion_grad(stego_img, cover)
            loss_freq = criterion_freq(cover_dwt, stego_dwt)

            loss = (cfg['w_secret'] * loss_secret +
                    cfg['w_mse'] * loss_mse +
                    cfg['w_l1'] * loss_l1 +
                    cfg['w_ssim'] * loss_ssim +
                    cfg['w_grad'] * loss_grad +
                    cfg['w_freq'] * loss_freq)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()

            # Metrics
            with torch.no_grad():
                preds = (torch.sigmoid(rec_logits) > 0.5).float()
                metrics['ber'] += (preds - secret).abs().mean().item()
                mse_val = loss_mse.item()
                metrics['psnr'] += 10 * np.log10(1.0 / max(mse_val, 1e-10))
                metrics['ssim'] += 1 - loss_ssim.item()

        scheduler.step()

        n = len(loader)
        avg_loss = total_loss / n
        avg_ber = metrics['ber'] / n
        avg_psnr = metrics['psnr'] / n
        avg_ssim = metrics['ssim'] / n

        print(f"Epoch {epoch+1:3d} [{cfg['phase']:9s}] | "
              f"Loss: {avg_loss:7.3f} | BER: {avg_ber:.4f} | "
              f"PSNR: {avg_psnr:5.2f} dB | SSIM: {avg_ssim:.4f}")

        # Save best model
        score = avg_ber * 2 + (40 - avg_psnr) / 40 + (1 - avg_ssim)
        if epoch > 15 and avg_ber < 0.12 and score < best_score:
            best_score = score
            torch.save({
                'encoder': netG.state_dict(),
                'decoder': netD.state_dict(),
                'epoch': epoch,
                'ber': avg_ber,
                'psnr': avg_psnr,
                'ssim': avg_ssim
            }, os.path.join(args.save_dir, "best_model.pth"))
            print(f"  ★ Saved! Score: {score:.4f}")

        # Periodic checkpoint
        if (epoch + 1) % 25 == 0:
            torch.save({
                'encoder': netG.state_dict(),
                'decoder': netD.state_dict(),
            }, os.path.join(args.save_dir, f"checkpoint_epoch{epoch+1}.pth"))

    print("\n>>> Training Complete!")
    print(f">>> Best combined score: {best_score:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Train Steganography Model")
    parser.add_argument("--data_root", type=str, required=True, help="Path to image dataset")
    parser.add_argument("--save_dir", type=str, default="checkpoints", help="Save directory")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--num_workers", type=int, default=2, help="Number of workers")

    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
