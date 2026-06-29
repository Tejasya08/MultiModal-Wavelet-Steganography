"""
sender.py - Encode secrets into cover images
"""

import os
import sys
import argparse
import glob
import cv2
import numpy as np
import torch
from tqdm import tqdm

# Import from utils
from utils import (
    StegoEncoder, StegoDecoder, Encryptor, Decryptor,
    dwt_init, idwt_init, QuantizationLayer,
    prepare_image_secret, prepare_text_secret,
    calc_image_metrics, create_cover_stego_composite
)


class Sender:
    """Steganography sender - embeds secrets into images"""
    
    def __init__(self, config):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.config = config
        self.alpha = config.get('alpha', 0.12)

        # Load encoder
        self.encoder = StegoEncoder().to(self.device)
        checkpoint = torch.load(config['model_path'], map_location=self.device, weights_only=False)
        self.encoder.load_state_dict(checkpoint['encoder'])
        self.encoder.eval()

        self.quantizer = QuantizationLayer().to(self.device)

        # Encryption
        if config.get('use_encryption', False):
            self.encryptor = Encryptor(config.get('encryption_key', 'default'))
        else:
            self.encryptor = None

        # Create output folders
        os.makedirs(config['output_folder'], exist_ok=True)
        os.makedirs(config.get('visualization_folder', 'visualizations'), exist_ok=True)

    def encode_single(self, cover_path, secret_grid, output_path):
        """Encode secret into a single cover image"""
        # Load and resize cover
        cover = cv2.imread(cover_path)
        if cover is None:
            raise FileNotFoundError(f"Cannot load: {cover_path}")
        cover = cv2.resize(cover, (256, 256))

        # Encrypt secret
        if self.encryptor:
            secret_grid = self.encryptor.encrypt(secret_grid).astype(np.float32)

        # Convert to YUV and normalize
        cover_yuv = cv2.cvtColor(cover, cv2.COLOR_BGR2YUV)
        cover_norm = (cover_yuv.astype(np.float32) / 127.5) - 1.0
        cover_tensor = torch.tensor(cover_norm).permute(2, 0, 1).unsqueeze(0).float().to(self.device)
        secret_tensor = torch.tensor(secret_grid).unsqueeze(0).unsqueeze(0).float().to(self.device)

        with torch.no_grad():
            cover_dwt = dwt_init(cover_tensor)
            residuals = self.encoder(cover_dwt[:, 3:9], secret_tensor)

            stego_dwt = cover_dwt.clone()
            stego_dwt[:, 3:6] = cover_dwt[:, 3:6] + self.alpha * residuals[:, 0:3]
            stego_dwt[:, 6:9] = cover_dwt[:, 6:9] + self.alpha * residuals[:, 3:6]

            stego = self.quantizer(torch.clamp(idwt_init(stego_dwt), -1, 1))

        # Convert back to BGR and save
        stego_np = ((stego.squeeze().permute(1, 2, 0).cpu().numpy() + 1) * 127.5).clip(0, 255).astype(np.uint8)
        stego_bgr = cv2.cvtColor(stego_np, cv2.COLOR_YUV2BGR)
        cv2.imwrite(output_path, stego_bgr)

        return calc_image_metrics(cover, stego_bgr)

    def run(self):
        """Process all cover images"""
        print(f"\n{'='*80}")
        print("ENCODING SECRETS")
        print(f"{'='*80}")

        # Prepare secret
        if self.config['secret_type'] == 'image':
            secret_grid = prepare_image_secret(self.config['secret_image'])
            print(f"  Secret: {self.config['secret_image']}")
        else:
            secret_grid = prepare_text_secret(self.config['secret_text'])
            print(f"  Secret: {self.config['secret_text'][:50]}...")

        # Save original secret
        original_img = (secret_grid * 255).astype(np.uint8)
        cv2.imwrite(os.path.join(self.config['output_folder'], "original_secret.png"), original_img)

        # Find covers
        covers = []
        for ext in ['*.png', '*.jpg', '*.jpeg']:
            covers.extend(glob.glob(os.path.join(self.config['cover_folder'], ext)))
        print(f"  Covers: {len(covers)} images found\n")

        print(f"{'Filename':<30} | {'PSNR':>7} | {'SSIM':>6} | {'MSE':>8} | {'UIQI':>6} | {'VIF':>6}")
        print("-"*80)

        results = []
        vis_cover_paths, vis_stego_paths, vis_metrics = [], [], []
        vis_samples = self.config.get('num_visualization_samples', 5)

        for idx, path in enumerate(tqdm(covers, desc="Encoding")):
            fn = os.path.basename(path)
            output = os.path.join(self.config['output_folder'], f"stego_{fn}")

            metrics = self.encode_single(path, secret_grid, output)
            results.append(metrics)

            print(f"stego_{fn:<24} | {metrics['psnr']:>7.2f} | {metrics['ssim']:>6.4f} | "
                  f"{metrics['mse']:>8.4f} | {metrics['uiqi']:>6.4f} | {metrics['vif']:>6.4f}")

            if idx < vis_samples:
                vis_cover_paths.append(path)
                vis_stego_paths.append(output)
                vis_metrics.append(metrics)

        # Summary
        print("-"*80)
        avg_psnr = np.mean([r['psnr'] for r in results])
        avg_ssim = np.mean([r['ssim'] for r in results])
        print(f"{'AVERAGE':<30} | {avg_psnr:>7.2f} | {avg_ssim:>6.4f}")

        # Create visualizations
        if vis_cover_paths:
            print(f"\n{'='*80}")
            print("CREATING VISUALIZATIONS")
            print(f"{'='*80}")
            composite_path = os.path.join(
                self.config['visualization_folder'],
                "cover_stego_composite.png"
            )
            create_cover_stego_composite(
                vis_cover_paths, vis_stego_paths, vis_metrics, composite_path
            )

        print(f"\n✓ Stego images saved to: {self.config['output_folder']}/")
        print(f"✓ Visualizations saved to: {self.config['visualization_folder']}/")


def main():
    parser = argparse.ArgumentParser(description="Steganography Sender")
    parser.add_argument("--secret_type", type=str, default="image", choices=["image", "text"])
    parser.add_argument("--secret_image", type=str, default="", help="Path to secret image")
    parser.add_argument("--secret_text", type=str, default="", help="Secret text message")
    parser.add_argument("--cover_folder", type=str, required=True, help="Folder with cover images")
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained model")
    parser.add_argument("--output_folder", type=str, default="stego_output", help="Output folder")
    parser.add_argument("--alpha", type=float, default=0.12, help="Embedding strength")
    parser.add_argument("--use_encryption", action="store_true", help="Enable encryption")
    parser.add_argument("--encryption_key", type=str, default="MySecretKey12345")
    parser.add_argument("--vis_samples", type=int, default=5, help="Number of visualization samples")

    args = parser.parse_args()

    config = {
        'secret_type': args.secret_type,
        'secret_image': args.secret_image,
        'secret_text': args.secret_text,
        'cover_folder': args.cover_folder,
        'model_path': args.model_path,
        'output_folder': args.output_folder,
        'alpha': args.alpha,
        'use_encryption': args.use_encryption,
        'encryption_key': args.encryption_key,
        'visualization_folder': f"{args.output_folder}_visualizations",
        'num_visualization_samples': args.vis_samples,
    }

    sender = Sender(config)
    sender.run()


if __name__ == "__main__":
    main()
