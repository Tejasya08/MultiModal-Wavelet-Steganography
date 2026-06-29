"""
receiver.py - Extract secrets from stego images
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
    prepare_image_secret, prepare_text_secret, extract_text_from_bits,
    calc_image_metrics, create_extraction_composite
)


class Receiver:
    """Steganography receiver - extracts secrets from images"""
    
    def __init__(self, config):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.config = config

        # Load decoder
        self.decoder = StegoDecoder().to(self.device)
        checkpoint = torch.load(config['model_path'], map_location=self.device, weights_only=False)
        self.decoder.load_state_dict(checkpoint['decoder'])
        self.decoder.eval()

        # Decryption
        if config.get('use_encryption', False):
            self.decryptor = Decryptor(config.get('encryption_key', 'default'))
        else:
            self.decryptor = None

        # Create output folders
        os.makedirs(config['output_folder'], exist_ok=True)
        os.makedirs(config.get('visualization_folder', 'visualizations'), exist_ok=True)

    def extract_single(self, stego_path):
        """Extract secret from a single stego image"""
        stego = cv2.imread(stego_path)
        if stego is None:
            raise FileNotFoundError(f"Cannot load: {stego_path}")

        stego = cv2.resize(stego, (256, 256))
        stego_yuv = cv2.cvtColor(stego, cv2.COLOR_BGR2YUV)
        stego_norm = (stego_yuv.astype(np.float32) / 127.5) - 1.0
        stego_tensor = torch.tensor(stego_norm).permute(2, 0, 1).unsqueeze(0).float().to(self.device)

        with torch.no_grad():
            logits = self.decoder(dwt_init(stego_tensor))
            extracted = (torch.sigmoid(logits) > 0.5).float().squeeze().cpu().numpy()

        if self.decryptor:
            extracted = self.decryptor.decrypt(extracted).astype(np.float32)

        return extracted

    def run(self):
        """Process all stego images"""
        print(f"\n{'='*80}")
        print("EXTRACTING SECRETS")
        print(f"{'='*80}")

        stegos = []
        for ext in ['stego_*.png', 'stego_*.jpg', 'stego_*.jpeg']:
            stegos.extend(glob.glob(os.path.join(self.config['stego_folder'], ext)))

        if not stegos:
            print("ERROR: No stego images found!")
            return

        print(f"  Found: {len(stegos)} stego images\n")

        # Load original secret for comparison if provided
        original_secret = None
        if self.config.get('original_secret') and os.path.exists(self.config['original_secret']):
            orig = cv2.imread(self.config['original_secret'], cv2.IMREAD_GRAYSCALE)
            if orig is not None:
                orig = cv2.resize(orig, (128, 128))
                original_secret = (orig > 127).astype(np.float32)

        all_extracted = []
        vis_stego_paths = []
        vis_extracted_secrets = []
        vis_samples = self.config.get('num_visualization_samples', 3)

        print(f"{'Filename':<30} | {'Output':<25} | {'Status'}")
        print("-"*70)

        for idx, path in enumerate(tqdm(stegos, desc="Extracting")):
            fn = os.path.basename(path)

            try:
                extracted = self.extract_single(path)

                # Save extracted as image
                output = os.path.join(
                    self.config['output_folder'],
                    fn.replace("stego_", "extracted_")
                )
                cv2.imwrite(output, (extracted * 255).astype(np.uint8))

                # Try to extract text if in text mode
                if self.config.get('secret_type') == 'text':
                    text, error = extract_text_from_bits(extracted)
                    if error:
                        status = f"Error: {error}"
                    else:
                        status = f"Text: {text[:30]}..." if len(text) > 30 else f"Text: {text}"
                    print(f"{fn:<30} | {os.path.basename(output):<25} | {status}")
                else:
                    # For image mode, check BER if original provided
                    if original_secret is not None:
                        ber = np.mean(np.abs(original_secret.flatten() - extracted.flatten()))
                        status = f"BER: {ber:.4f}"
                    else:
                        status = "Extracted ✓"
                    print(f"{fn:<30} | {os.path.basename(output):<25} | {status}")

                all_extracted.append({'filename': fn, 'extracted': extracted})

                # Store for visualization
                if idx < vis_samples:
                    vis_stego_paths.append(path)
                    vis_extracted_secrets.append(extracted)

            except Exception as e:
                print(f"{fn:<30} | {'ERROR':<25} | {str(e)}")

        # Create visualizations
        if vis_stego_paths:
            print(f"\n{'='*80}")
            print("CREATING VISUALIZATIONS")
            print(f"{'='*80}")

            composite_path = os.path.join(
                self.config['visualization_folder'],
                "extraction_composite.png"
            )
            create_extraction_composite(
                vis_stego_paths, vis_extracted_secrets,
                original_secret, composite_path
            )

        print(f"\n✓ Extracted secrets saved to: {self.config['output_folder']}/")
        print(f"✓ Visualizations saved to: {self.config['visualization_folder']}/")


def main():
    parser = argparse.ArgumentParser(description="Steganography Receiver")
    parser.add_argument("--stego_folder", type=str, required=True, help="Folder with stego images")
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained model")
    parser.add_argument("--output_folder", type=str, default="extracted_secrets", help="Output folder")
    parser.add_argument("--original_secret", type=str, default="", help="Path to original secret for comparison")
    parser.add_argument("--secret_type", type=str, default="image", choices=["image", "text"])
    parser.add_argument("--use_encryption", action="store_true", help="Enable decryption")
    parser.add_argument("--encryption_key", type=str, default="MySecretKey12345")
    parser.add_argument("--vis_samples", type=int, default=3, help="Number of visualization samples")

    args = parser.parse_args()

    config = {
        'stego_folder': args.stego_folder,
        'model_path': args.model_path,
        'output_folder': args.output_folder,
        'original_secret': args.original_secret,
        'secret_type': args.secret_type,
        'use_encryption': args.use_encryption,
        'encryption_key': args.encryption_key,
        'visualization_folder': f"{args.output_folder}_visualizations",
        'num_visualization_samples': args.vis_samples,
    }

    receiver = Receiver(config)
    receiver.run()


if __name__ == "__main__":
    main()
