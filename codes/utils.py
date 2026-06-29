"""
utils.py - Shared utilities for steganography
Contains: Models, Loss Functions, DWT/IDWT, Metrics, Visualizations
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import os
import hashlib
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error as mse_metric
from scipy.ndimage import uniform_filter
from torch.utils.data import Dataset, DataLoader
import glob


# ============================================
# 1. DWT / IDWT
# ============================================

def dwt_init(x):
    """ 2D Haar Discrete Wavelet Transform """
    x01 = x[:, :, 0::2, :] / 2
    x02 = x[:, :, 1::2, :] / 2
    x1 = x01[:, :, :, 0::2]
    x2 = x02[:, :, :, 0::2]
    x3 = x01[:, :, :, 1::2]
    x4 = x02[:, :, :, 1::2]
    return torch.cat((x1 + x2 + x3 + x4, -x1 - x2 + x3 + x4, 
                      -x1 + x2 - x3 + x4, x1 - x2 - x3 + x4), 1)


def idwt_init(x):
    """ 2D Inverse Haar Discrete Wavelet Transform """
    r = 2
    in_b, in_c, in_h, in_w = x.size()
    out_c = in_c // 4
    c = [x[:, i * out_c:(i + 1) * out_c] / 2 for i in range(4)]
    h = torch.zeros([in_b, out_c, r * in_h, r * in_w], device=x.device)
    h[:, :, 0::2, 0::2] = c[0] - c[1] - c[2] + c[3]
    h[:, :, 1::2, 0::2] = c[0] - c[1] + c[2] - c[3]
    h[:, :, 0::2, 1::2] = c[0] + c[1] - c[2] - c[3]
    h[:, :, 1::2, 1::2] = c[0] + c[1] + c[2] + c[3]
    return h


# ============================================
# 2. Models
# ============================================

class SEBlock(nn.Module):
    """ Squeeze-and-Excitation Channel Attention """
    def __init__(self, ch, reduction=8):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(ch, ch // reduction), nn.ReLU(),
            nn.Linear(ch // reduction, ch), nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        w = self.fc(self.pool(x).view(b, c)).view(b, c, 1, 1)
        return x * w


class ResBlockSE(nn.Module):
    """ Residual Block with SE Attention """
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1), nn.BatchNorm2d(dim), 
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(dim, dim, 3, padding=1), nn.BatchNorm2d(dim), 
            SEBlock(dim)
        )
        self.act = nn.LeakyReLU(0.2, True)

    def forward(self, x):
        return self.act(self.block(x) + x)


class StegoEncoder(nn.Module):
    """ Channel-Aware Encoder - embeds less in Y, more in UV """
    def __init__(self):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(7, 64, 3, padding=1), nn.LeakyReLU(0.2, True)
        )
        self.body = nn.Sequential(*[ResBlockSE(64) for _ in range(6)])

        self.y_branch = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1), nn.LeakyReLU(0.2),
            nn.Conv2d(32, 2, 3, padding=1), nn.Tanh()
        )
        self.uv_branch = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1), nn.LeakyReLU(0.2),
            nn.Conv2d(32, 4, 3, padding=1), nn.Tanh()
        )
        self.y_scale = 0.25

    def forward(self, c_bands, secret):
        x = self.body(self.head(torch.cat([c_bands, secret], dim=1)))
        y_out = self.y_branch(x) * self.y_scale
        uv_out = self.uv_branch(x)
        hl = torch.cat([y_out[:, 0:1], uv_out[:, 0:2]], dim=1)
        lh = torch.cat([y_out[:, 1:2], uv_out[:, 2:4]], dim=1)
        return torch.cat([hl, lh], dim=1)


class StegoDecoder(nn.Module):
    """ Decoder for recovering hidden secrets """
    def __init__(self):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(12, 64, 3, padding=1), nn.LeakyReLU(0.2, True)
        )
        self.body = nn.Sequential(*[ResBlockSE(64) for _ in range(6)])
        self.tail = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1), nn.LeakyReLU(0.2),
            nn.Conv2d(32, 1, 3, padding=1)
        )

    def forward(self, x):
        return self.tail(self.body(self.head(x)))


class QuantizationLayer(nn.Module):
    """ Simulates PNG/JPEG quantization """
    def forward(self, x):
        x_img = (x + 1.0) * 127.5
        x_rounded = torch.round(x_img)
        x_norm = (x_rounded / 127.5) - 1.0
        return x + (x_norm - x).detach()


# ============================================
# 3. Loss Functions
# ============================================

class SSIMLoss(nn.Module):
    """ Structural Similarity Index Loss """
    def __init__(self, window_size=11):
        super().__init__()
        self.window_size = window_size
        self.window = self._create_window(window_size)

    def _gaussian(self, window_size, sigma=1.5):
        coords = torch.arange(window_size, dtype=torch.float32) - window_size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g /= g.sum()
        return g

    def _create_window(self, window_size):
        _1d = self._gaussian(window_size).unsqueeze(1)
        _2d = _1d.mm(_1d.t()).unsqueeze(0).unsqueeze(0)
        return _2d

    def forward(self, img1, img2):
        channel = img1.size(1)
        window = self.window.expand(channel, 1, -1, -1).to(img1.device).type(img1.dtype)
        
        mu1 = F.conv2d(img1, window, padding=self.window_size//2, groups=channel)
        mu2 = F.conv2d(img2, window, padding=self.window_size//2, groups=channel)
        
        mu1_sq, mu2_sq = mu1**2, mu2**2
        mu1_mu2 = mu1 * mu2
        
        sigma1_sq = F.conv2d(img1*img1, window, padding=self.window_size//2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2*img2, window, padding=self.window_size//2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1*img2, window, padding=self.window_size//2, groups=channel) - mu1_mu2
        
        C1, C2 = 0.01**2, 0.03**2
        ssim = ((2*mu1_mu2 + C1) * (2*sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        return 1 - ssim.mean()


class GradientLoss(nn.Module):
    """ Edge preservation loss using Sobel operators """
    def __init__(self):
        super().__init__()
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
        self.register_buffer('kernel_x', sobel_x.view(1, 1, 3, 3))
        self.register_buffer('kernel_y', sobel_y.view(1, 1, 3, 3))

    def forward(self, pred, target):
        loss = 0
        for c in range(pred.size(1)):
            p, t = pred[:, c:c+1], target[:, c:c+1]
            loss += F.l1_loss(F.conv2d(p, self.kernel_x, padding=1), 
                             F.conv2d(t, self.kernel_x, padding=1))
            loss += F.l1_loss(F.conv2d(p, self.kernel_y, padding=1), 
                             F.conv2d(t, self.kernel_y, padding=1))
        return loss / pred.size(1)


class FrequencyLoss(nn.Module):
    """ Frequency domain loss with band-specific weights """
    def forward(self, cover_dwt, stego_dwt):
        ll_loss = F.mse_loss(stego_dwt[:, 0:3], cover_dwt[:, 0:3]) * 20.0
        hh_loss = F.mse_loss(stego_dwt[:, 9:12], cover_dwt[:, 9:12]) * 3.0
        mid_loss = F.mse_loss(stego_dwt[:, 3:9], cover_dwt[:, 3:9]) * 1.0
        return ll_loss + hh_loss + mid_loss


# ============================================
# 4. Dataset
# ============================================

class RobustDataset(Dataset):
    """ Dataset for steganography training """
    def __init__(self, root_dir, img_size=256, secret_size=128):
        self.img_size = img_size
        self.secret_size = secret_size
        self.files = (
            glob.glob(os.path.join(root_dir, "**/*.png"), recursive=True) +
            glob.glob(os.path.join(root_dir, "**/*.jpg"), recursive=True) +
            glob.glob(os.path.join(root_dir, "**/*.jpeg"), recursive=True)
        )
        print(f">>> Images Found: {len(self.files)}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        try:
            img_bgr = cv2.imread(self.files[idx])
            if img_bgr is None:
                raise Exception("Invalid")
            img_bgr = cv2.resize(img_bgr, (self.img_size, self.img_size))
            img_yuv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YUV)
            img_norm = (img_yuv.astype(np.float32) / 127.5) - 1.0
            cover = torch.tensor(img_norm).permute(2, 0, 1).float()
            secret = np.random.randint(0, 2, (1, self.secret_size, self.secret_size)).astype(np.float32)
            return cover, torch.tensor(secret).float()
        except:
            return self.__getitem__(np.random.randint(0, len(self)))


# ============================================
# 5. Encryption / Decryption
# ============================================

class Encryptor:
    """ XOR-based encryption for binary secrets """
    def __init__(self, key, secret_size=128):
        self.secret_size = secret_size
        self.key = hashlib.sha256(key.encode()).digest()
        np.random.seed(int.from_bytes(self.key[:4], 'big'))
        self.stream = np.random.randint(0, 2, secret_size * secret_size, dtype=np.uint8)

    def encrypt(self, bits):
        flat = bits.flatten().astype(np.uint8)
        encrypted = np.bitwise_xor(flat, self.stream[:len(flat)])
        return encrypted.reshape(bits.shape)


class Decryptor:
    """ XOR-based decryption for binary secrets """
    def __init__(self, key, secret_size=128):
        self.secret_size = secret_size
        self.key = hashlib.sha256(key.encode()).digest()
        np.random.seed(int.from_bytes(self.key[:4], 'big'))
        self.stream = np.random.randint(0, 2, secret_size * secret_size, dtype=np.uint8)

    def decrypt(self, bits):
        flat = bits.flatten().astype(np.uint8)
        decrypted = np.bitwise_xor(flat, self.stream[:len(flat)])
        return decrypted.reshape(bits.shape)


# ============================================
# 6. Metrics
# ============================================

def calculate_npcr(image1, image2):
    """ Number of Pixels Change Rate (%) """
    image1 = image1.astype(np.int32)
    image2 = image2.astype(np.int32)
    if len(image1.shape) == 3:
        diff = np.any(image1 != image2, axis=2)
        total = image1.shape[0] * image1.shape[1]
    else:
        diff = (image1 != image2)
        total = image1.size
    return (np.sum(diff) / total) * 100


def calculate_uaci(image1, image2):
    """ Unified Average Changing Intensity (%) """
    image1 = image1.astype(np.float64)
    image2 = image2.astype(np.float64)
    diff = np.abs(image1 - image2)
    return (np.sum(diff) / (255.0 * image1.size)) * 100


def calculate_uiqi(image1, image2, window_size=8):
    """ Universal Image Quality Index (UIQI) """
    image1 = image1.astype(np.float64)
    image2 = image2.astype(np.float64)
    if len(image1.shape) == 3:
        image1 = cv2.cvtColor(image1.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float64)
        image2 = cv2.cvtColor(image2.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float64)
    
    mu1 = uniform_filter(image1, size=window_size)
    mu2 = uniform_filter(image2, size=window_size)
    sigma1_sq = uniform_filter(image1 ** 2, size=window_size) - mu1 ** 2
    sigma2_sq = uniform_filter(image2 ** 2, size=window_size) - mu2 ** 2
    sigma12 = uniform_filter(image1 * image2, size=window_size) - mu1 * mu2
    sigma1_sq = np.maximum(sigma1_sq, 0)
    sigma2_sq = np.maximum(sigma2_sq, 0)
    
    numerator = 4 * sigma12 * mu1 * mu2
    denominator = (sigma1_sq + sigma2_sq) * (mu1 ** 2 + mu2 ** 2)
    valid = denominator > 1e-10
    uiqi_map = np.zeros_like(numerator)
    uiqi_map[valid] = numerator[valid] / denominator[valid]
    return np.mean(uiqi_map)


def calculate_vif(image1, image2, sigma_nsq=2):
    """ Visual Information Fidelity (VIF) """
    image1 = image1.astype(np.float64)
    image2 = image2.astype(np.float64)
    if len(image1.shape) == 3:
        image1 = cv2.cvtColor(image1.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float64)
        image2 = cv2.cvtColor(image2.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float64)
    
    eps = 1e-10
    window_size = 11
    mu1 = uniform_filter(image1, size=window_size)
    mu2 = uniform_filter(image2, size=window_size)
    mu1_sq, mu2_sq = mu1 ** 2, mu2 ** 2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = uniform_filter(image1 ** 2, size=window_size) - mu1_sq
    sigma2_sq = uniform_filter(image2 ** 2, size=window_size) - mu2_sq
    sigma12 = uniform_filter(image1 * image2, size=window_size) - mu1_mu2
    sigma1_sq = np.maximum(sigma1_sq, 0)
    sigma2_sq = np.maximum(sigma2_sq, 0)
    
    g = sigma12 / (sigma1_sq + eps)
    sv_sq = sigma2_sq - g * sigma12
    g = np.maximum(g, 0)
    sv_sq = np.maximum(sv_sq, eps)
    
    num = np.log2(1 + g ** 2 * sigma1_sq / (sv_sq + sigma_nsq))
    den = np.log2(1 + sigma1_sq / sigma_nsq)
    vif = np.sum(num) / (np.sum(den) + eps)
    return np.clip(vif, 0, 1)


# ============================================
# 7. Visualization
# ============================================

def create_cover_stego_composite(cover_paths, stego_paths, metrics_list, save_path, img_size=256):
    """ Create cover-stego comparison composite """
    n_samples = len(cover_paths)
    fig = plt.figure(figsize=(20, 4 * n_samples))
    gs = GridSpec(n_samples, 4, figure=fig, hspace=0.2, wspace=0.15)
    
    for i, (cover_path, stego_path, metrics) in enumerate(zip(cover_paths, stego_paths, metrics_list)):
        cover = cv2.resize(cv2.imread(cover_path), (img_size, img_size))
        stego = cv2.resize(cv2.imread(stego_path), (img_size, img_size))
        
        cover_rgb = cv2.cvtColor(cover, cv2.COLOR_BGR2RGB)
        stego_rgb = cv2.cvtColor(stego, cv2.COLOR_BGR2RGB)
        
        # Cover
        ax_cover = fig.add_subplot(gs[i, 0])
        ax_cover.imshow(cover_rgb)
        ax_cover.set_title(f'Cover {i+1}', fontsize=12, fontweight='bold')
        ax_cover.axis('off')
        
        # Stego
        ax_stego = fig.add_subplot(gs[i, 1])
        ax_stego.imshow(stego_rgb)
        ax_stego.set_title(f'Stego {i+1}', fontsize=12, fontweight='bold')
        ax_stego.axis('off')
        
        # Difference
        diff = np.abs(cover.astype(np.float32) - stego.astype(np.float32))
        diff_gray = np.mean(diff, axis=2)
        diff_gray = (diff_gray / diff_gray.max() * 255) if diff_gray.max() > 0 else diff_gray
        ax_diff = fig.add_subplot(gs[i, 2])
        ax_diff.imshow(diff_gray, cmap='hot', vmin=0, vmax=10)
        ax_diff.set_title('Difference Map', fontsize=12)
        ax_diff.axis('off')
        
        # Metrics
        ax_metrics = fig.add_subplot(gs[i, 3])
        ax_metrics.axis('off')
        metrics_text = f"""
        Sample {i+1} Metrics:
        PSNR: {metrics['psnr']:.2f} dB
        SSIM: {metrics['ssim']:.4f}
        MSE: {metrics['mse']:.4f}
        UIQI: {metrics['uiqi']:.4f}
        VIF: {metrics['vif']:.4f}
        NPCR: {metrics['npcr']:.2f}%
        UACI: {metrics['uaci']:.4f}%
        """
        ax_metrics.text(0.05, 0.5, metrics_text, fontsize=10, verticalalignment='center',
                       transform=ax_metrics.transAxes, fontfamily='monospace')
    
    plt.suptitle('Cover-Stego Image Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Composite saved: {save_path}")


def create_extraction_composite(stego_paths, extracted_secrets, original_secret, save_path, img_size=256):
    """ Create extraction results composite. """
    n_samples = min(3, len(stego_paths))
    fig = plt.figure(figsize=(18, 12))
    fig.suptitle('Steganography Extraction Results', fontsize=14, fontweight='bold')
    
    gs = GridSpec(3, n_samples + 1, figure=fig, hspace=0.3, wspace=0.2)
    
    # Original secret
    if original_secret is not None:
        orig_display = (original_secret * 255).astype(np.uint8)
        ax_orig = fig.add_subplot(gs[:2, 0])
        ax_orig.imshow(orig_display, cmap='gray')
        ax_orig.set_title('Original Secret', fontsize=12, fontweight='bold')
        ax_orig.axis('off')
    
    for i, (stego_path, extracted) in enumerate(zip(stego_paths[:n_samples], extracted_secrets[:n_samples])):
        col = i + 1
        
        stego = cv2.resize(cv2.imread(stego_path), (img_size, img_size))
        stego_rgb = cv2.cvtColor(stego, cv2.COLOR_BGR2RGB)
        ax_stego = fig.add_subplot(gs[0, col])
        ax_stego.imshow(stego_rgb)
        ax_stego.set_title(f'Stego {i+1}', fontsize=12, fontweight='bold')
        ax_stego.axis('off')
        
        extracted_display = (extracted * 255).astype(np.uint8)
        if extracted_display.shape != (128, 128):
            extracted_display = cv2.resize(extracted_display, (128, 128))
        ax_extracted = fig.add_subplot(gs[1, col])
        ax_extracted.imshow(extracted_display, cmap='gray')
        ax_extracted.set_title(f'Extracted {i+1}', fontsize=12)
        ax_extracted.axis('off')
        
        if original_secret is not None:
            orig_binary = (original_secret > 0.5).astype(np.uint8)
            extracted_binary = (extracted > 0.5).astype(np.uint8)
            diff = np.abs(orig_binary - extracted_binary) * 255
            diff = cv2.resize(diff, (128, 128))
            ax_diff = fig.add_subplot(gs[2, col])
            ax_diff.imshow(diff, cmap='hot')
            error_pixels = np.sum(diff > 0) / (128 * 128) * 100
            ax_diff.set_title(f'Errors: {error_pixels:.2f}%', fontsize=10)
            ax_diff.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Extraction composite saved: {save_path}")


def calc_image_metrics(original, stego):
    """ Calculate all image quality metrics """
    return {
        'psnr': psnr(original, stego, data_range=255),
        'ssim': ssim(cv2.cvtColor(original, cv2.COLOR_BGR2GRAY),
                     cv2.cvtColor(stego, cv2.COLOR_BGR2GRAY), data_range=255),
        'mse': mse_metric(original.astype(float), stego.astype(float)),
        'npcr': calculate_npcr(original, stego),
        'uaci': calculate_uaci(original, stego),
        'uiqi': calculate_uiqi(original, stego),
        'vif': calculate_vif(original, stego),
    }


# ============================================
# 8. Helper Functions
# ============================================

def prepare_image_secret(path, secret_size=128):
    """ Convert image to binary secret """
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot load: {path}")
    img = cv2.resize(img, (secret_size, secret_size))
    _, binary = cv2.threshold(img, 127, 1, cv2.THRESH_BINARY)
    return binary.astype(np.float32)


def prepare_text_secret(text, secret_size=128):
    """Convert text to binary grid"""
    text_bytes = text.encode('utf-8')
    length = len(text_bytes)
    header = length.to_bytes(2, 'big')
    payload = header + text_bytes
    bits = np.unpackbits(np.frombuffer(payload, dtype=np.uint8))
    grid = np.zeros(secret_size * secret_size, dtype=np.float32)
    grid[:len(bits)] = bits
    return grid.reshape(secret_size, secret_size)


def extract_text_from_bits(extracted_bits):
    """ Convert extracted bits back to text """
    bits = extracted_bits.flatten().astype(np.uint8)
    if len(bits) < 16:
        return None, "Insufficient bits for header"
    header_bytes = np.packbits(bits[:16])
    length = int.from_bytes(header_bytes.tobytes(), 'big')
    if length <= 0 or length > 2000:
        return None, f"Invalid length: {length}"
    text_bits = bits[16:16 + length * 8]
    if len(text_bits) < length * 8:
        return None, f"Insufficient text bits"
    text_bytes = np.packbits(text_bits).tobytes()[:length]
    try:
        return text_bytes.decode('utf-8'), None
    except Exception as e:
        return None, f"UTF-8 decode failed: {e}"
