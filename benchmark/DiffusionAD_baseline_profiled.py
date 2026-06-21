import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler
from torchvision import transforms
from PIL import Image
import numpy as np
import os
import glob
from tqdm import tqdm
import random
import math
import re

try:
    from pytorch_msssim import ssim
except ImportError:
    pass

import time
import psutil
from functools import wraps

def profiled(name=None, synchronize_cuda=True):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            label = name or func.__name__
            if synchronize_cuda and torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
            
            start = time.perf_counter()
            process = psutil.Process(os.getpid())
            mem_start = process.memory_info().rss
            
            try:
                return func(*args, **kwargs)
            finally:
                if synchronize_cuda and torch.cuda.is_available():
                    torch.cuda.synchronize()
                    max_gpu_mem = torch.cuda.max_memory_allocated() / (1024 ** 2)
                else:
                    max_gpu_mem = 0.0
                
                elapsed = time.perf_counter() - start
                mem_end = process.memory_info().rss
                mem_diff_mb = (mem_end - mem_start) / (1024 ** 2)
                
                print(f"[PROFILER] {label} | Time: {elapsed:.4f} sec | Peak GPU Mem: {max_gpu_mem:.2f} MB | CPU Mem Diff: {mem_diff_mb:.2f} MB")
        return wrapper
    return decorator


class PrecomputedDefectDataset(Dataset):
    @profiled("Dataset_Initialization")
    def __init__(self, normal_dir, defect_dir, img_size=512):
        self.normal_dir = normal_dir
        self.defect_paths = sorted(glob.glob(os.path.join(defect_dir, "*.*")))
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3),
        ])
    
    def __len__(self):
        return len(self.defect_paths)
    
    def __getitem__(self, idx):
        defect_path = self.defect_paths[idx]
        defect_filename = os.path.basename(defect_path)
        normal_filename = re.sub(r'_var\d+', '', defect_filename)
        normal_path = os.path.join(self.normal_dir, normal_filename)
        img_L = Image.open(normal_path).convert('RGB')
        img_D = Image.open(defect_path).convert('RGB')
        L_gt = self.transform(img_L)
        D = self.transform(img_D)
        diff = torch.abs(D - L_gt)
        M = (diff.max(dim=0, keepdim=True)[0] > 0.05).float()
        S_gt = D - L_gt
        return {'L_gt': L_gt, 'D': D, 'S_gt': S_gt, 'M': M}

class DualBranchUNet(nn.Module):
    def __init__(self, pretrained_model_name="stable-diffusion-v1-5/stable-diffusion-v1-5"):
        super().__init__()
        self.unet = UNet2DConditionModel.from_pretrained(pretrained_model_name, subfolder="unet")
        old_conv = self.unet.conv_in
        self.unet.conv_in = nn.Conv2d(8, old_conv.out_channels, old_conv.kernel_size, old_conv.stride, old_conv.padding)
        with torch.no_grad():
            self.unet.conv_in.weight[:, :4, :, :] = old_conv.weight
            self.unet.conv_in.weight[:, 4:, :, :] = old_conv.weight * 0.01
            self.unet.conv_in.bias.copy_(old_conv.bias)
        out_channels = self.unet.conv_out.in_channels
        self.l_branch = nn.Sequential(nn.Conv2d(out_channels, out_channels, 3, padding=1), nn.SiLU(), nn.Conv2d(out_channels, 4, 3, padding=1))
        self.s_branch = nn.Sequential(nn.Conv2d(out_channels, out_channels, 3, padding=1), nn.SiLU(), nn.Conv2d(out_channels, 4, 3, padding=1))
    
    def forward(self, x_t, z_D, timestep, encoder_hidden_states):
        unet_input = torch.cat([x_t, z_D], dim=1)
        sample = self.unet.conv_in(unet_input)
        t_emb = self.unet.time_proj(timestep)
        t_emb = t_emb.to(dtype=sample.dtype)
        emb = self.unet.time_embedding(t_emb)
        down_block_res_samples = (sample,)
        for downsample_block in self.unet.down_blocks:
            if hasattr(downsample_block, "has_cross_attention") and downsample_block.has_cross_attention:
                sample, res_samples = downsample_block(hidden_states=sample, temb=emb, encoder_hidden_states=encoder_hidden_states)
            else:
                sample, res_samples = downsample_block(hidden_states=sample, temb=emb)
            down_block_res_samples += res_samples
        if self.unet.mid_block is not None:
            if hasattr(self.unet.mid_block, "has_cross_attention") and self.unet.mid_block.has_cross_attention:
                sample = self.unet.mid_block(sample, emb, encoder_hidden_states=encoder_hidden_states)
            else:
                sample = self.unet.mid_block(sample, emb)
        for i, upsample_block in enumerate(self.unet.up_blocks):
            res_samples = down_block_res_samples[-len(upsample_block.resnets):]
            down_block_res_samples = down_block_res_samples[:-len(upsample_block.resnets)]
            if hasattr(upsample_block, "has_cross_attention") and upsample_block.has_cross_attention:
                sample = upsample_block(hidden_states=sample, temb=emb, res_hidden_states_tuple=res_samples, encoder_hidden_states=encoder_hidden_states)
            else:
                sample = upsample_block(hidden_states=sample, temb=emb, res_hidden_states_tuple=res_samples)
        sample = self.unet.conv_norm_out(sample)
        sample = self.unet.conv_act(sample)
        L_hat = self.l_branch(sample)
        S_hat = self.s_branch(sample)
        return L_hat, S_hat

class ModifiedDiffusionLoss(nn.Module):
    def __init__(self, device, w_defect=10.0, w_ssim=1.0):
        super().__init__()
        self.w_defect = w_defect
        self.w_ssim = w_ssim
            
    def forward(self, L_hat, S_hat, z_L_gt, z_S_gt, L_hat_decoded, L_gt, M_latent):
        base_latent_mse = F.mse_loss(L_hat, z_L_gt, reduction='none')
        loss_L_mse = base_latent_mse.mean()
        if M_latent.sum() > 0:
            defect_penalty = (base_latent_mse * M_latent).sum() / (M_latent.sum() + 1e-8)
        else:
            defect_penalty = torch.tensor(0.0, device=L_hat.device)
        L_hat_decoded_01 = (L_hat_decoded + 1.0) / 2.0
        L_gt_01 = (L_gt + 1.0) / 2.0
        loss_L_ssim = 1 - ssim(L_hat_decoded_01, L_gt_01, data_range=1.0, size_average=True, win_size=5)
        loss_L = loss_L_mse + self.w_defect * defect_penalty + self.w_ssim * loss_L_ssim
        loss_S = F.mse_loss(S_hat, z_S_gt)
        loss_total = loss_L + loss_S
        return {'total': loss_total, 'L_mse': loss_L_mse, 'L_defect': defect_penalty, 'L_ssim': loss_L_ssim, 'L': loss_L, 'S': loss_S}

@torch.no_grad()
def inference(model, vae, noise_scheduler, defect_image_path, config, device):
    model.eval()
    transform = transforms.Compose([
        transforms.Resize((config['img_size'], config['img_size'])),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3),
    ])
    img = Image.open(defect_image_path).convert('RGB')
    D = transform(img).unsqueeze(0).to(device) 
    z_D = vae.encode(D).latent_dist.mode() * vae.config.scaling_factor
    T = noise_scheduler.config.num_train_timesteps
    t_start = T // 4  
    noise = torch.randn_like(z_D)
    t_tensor = torch.tensor([t_start], device=device).long()
    x_t = noise_scheduler.add_noise(z_D, noise, t_tensor)
    text_emb = torch.zeros(1, 77, 768).to(device)
    S_accumulated = torch.zeros_like(z_D)
    num_steps = 0
    noise_scheduler.set_timesteps(T)
    timesteps = [t for t in noise_scheduler.timesteps if t <= t_start]
    
    for t in timesteps: # Removed tqdm inside inference for cleaner bench log
        t_batch = torch.tensor([t], device=device).long()
        L_hat, S_hat = model(x_t, z_D, t_batch, text_emb)
        S_accumulated += S_hat
        num_steps += 1
        alpha_prod_t = noise_scheduler.alphas_cumprod[t]
        predicted_noise = (x_t - torch.sqrt(alpha_prod_t) * L_hat) / torch.sqrt(1 - alpha_prod_t)
        scheduler_output = noise_scheduler.step(predicted_noise, t, x_t)
        x_t = scheduler_output.prev_sample
    
    S_final_latent = S_accumulated / num_steps 
    L_final_latent = L_hat 
    L_final = vae.decode(L_final_latent / vae.config.scaling_factor).sample
    L_final = (L_final.clamp(-1, 1) + 1) / 2
    D_normalized = (D + 1) / 2
    S_final = torch.abs(D_normalized - L_final)
    return L_final.cpu(), S_final.cpu(), D_normalized.cpu()

@profiled("run_inference_on_folder")
@torch.no_grad()
def run_inference_on_folder(config, model, vae, noise_scheduler, device):
    image_paths = []
    for ext in ['*.png', '*.jpg', '*.jpeg', '*.bmp']:
        image_paths.extend(glob.glob(os.path.join(config['defect_image_folder'], ext)))
    
    os.makedirs(config['inference_output_dir'], exist_ok=True)
    print(f"{len(image_paths)}개의 결함 이미지에 대해 추론을 시작합니다.")
    for img_path in tqdm(image_paths):
        filename = os.path.basename(img_path)
        L_final, S_final, D_orig = inference(model, vae, noise_scheduler, img_path, config, device)
        
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        axes[0].imshow(D_orig[0].permute(1, 2, 0).numpy())
        axes[0].set_title('1. Defect Image (D)')
        axes[0].axis('off')
        axes[1].imshow(L_final[0].permute(1, 2, 0).numpy())
        axes[1].set_title('2. Restored Background (L)')
        axes[1].axis('off')
        S_vis = S_final[0].mean(dim=0).numpy()
        axes[2].imshow(S_vis, cmap='hot')
        axes[2].set_title('3. Defect Map (S)')
        axes[2].axis('off')
        axes[3].imshow(D_orig[0].permute(1, 2, 0).numpy())
        axes[3].imshow(S_vis, cmap='hot', alpha=0.5)
        axes[3].set_title('4. Overlay')
        axes[3].axis('off')
        plt.tight_layout()
        save_path = os.path.join(config['inference_output_dir'], f"result_{filename}")
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    print("추론 완료!")

@profiled("total_training")
def train(config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    vae = AutoencoderKL.from_pretrained(config['pretrained_model'], subfolder="vae").to(device)
    vae.eval()
    for param in vae.parameters():
        param.requires_grad = False
    noise_scheduler = DDPMScheduler.from_pretrained(config['pretrained_model'], subfolder="scheduler")
    model = DualBranchUNet(pretrained_model_name=config['pretrained_model']).to(device)
    empty_text_embedding = torch.zeros(1, 77, 768).to(device)
    
    dataset = PrecomputedDefectDataset(
        normal_dir=config['normal_image_folder'], 
        defect_dir=config['defect_synthetic_folder'], 
        img_size=config['img_size'],
    )
    dataloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True, num_workers=0, pin_memory=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=1e-4)
    criterion = ModifiedDiffusionLoss(device=device, w_defect=config.get('w_defect', 10.0), w_ssim=config.get('w_ssim', 1.0)).to(device)
    
    sample_output_dir = os.path.join(config['output_dir'], "training_samples")
    os.makedirs(sample_output_dir, exist_ok=True)
    sample_defect_paths = glob.glob(os.path.join(config['defect_image_folder'], '*.*'))
    sample_img_path = sample_defect_paths[0] if sample_defect_paths else None

    model.train()
    
    for epoch in range(config['num_epochs']):
        epoch_losses = {'total': 0, 'L_mse': 0, 'L_defect': 0, 'L_ssim': 0, 'L': 0, 'S': 0}
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{config['num_epochs']}")
        
        for batch in pbar:
            L_gt = batch['L_gt'].to(device)
            D = batch['D'].to(device)
            M = batch['M'].to(device)
            B = L_gt.shape[0]
            
            with torch.no_grad():
                z_L_gt = vae.encode(L_gt).latent_dist.mode() * vae.config.scaling_factor
                z_D = vae.encode(D).latent_dist.mode() * vae.config.scaling_factor
                M_latent = F.interpolate(M, size=(z_L_gt.shape[2], z_L_gt.shape[3]), mode='nearest')
                z_S_gt = (z_D - z_L_gt) * M_latent
            
            t = torch.randint(0, noise_scheduler.config.num_train_timesteps, (B,), device=device).long()
            noise = torch.randn_like(z_L_gt)
            x_t = noise_scheduler.add_noise(z_L_gt, noise, t)
            text_emb = empty_text_embedding.expand(B, -1, -1)
            L_hat, S_hat = model(x_t, z_D, t, text_emb)
            
            L_hat_decoded = vae.decode(L_hat / vae.config.scaling_factor).sample
            losses = criterion(L_hat, S_hat, z_L_gt, z_S_gt, L_hat_decoded, L_gt, M_latent)
            
            optimizer.zero_grad()
            losses['total'].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            for key in epoch_losses:
                if key in losses:
                    epoch_losses[key] += losses[key].item()
            
            pbar.set_postfix({'L': f"{losses['L'].item():.4f}", 'S': f"{losses['S'].item():.4f}"})
        
        num_batches = len(dataloader)
        print(f"\nEpoch {epoch+1} 평균 손실:")
        for key in epoch_losses:
            print(f"  {key}: {epoch_losses[key] / num_batches:.6f}")
            
        if (epoch + 1) % config['visual_every'] == 0 and sample_img_path is not None:
            model.eval()
            L_final, S_final, D_orig = inference(model, vae, noise_scheduler, sample_img_path, config, device)
            import matplotlib.pyplot as plt
            fig, axes = plt.subplots(1, 4, figsize=(20, 5))
            axes[0].imshow(D_orig[0].permute(1, 2, 0).numpy())
            axes[1].imshow(L_final[0].permute(1, 2, 0).numpy())
            S_vis = S_final[0].mean(dim=0).numpy()
            axes[2].imshow(S_vis, cmap='hot')
            axes[3].imshow(D_orig[0].permute(1, 2, 0).numpy())
            axes[3].imshow(S_vis, cmap='hot', alpha=0.5)
            plt.tight_layout()
            sample_save_path = os.path.join(sample_output_dir, f"sample_epoch_{epoch+1:03d}.png")
            plt.savefig(sample_save_path, dpi=150, bbox_inches='tight')
            plt.close()
            model.train()
            
        if (epoch + 1) % config['save_every'] == 0:
            save_path = os.path.join(config['output_dir'], f"checkpoint_epoch{epoch+1}.pt")
            torch.save({'epoch': epoch + 1, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'config': config}, save_path)
    
    final_path = os.path.join(config['output_dir'], "model_final.pt")
    torch.save({'model_state_dict': model.state_dict(), 'config': config}, final_path)
    return model, vae, noise_scheduler, device

@profiled("total_execution_script")
def run_benchmark(base_config, bs, run_inference=False):
    config = base_config.copy()
    config['batch_size'] = bs
    print(f"\n" + "="*60)
    print(f"[BASELINE] 배치 크기 {bs} 로 성능 측정 시작")
    print("="*60)
    model, vae, noise_scheduler, device = train(config)
    
    if run_inference:
        run_inference_on_folder(config, model, vae, noise_scheduler, device)
    else:
        print("\n[INFO] 추론 단계가 생략되었습니다. 실행하려면 --inference 플래그를 추가하세요.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--inference", action="store_true", help="Run folder inference after training")
    args = parser.parse_args()
    
    base_config = {
        'pretrained_model': 'stable-diffusion-v1-5/stable-diffusion-v1-5',
        'normal_image_folder': './sample_dataset/normal',
        'defect_image_folder': './sample_dataset/defect',
        'defect_synthetic_folder': './sample_dataset/synthetic',
        'output_dir': '/ssd1/cwoo/DFGNet/pythonsubject_baseline',
        'inference_output_dir': '/ssd1/cwoo/DFGNet/pythonsubject_baseline_inference',
        'img_size': 512,
        'batch_size': 2,
        'num_epochs': 1,     # 측정용으로 1 설정
        'learning_rate': 1e-5,
        'save_every': 5,
        'visual_every': 1,   # 1로 설정하여 무조건 1번 추론하도록
        'w_defect': 10.0,
        'w_ssim': 1.0,
    }
    
    os.makedirs(base_config['output_dir'], exist_ok=True)
    os.makedirs(base_config['inference_output_dir'], exist_ok=True)
    
    # 벤치마크 (배치 1, 2)
    for bs in [1, 2]:
        run_benchmark(base_config, bs, run_inference=args.inference)
