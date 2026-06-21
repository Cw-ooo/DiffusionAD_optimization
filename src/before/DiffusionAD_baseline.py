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

# [NEW] 필요한 라이브러리 추가
try:
    from pytorch_msssim import ssim
except ImportError:
    print("WARNING: pytorch_msssim 패키지가 필요합니다.")
    print("설치 명령어: pip install pytorch-msssim")

# ============================================================
# 2. 데이터셋 (Precomputed Dataset)
# ============================================================

class PrecomputedDefectDataset(Dataset):
    """
    1:N(여러 변형)으로 생성된 합성 데이터셋을 로드하는 Dataset.
    합성 이미지 파일명(예: img_var1.jpg)에서 원본 파일명(img.jpg)을 추적하여 쌍을 맞춥니다.
    """
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
        
        return {
            'L_gt': L_gt,
            'D': D,
            'S_gt': S_gt,
            'M': M,
        }


# ============================================================
# 3. Dual-Branch U-Net (Stable Diffusion U-Net 기반)
# ============================================================

class DualBranchUNet(nn.Module):
    """
    Stable Diffusion의 U-Net을 기반으로 한 Dual-Branch 구조.
    """
    
    def __init__(self, pretrained_model_name="stable-diffusion-v1-5/stable-diffusion-v1-5"):
        super().__init__()
        
        self.unet = UNet2DConditionModel.from_pretrained(
            pretrained_model_name, subfolder="unet"
        )
        
        old_conv = self.unet.conv_in
        self.unet.conv_in = nn.Conv2d(
            in_channels=8,
            out_channels=old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
        )
        with torch.no_grad():
            self.unet.conv_in.weight[:, :4, :, :] = old_conv.weight
            self.unet.conv_in.weight[:, 4:, :, :] = old_conv.weight * 0.01
            self.unet.conv_in.bias.copy_(old_conv.bias)
        
        out_channels = self.unet.conv_out.in_channels
        
        self.l_branch = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(out_channels, 4, 3, padding=1),
        )
        
        self.s_branch = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(out_channels, 4, 3, padding=1),
        )
    
    def forward(self, x_t, z_D, timestep, encoder_hidden_states):
        unet_input = torch.cat([x_t, z_D], dim=1)
        
        sample = self.unet.conv_in(unet_input)
        
        t_emb = self.unet.time_proj(timestep)
        t_emb = t_emb.to(dtype=sample.dtype)
        emb = self.unet.time_embedding(t_emb)
        
        down_block_res_samples = (sample,)
        for downsample_block in self.unet.down_blocks:
            if hasattr(downsample_block, "has_cross_attention") and downsample_block.has_cross_attention:
                sample, res_samples = downsample_block(
                    hidden_states=sample,
                    temb=emb,
                    encoder_hidden_states=encoder_hidden_states,
                )
            else:
                sample, res_samples = downsample_block(
                    hidden_states=sample,
                    temb=emb,
                )
            down_block_res_samples += res_samples
        
        if self.unet.mid_block is not None:
            if hasattr(self.unet.mid_block, "has_cross_attention") and self.unet.mid_block.has_cross_attention:
                sample = self.unet.mid_block(
                    sample,
                    emb,
                    encoder_hidden_states=encoder_hidden_states,
                )
            else:
                sample = self.unet.mid_block(sample, emb)
        
        for i, upsample_block in enumerate(self.unet.up_blocks):
            res_samples = down_block_res_samples[-len(upsample_block.resnets):]
            down_block_res_samples = down_block_res_samples[:-len(upsample_block.resnets)]
            
            if hasattr(upsample_block, "has_cross_attention") and upsample_block.has_cross_attention:
                sample = upsample_block(
                    hidden_states=sample,
                    temb=emb,
                    res_hidden_states_tuple=res_samples,
                    encoder_hidden_states=encoder_hidden_states,
                )
            else:
                sample = upsample_block(
                    hidden_states=sample,
                    temb=emb,
                    res_hidden_states_tuple=res_samples,
                )
        
        sample = self.unet.conv_norm_out(sample)
        sample = self.unet.conv_act(sample)
        
        L_hat = self.l_branch(sample)
        S_hat = self.s_branch(sample)
        
        return L_hat, S_hat


# ============================================================
# 4. 학습 손실 함수 (수정됨)
# ============================================================

class ModifiedDiffusionLoss(nn.Module):
    """
    최종 수정된 손실 함수:
    - L 복원 손실: 전역 MSE(Latent) + 결함 영역 페널티(Latent) + 단일 해상도 SSIM(Pixel, win_size=5)
    - S 복원 손실: 단순 MSE(Latent) (Smooth L1 제거)
    - 희소성(Sparsity) 및 일관성(Consistency) 손실 제거
    """
    def __init__(self, device, w_defect=10.0, w_ssim=1.0):
        super().__init__()
        self.w_defect = w_defect
        self.w_ssim = w_ssim
            
    def forward(self, L_hat, S_hat, z_L_gt, z_S_gt, L_hat_decoded, L_gt, M_latent):
        # ==========================================
        # 1. L 복원 손실 (배경 정상화)
        # ==========================================
        # 1-1. 전역 잠재 공간 MSE
        base_latent_mse = F.mse_loss(L_hat, z_L_gt, reduction='none')
        loss_L_mse = base_latent_mse.mean()
        
        # 1-2. 미세 결함 영역 집중 페널티 (Latent Space)
        # 결함이 존재하는 영역(M_latent == 1)에서의 오차만 크게 증폭시켜 페널티 부여
        if M_latent.sum() > 0:
            defect_penalty = (base_latent_mse * M_latent).sum() / (M_latent.sum() + 1e-8)
        else:
            defect_penalty = torch.tensor(0.0, device=L_hat.device)
            
        # 1-3. 단일 해상도 SSIM 손실 (Pixel Space)
        # 미세 결함을 더 잘 찾을 수 있도록 win_size를 11에서 5로 축소
        L_hat_decoded_01 = (L_hat_decoded + 1.0) / 2.0
        L_gt_01 = (L_gt + 1.0) / 2.0
        loss_L_ssim = 1 - ssim(L_hat_decoded_01, L_gt_01, data_range=1.0, size_average=True, win_size=5)
        
        # 최종 L 복원 손실 조합
        loss_L = loss_L_mse + self.w_defect * defect_penalty + self.w_ssim * loss_L_ssim
        
        # ==========================================
        # 2. S 복원 손실 (결함)
        # ==========================================
        # L1 Smooth를 제거하고 단순 MSE만 사용
        loss_S = F.mse_loss(S_hat, z_S_gt)
        
        # ==========================================
        # 3. 전체 손실
        # ==========================================
        loss_total = loss_L + loss_S
        
        return {
            'total': loss_total,
            'L_mse': loss_L_mse,
            'L_defect': defect_penalty,
            'L_ssim': loss_L_ssim,
            'L': loss_L,
            'S': loss_S,
        }


# ============================================================
# 5. 학습 루프
# ============================================================

def train(config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    vae = AutoencoderKL.from_pretrained(
        config['pretrained_model'], subfolder="vae"
    ).to(device)
    vae.eval()
    for param in vae.parameters():
        param.requires_grad = False
    
    noise_scheduler = DDPMScheduler.from_pretrained(
        config['pretrained_model'], subfolder="scheduler"
    )
    
    model = DualBranchUNet(
        pretrained_model_name=config['pretrained_model']
    ).to(device)
    
    empty_text_embedding = torch.zeros(1, 77, 768).to(device)
    
    dataset = PrecomputedDefectDataset(
        normal_dir=config['normal_image_folder'], 
        defect_dir=config['defect_synthetic_folder'], 
        img_size=config['img_size'],
    )
    dataloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True, num_workers=0, pin_memory=True)
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=1e-4,
    )
    
    # [수정] 최종 손실 함수 적용
    criterion = ModifiedDiffusionLoss(
        device=device,
        w_defect=config.get('w_defect', 10.0),
        w_ssim=config.get('w_ssim', 1.0)
    ).to(device)
    
    sample_output_dir = os.path.join(config['output_dir'], "training_samples")
    os.makedirs(sample_output_dir, exist_ok=True)
    
    sample_defect_paths = glob.glob(os.path.join(config['defect_image_folder'], '*.*'))
    sample_img_path = sample_defect_paths[0] if sample_defect_paths else None
    if sample_img_path:
        print(f"학습 중 모니터링 샘플: {os.path.basename(sample_img_path)}")


    print(f"학습 시작: {len(dataset)}개의 정상 이미지")
    print(f"설정: w_defect={config.get('w_defect', 10.0)}, w_ssim={config.get('w_ssim', 1.0)}")
    
    model.train()
    global_step = 0
    
    for epoch in range(config['num_epochs']):
        # 손실 로깅 딕셔너리 수정
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
                
                # 마스크를 곱해 S_gt 생성
                z_S_gt = (z_D - z_L_gt) * M_latent
            
            t = torch.randint(
                0, noise_scheduler.config.num_train_timesteps,
                (B,), device=device
            ).long()
            
            noise = torch.randn_like(z_L_gt)
            x_t = noise_scheduler.add_noise(z_L_gt, noise, t)
            
            text_emb = empty_text_embedding.expand(B, -1, -1)
            L_hat, S_hat = model(x_t, z_D, t, text_emb)
            
            # SSIM 계산을 위해 L_hat을 이미지 공간으로 디코딩
            L_hat_decoded = vae.decode(L_hat / vae.config.scaling_factor).sample
            
            # [수정] 파라미터에 M_latent 추가 전달
            losses = criterion(L_hat, S_hat, z_L_gt, z_S_gt, L_hat_decoded, L_gt, M_latent)
            
            optimizer.zero_grad()
            losses['total'].backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            global_step += 1
            for key in epoch_losses:
                if key in losses:
                    epoch_losses[key] += losses[key].item()
            
            pbar.set_postfix({
                'L': f"{losses['L'].item():.4f}",
                'S': f"{losses['S'].item():.4f}",
                'def': f"{losses['L_defect'].item():.4f}",
                'ssim': f"{losses['L_ssim'].item():.4f}",
            })
        
        num_batches = len(dataloader)
        print(f"\nEpoch {epoch+1} 평균 손실:")
        for key in epoch_losses:
            print(f"  {key}: {epoch_losses[key] / num_batches:.6f}")
            
        if (epoch + 1) % 5 == 0 and sample_img_path is not None:
            print(f"\n[Visual Check] 에폭 {epoch+1} 샘플 추론 결과 생성 중...")
            model.eval()
            
            L_final, S_final, D_orig = inference(
                model, vae, noise_scheduler, sample_img_path, config, device
            )
            
            import matplotlib.pyplot as plt
            fig, axes = plt.subplots(1, 4, figsize=(20, 5))
            
            axes[0].imshow(D_orig[0].permute(1, 2, 0).numpy())
            axes[0].set_title(f'Defect Image (Epoch {epoch+1})')
            axes[0].axis('off')
            
            axes[1].imshow(L_final[0].permute(1, 2, 0).numpy())
            axes[1].set_title('Restored Background')
            axes[1].axis('off')
            
            S_vis = S_final[0].mean(dim=0).numpy()
            axes[2].imshow(S_vis, cmap='hot')
            axes[2].set_title('Defect Map')
            axes[2].axis('off')
            
            axes[3].imshow(D_orig[0].permute(1, 2, 0).numpy())
            axes[3].imshow(S_vis, cmap='hot', alpha=0.5)
            axes[3].set_title('Overlay')
            axes[3].axis('off')
            
            plt.tight_layout()
            sample_save_path = os.path.join(sample_output_dir, f"sample_epoch_{epoch+1:03d}.png")
            plt.savefig(sample_save_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"샘플 이미지 저장 완료: {sample_save_path}\n")
            
            model.train()
            
        if (epoch + 1) % config['save_every'] == 0:
            save_path = os.path.join(config['output_dir'], f"checkpoint_epoch{epoch+1}.pt")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': config,
            }, save_path)
            print(f"체크포인트 저장: {save_path}")
    
    final_path = os.path.join(config['output_dir'], "model_final.pt")
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
    }, final_path)
    print(f"학습 완료. 최종 모델 저장: {final_path}")
    
    return model


# ============================================================
# 6. 추론 (결함 검출)
# ============================================================

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
    
    for t in tqdm(timesteps, desc="Reverse diffusion"):
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


@torch.no_grad()
def run_inference_on_folder(config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    vae = AutoencoderKL.from_pretrained(
        config['pretrained_model'], subfolder="vae"
    ).to(device)
    vae.eval()
    
    noise_scheduler = DDPMScheduler.from_pretrained(
        config['pretrained_model'], subfolder="scheduler"
    )
    
    model = DualBranchUNet(
        pretrained_model_name=config['pretrained_model']
    ).to(device)
    
    checkpoint = torch.load(config['checkpoint_path'], map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    image_paths = []
    for ext in ['*.png', '*.jpg', '*.jpeg', '*.bmp']:
        image_paths.extend(glob.glob(os.path.join(config['defect_image_folder'], ext)))
    
    os.makedirs(config['inference_output_dir'], exist_ok=True)
    
    print(f"{len(image_paths)}개의 결함 이미지에 대해 추론을 시작합니다.")
    
    for img_path in tqdm(image_paths):
        filename = os.path.basename(img_path)
        
        L_final, S_final, D_orig = inference(
            model, vae, noise_scheduler, img_path, config, device
        )
        
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


# ============================================================
# 7. 실행
# ============================================================

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--inference", action="store_true", help="Run folder inference after training")
    args = parser.parse_args()
    
    config = {
        # 모델
        'pretrained_model': 'stable-diffusion-v1-5/stable-diffusion-v1-5',
        
        # 데이터 경로
        'normal_image_folder': './sample_dataset/normal',
        'defect_image_folder': './sample_dataset/defect',
        'defect_synthetic_folder': './sample_dataset/synthetic',
        'output_dir': '/ssd1/cwoo/DFGNet/pythonsubject1',
        'inference_output_dir': '/ssd1/cwoo/DFGNet/pythonsubject1',
        
        # 학습 설정
        'img_size': 512,
        'batch_size': 2,           
        'num_epochs': 0,
        'learning_rate': 1e-5,
        'save_every': 5,
        
        # 손실 가중치 (수정됨)
        'w_defect': 10.0,  # 미세 결함에 부여할 10배 페널티
        'w_ssim': 1.0,     # SSIM (win_size=5) 가중치
    }
    
    os.makedirs(config['output_dir'], exist_ok=True)
    os.makedirs(config['inference_output_dir'], exist_ok=True)
    
    print("=" * 60)
    print("최종 수정된 손실 함수 기반 학습 시작")
    print("=" * 60)
    model = train(config)
    
    config['checkpoint_path'] = os.path.join(config['output_dir'], 'model_final.pt')
    
    if getattr(args, 'inference', False):
        print("\n" + "=" * 60)
        print("결함 검출 추론 시작")
        print("=" * 60)
        run_inference_on_folder(config)
    else:
        print("\n[INFO] 추론 단계가 생략되었습니다. 실행하려면 --inference 플래그를 추가하세요.")
