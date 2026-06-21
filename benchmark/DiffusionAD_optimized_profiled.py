import os
import glob
import time
import re
import psutil
from functools import wraps
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

try:
    from pytorch_msssim import ssim
except ImportError:
    pass

# ============================================================
# 5.2.4 Decorator 기반 실행 시간 및 메모리 측정 로직 분리
# ============================================================
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

# ============================================================
# 5.2.3 Class 기반 구조 개선을 위한 Config 및 State
# ============================================================
@dataclass(frozen=True, slots=True)
class TrainConfig:
    pretrained_model: str
    normal_image_folder: str
    defect_image_folder: str
    defect_synthetic_folder: str
    output_dir: str
    inference_output_dir: str
    img_size: int = 512
    batch_size: int = 2
    num_epochs: int = 100
    learning_rate: float = 1e-5
    weight_decay: float = 1e-4
    save_every: int = 5
    visual_every: int = 5
    w_defect: float = 10.0
    w_ssim: float = 1.0
    num_workers: int = 4
    use_amp: bool = True
    seed: int = 42

@dataclass(slots=True)
class TrainerState:
    epoch: int = 0
    global_step: int = 0
    epoch_losses: Dict[str, float] = field(default_factory=dict)
    
    def reset_epoch_losses(self):
        self.epoch_losses = {
            'total': 0.0, 'L_mse': 0.0, 'L_defect': 0.0,
            'L_ssim': 0.0, 'L': 0.0, 'S': 0.0,
        }

# ============================================================
# 5.2.1 자료구조 및 복잡도 기반 개선 (Dataset)
# ============================================================
class PrecomputedDefectDataset(Dataset):
    VALID_EXTS = {".png", ".jpg", ".jpeg", ".bmp"}
    
    @profiled("Dataset_Initialization")
    def __init__(self, normal_dir, defect_dir, img_size=512):
        self.normal_dir = Path(normal_dir)
        self.defect_dir = Path(defect_dir)
        self.variant_pattern = re.compile(r"_var\d+")
        self.normal_index = self._build_normal_index()
        self.pairs = self._build_pairs()
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3),
        ])

    def _is_image_file(self, path: Path) -> bool:
        return path.suffix.lower() in self.VALID_EXTS

    @profiled("Dataset._build_normal_index")
    def _build_normal_index(self) -> dict[str, Path]:
        normal_index: dict[str, Path] = {}
        for normal_path in sorted(self.normal_dir.iterdir()):
            if not self._is_image_file(normal_path): continue
            normal_index[normal_path.name] = normal_path
        return normal_index

    @profiled("Dataset._build_pairs")
    def _build_pairs(self) -> list[tuple[Path, Path]]:
        pairs: list[tuple[Path, Path]] = []
        for defect_path in sorted(self.defect_dir.iterdir()):
            if not self._is_image_file(defect_path): continue
            normal_filename = self.variant_pattern.sub("", defect_path.name)
            try:
                normal_path = self.normal_index[normal_filename]
            except KeyError as e:
                pass # skip
            pairs.append((normal_path, defect_path))
        return pairs

    def __len__(self): return len(self.pairs)

    def __getitem__(self, idx):
        normal_path, defect_path = self.pairs[idx]
        img_L = Image.open(normal_path).convert("RGB")
        img_D = Image.open(defect_path).convert("RGB")
        L_gt = self.transform(img_L)
        D = self.transform(img_D)
        diff = torch.abs(D - L_gt)
        M = (diff.max(dim=0, keepdim=True)[0] > 0.05).float()
        return {"L_gt": L_gt, "D": D, "M": M}

# ============================================================
# Dual-Branch U-Net 및 Loss
# ============================================================
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

class CheckpointManager:
    def __init__(self, output_dir):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def save_checkpoint(self, epoch, model, optimizer, config):
        save_path = os.path.join(self.output_dir, f"checkpoint_epoch{epoch}.pt")
        torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'config': config}, save_path)
        print(f"체크포인트 저장: {save_path}")

    def save_final(self, model, config):
        final_path = os.path.join(self.output_dir, "model_final.pt")
        torch.save({'model_state_dict': model.state_dict(), 'config': config}, final_path)
        print(f"최종 모델 저장: {final_path}")

class InferenceVisualizer:
    def __init__(self, output_dir):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def save_result(self, D_orig, L_final, S_final, save_path, title_prefix=""):
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        axes[0].imshow(D_orig[0].permute(1, 2, 0).numpy())
        axes[1].imshow(L_final[0].permute(1, 2, 0).numpy())
        S_vis = S_final[0].mean(dim=0).numpy()
        axes[2].imshow(S_vis, cmap='hot')
        axes[3].imshow(D_orig[0].permute(1, 2, 0).numpy())
        axes[3].imshow(S_vis, cmap='hot', alpha=0.5)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

# ============================================================
# 추론용 Lazy Evaluation Generator (5.2.2)
# ============================================================
def iter_image_paths(folder, patterns=("*.png", "*.jpg", "*.jpeg", "*.bmp")):
    for pattern in patterns:
        yield from glob.iglob(os.path.join(folder, pattern))

@torch.inference_mode()
def inference(model, vae, noise_scheduler, img_path, config, device):
    transform = transforms.Compose([transforms.Resize((config.img_size, config.img_size)), transforms.ToTensor(), transforms.Normalize([0.5]*3, [0.5]*3)])
    img = Image.open(img_path).convert('RGB')
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
    timesteps = (t for t in noise_scheduler.timesteps if t <= t_start)
    for t in timesteps:
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
@torch.inference_mode()
def run_inference_on_folder(config, model, vae, noise_scheduler, device):
    os.makedirs(config.inference_output_dir, exist_ok=True)
    image_iter = iter_image_paths(config.defect_image_folder)
    print("결함 이미지 폴더에 대해 lazy inference를 시작합니다.")
    for img_path in tqdm(list(image_iter), desc="Folder inference"): 
        filename = os.path.basename(img_path)
        L_final, S_final, D_orig = inference(model, vae, noise_scheduler, img_path, config, device)
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        axes[0].imshow(D_orig[0].permute(1, 2, 0).numpy())
        axes[1].imshow(L_final[0].permute(1, 2, 0).numpy())
        S_vis = S_final[0].mean(dim=0).numpy()
        axes[2].imshow(S_vis, cmap='hot')
        axes[3].imshow(D_orig[0].permute(1, 2, 0).numpy())
        axes[3].imshow(S_vis, cmap='hot', alpha=0.5)
        plt.tight_layout()
        save_path = os.path.join(config.inference_output_dir, f"result_{filename}")
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    print("추론 완료!")

# ============================================================
# 5.2.3 Trainer 구조 개선 & 5.2.5 딥러닝 최적화 (AMP, Dataloader)
# ============================================================
class DefectTrainer:
    def __init__(self, config: TrainConfig):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.state = TrainerState()
        self.checkpoint_manager = CheckpointManager(config.output_dir)
        self.visualizer = InferenceVisualizer(os.path.join(config.output_dir, "training_samples"))
        self.use_amp = self.config.use_amp and self.device.type == "cuda"
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)
        self._build_components()

    def _build_components(self):
        self.vae = AutoencoderKL.from_pretrained(self.config.pretrained_model, subfolder="vae").to(self.device)
        self.vae.eval()
        self.vae.requires_grad_(False)
        self.noise_scheduler = DDPMScheduler.from_pretrained(self.config.pretrained_model, subfolder="scheduler")
        self.model = DualBranchUNet(pretrained_model_name=self.config.pretrained_model).to(self.device)
        self.dataset = PrecomputedDefectDataset(normal_dir=self.config.normal_image_folder, defect_dir=self.config.defect_synthetic_folder, img_size=self.config.img_size)
        self.dataloader = DataLoader(self.dataset, batch_size=self.config.batch_size, shuffle=True, num_workers=self.config.num_workers, pin_memory=torch.cuda.is_available(), persistent_workers=self.config.num_workers > 0)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.config.learning_rate, weight_decay=self.config.weight_decay)
        self.criterion = ModifiedDiffusionLoss(device=self.device, w_defect=self.config.w_defect, w_ssim=self.config.w_ssim).to(self.device)
        self.empty_text_embedding = torch.zeros(1, 77, 768, device=self.device)

    def train_step(self, batch):
        L_gt = batch['L_gt'].to(self.device, non_blocking=True)
        D = batch['D'].to(self.device, non_blocking=True)
        M = batch['M'].to(self.device, non_blocking=True)
        B = L_gt.shape[0]

        with torch.no_grad():
            z_L_gt = self.vae.encode(L_gt).latent_dist.mode() * self.vae.config.scaling_factor
            z_D = self.vae.encode(D).latent_dist.mode() * self.vae.config.scaling_factor
            M_latent = F.interpolate(M, size=(z_L_gt.shape[2], z_L_gt.shape[3]), mode='nearest')
            z_S_gt = (z_D - z_L_gt) * M_latent
            t = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (B,), device=self.device).long()
            noise = torch.randn_like(z_L_gt)
            x_t = self.noise_scheduler.add_noise(z_L_gt, noise, t)
            text_emb = self.empty_text_embedding.expand(B, -1, -1)

        self.optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=self.use_amp):
            L_hat, S_hat = self.model(x_t, z_D, t, text_emb)
            L_hat_decoded = self.vae.decode(L_hat / self.vae.config.scaling_factor).sample
            losses = self.criterion(L_hat, S_hat, z_L_gt, z_S_gt, L_hat_decoded, L_gt, M_latent)

        self.scaler.scale(losses['total']).backward()
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        return losses

    def train_one_epoch(self):
        self.state.reset_epoch_losses()
        pbar = tqdm(self.dataloader, desc=f"Epoch {self.state.epoch}/{self.config.num_epochs}")
        for batch in pbar:
            losses = self.train_step(batch)
            self.state.global_step += 1
            for key in self.state.epoch_losses:
                if key in losses:
                    self.state.epoch_losses[key] += losses[key].item()
            pbar.set_postfix({'L': f"{losses['L'].item():.4f}", 'S': f"{losses['S'].item():.4f}"})

    def run_visual_check(self, sample_img_path):
        self.model.eval()
        L_final, S_final, D_orig = inference(self.model, self.vae, self.noise_scheduler, sample_img_path, self.config, self.device)
        save_path = os.path.join(self.visualizer.output_dir, f"sample_epoch_{self.state.epoch:03d}.png")
        self.visualizer.save_result(D_orig, L_final, S_final, save_path, title_prefix=f"Epoch {self.state.epoch} ")
        self.model.train()

    @profiled("total_training")
    def train(self):
        self.model.train()
        sample_defect_paths = glob.glob(os.path.join(self.config.defect_image_folder, '*.*'))
        sample_img_path = sample_defect_paths[0] if sample_defect_paths else None
        
        for epoch in range(self.config.num_epochs):
            self.state.epoch = epoch + 1
            self.train_one_epoch()
            
            num_batches = len(self.dataloader)
            print(f"\nEpoch {self.state.epoch} 평균 손실:")
            for key in self.state.epoch_losses:
                print(f"  {key}: {self.state.epoch_losses[key] / num_batches:.6f}")

            if self.state.epoch % self.config.visual_every == 0 and sample_img_path:
                self.run_visual_check(sample_img_path)
            
            if self.state.epoch % self.config.save_every == 0:
                self.checkpoint_manager.save_checkpoint(epoch=self.state.epoch, model=self.model, optimizer=self.optimizer, config=self.config)
        self.checkpoint_manager.save_final(model=self.model, config=self.config)
        return self.model


@profiled("total_execution_script")
def run_benchmark(base_config_kwargs, bs, run_inference=False):
    print("\n" + "=" * 60)
    print(f"[OPTIMIZED] 배치 크기 {bs} 로 성능 측정 시작")
    print("=" * 60)
    
    config_kwargs = base_config_kwargs.copy()
    config_kwargs['batch_size'] = bs
    
    config = TrainConfig(**config_kwargs)
    trainer = DefectTrainer(config)
    trainer.train()
    
    if run_inference:
        run_inference_on_folder(config, trainer.model, trainer.vae, trainer.noise_scheduler, trainer.device)
    else:
        print("\n[INFO] 추론 단계가 생략되었습니다. 실행하려면 --inference 플래그를 추가하세요.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--inference", action="store_true", help="Run folder inference after training")
    args = parser.parse_args()
    
    base_kwargs = {
        'pretrained_model': 'stable-diffusion-v1-5/stable-diffusion-v1-5',
        'normal_image_folder': './sample_dataset/normal',
        'defect_image_folder': './sample_dataset/defect',
        'defect_synthetic_folder': './sample_dataset/synthetic',
        'output_dir': '/ssd1/cwoo/DFGNet/pythonsubject_optimized',
        'inference_output_dir': '/ssd1/cwoo/DFGNet/pythonsubject_optimized_inference',
        'img_size': 512,
        'learning_rate': 1e-5,
        'save_every': 5,
        'visual_every': 1,  # 1번 추론 포함
        'w_defect': 10.0,
        'w_ssim': 1.0,
        'num_workers': 4,
        'use_amp': True,
        'num_epochs': 1,    # 측정용으로 1 설정
    }
    
    for bs in [1, 2]:
        run_benchmark(base_kwargs, bs, run_inference=args.inference)
