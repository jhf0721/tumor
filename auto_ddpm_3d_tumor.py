
import os
import shutil
import tempfile
import time
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from monai import transforms
from monai.apps import DecathlonDataset
from monai.config import print_config
from monai.data import DataLoader
from monai.utils import first, set_determinism
from torch.cuda.amp import GradScaler, autocast
from torch.nn import L1Loss
from tqdm import tqdm
import sys

from generative.inferers import LatentDiffusionInferer
from generative.losses import PatchAdversarialLoss, PerceptualLoss
from generative.networks.nets import AutoencoderKL, DiffusionModelUNet, PatchDiscriminator
from generative.networks.schedulers import DDPMScheduler
import nibabel as nib
print_config()

set_determinism(42)

def clear_directory(directory_path):
    # 检查目录是否存在
    if os.path.exists(directory_path):
        # 遍历文件夹中的所有文件和文件夹
        for filename in os.listdir(directory_path):
            file_path = os.path.join(directory_path, filename)
            try:
                # 如果是文件或链接，则删除
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                # 如果是目录，则递归删除
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f'Failed to delete {file_path}. Reason: {e}')

def ensure_empty_directory(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
    else:
        clear_directory(directory_path)

directory_path = '/Address where the model is stored'

root_dir="/Address of the lits dataset"
print(root_dir)
batch_size = 2
channel = 0  # 0 = Flair
def load_nifti_file(file_path):
    # 读取.nii或.nii.gz文件
    img_ct = nib.load(file_path)
    # 获取数据
    data = img_ct.get_fdata()
    return data
def create_dataset(folder_path,roi_size,spatial_size):
    # 创建数据集列表
    dataset = []
    # 遍历文件夹中的所有文件
    file_paths = [os.path.join(folder_path, file_name) for file_name in os.listdir(folder_path) if file_name.endswith(".nii") or file_name.endswith(".nii.gz")]
    for file_path in tqdm(file_paths,desc="Loading Nifti files"):
        # 加载NIfTI文件
        image_data = torch.from_numpy(load_nifti_file(file_path)).float()

        transformed_image =  F.interpolate(image_data.unsqueeze(0).unsqueeze(0), size=spatial_size, mode='trilinear', align_corners=False)
        transformed_image = transformed_image.squeeze(0)
        # 将图像添加到数据集
        dataset.append(transformed_image)
    return dataset
# 文件路径列表
file_paths = os.path.join(root_dir, "tumor_reshape")
# 定义新的图像形状，例如：(128, 128, 128)
new_shape = (96,96,64)
# 创建数据集
train_data = create_dataset(file_paths,roi_size=(400, 400, 300),spatial_size=new_shape)
val_size = 0.1
val_idx = int(len(train_data) * (1 - val_size))
train_data, val_data = torch.utils.data.random_split(train_data, [val_idx, len(train_data) - val_idx])
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4, persistent_workers=True)
val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=4, persistent_workers=True)
check_data = first(train_loader)
idx = 0

img = check_data[idx, 0]
fig, axs = plt.subplots(nrows=1, ncols=3)
for ax in axs:
    ax.axis("off")
ax = axs[0]
ax.imshow(img[..., img.shape[2] // 2], cmap="gray")
ax = axs[1]
ax.imshow(img[:, img.shape[1] // 2, ...], cmap="gray")
ax = axs[2]
ax.imshow(img[img.shape[0] // 2, ...], cmap="gray")
plt.show()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device}")

# +
autoencoder = AutoencoderKL(
    spatial_dims=3,
    in_channels=1,
    out_channels=1,
    num_channels=(32, 64, 64),
    latent_channels=3,
    num_res_blocks=1,
    norm_num_groups=16,
    attention_levels=(False, False, True),
)
autoencoder.to(device)

# autoencoder.load_state_dict(torch.load(os.path.join(directory_path, "auto_ddpm_3d_autoencoder_liver.pth")))
discriminator = PatchDiscriminator(spatial_dims=3, num_layers_d=3, num_channels=32, in_channels=1, out_channels=1)
discriminator.to(device)

l1_loss = L1Loss()
adv_loss = PatchAdversarialLoss(criterion="least_squares")
loss_perceptual = PerceptualLoss(spatial_dims=3, network_type="squeeze", is_fake_3d=True, fake_3d_ratio=0.2)
loss_perceptual.to(device)


def KL_loss(z_mu, z_sigma):
    kl_loss = 0.5 * torch.sum(z_mu.pow(2) + z_sigma.pow(2) - torch.log(z_sigma.pow(2)) - 1, dim=[1, 2, 3, 4])
    return torch.sum(kl_loss) / kl_loss.shape[0]


adv_weight = 0.01
perceptual_weight = 0.001
kl_weight = 1e-6
# -

optimizer_g = torch.optim.Adam(params=autoencoder.parameters(), lr=1e-4)
optimizer_d = torch.optim.Adam(params=discriminator.parameters(), lr=1e-4)

n_epochs = 2500
autoencoder_warm_up_n_epochs = 5
val_interval = 100
epoch_recon_loss_list = []
epoch_gen_loss_list = []
epoch_disc_loss_list = []
val_recon_epoch_loss_list = []
intermediary_images = []
n_example_images = 4

for epoch in range(n_epochs):
    autoencoder.train()
    discriminator.train()
    epoch_loss = 0
    gen_epoch_loss = 0
    disc_epoch_loss = 0
    progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), ncols=110)
    progress_bar.set_description(f"Epoch {epoch}")
    for step, batch in progress_bar:
        images = batch.to(device)  # choose only one of Brats channels

        # Generator part
        optimizer_g.zero_grad(set_to_none=True)
        reconstruction, z_mu, z_sigma = autoencoder(images)
        kl_loss = KL_loss(z_mu, z_sigma)

        recons_loss = l1_loss(reconstruction.float(), images.float())
        p_loss = loss_perceptual(reconstruction.float(), images.float())
        loss_g = recons_loss + kl_weight * kl_loss + perceptual_weight * p_loss

        if epoch > autoencoder_warm_up_n_epochs:
            logits_fake = discriminator(reconstruction.contiguous().float())[-1]
            generator_loss = adv_loss(logits_fake, target_is_real=True, for_discriminator=False)
            loss_g += adv_weight * generator_loss

        loss_g.backward()
        optimizer_g.step()

        if epoch > autoencoder_warm_up_n_epochs:
            # Discriminator part
            optimizer_d.zero_grad(set_to_none=True)
            logits_fake = discriminator(reconstruction.contiguous().detach())[-1]
            loss_d_fake = adv_loss(logits_fake, target_is_real=False, for_discriminator=True)
            logits_real = discriminator(images.contiguous().detach())[-1]
            loss_d_real = adv_loss(logits_real, target_is_real=True, for_discriminator=True)
            discriminator_loss = (loss_d_fake + loss_d_real) * 0.5

            loss_d = adv_weight * discriminator_loss

            loss_d.backward()
            optimizer_d.step()

        epoch_loss += recons_loss.item()
        if epoch > autoencoder_warm_up_n_epochs:
            gen_epoch_loss += generator_loss.item()
            disc_epoch_loss += discriminator_loss.item()

        progress_bar.set_postfix(
            {
                "recons_loss": epoch_loss / (step + 1),
                "gen_loss": gen_epoch_loss / (step + 1),
                "disc_loss": disc_epoch_loss / (step + 1),
            }
        )
    if (epoch + 1) % val_interval == 0:
        z = autoencoder.encode_stage_2_inputs(check_data.to(device))
        idx = 0
        img = reconstruction[idx, channel].detach().cpu().numpy()
        fig, axs = plt.subplots(nrows=1, ncols=3)
        for ax in axs:
            ax.axis("off")
        ax = axs[0]
        ax.imshow(img[..., img.shape[2] // 2], cmap="gray")
        ax = axs[1]
        ax.imshow(img[:, img.shape[1] // 2, ...], cmap="gray")
        ax = axs[2]
        ax.imshow(img[img.shape[0] // 2, ...], cmap="gray")
        plt.show()

    epoch_recon_loss_list.append(epoch_loss / (step + 1))
    epoch_gen_loss_list.append(gen_epoch_loss / (step + 1))
    epoch_disc_loss_list.append(disc_epoch_loss / (step + 1))
model_dir = os.path.join(directory_path, "auto_ddpm_3d_autoencoder_tumor_"+time.strftime("%Y-%m-%d", time.localtime())+".pth")
torch.save(autoencoder.state_dict(), model_dir)
del discriminator
del loss_perceptual

torch.cuda.empty_cache()
# -

plt.style.use("ggplot")
plt.title("Learning Curves", fontsize=20)
plt.plot(epoch_recon_loss_list)
plt.yticks(fontsize=12)
plt.xticks(fontsize=12)
plt.xlabel("Epochs", fontsize=16)
plt.ylabel("Loss", fontsize=16)
plt.legend(prop={"size": 14})
plt.show()

plt.title("Adversarial Training Curves", fontsize=20)
plt.plot(epoch_gen_loss_list, color="C0", linewidth=2.0, label="Generator")
plt.plot(epoch_disc_loss_list, color="C1", linewidth=2.0, label="Discriminator")
plt.yticks(fontsize=12)
plt.xticks(fontsize=12)
plt.xlabel("Epochs", fontsize=16)
plt.ylabel("Loss", fontsize=16)
plt.legend(prop={"size": 14})
plt.show()



unet = DiffusionModelUNet(
    spatial_dims=3,
    in_channels=3,
    out_channels=3,
    num_res_blocks=1,
    num_channels=(32, 64, 64),
    attention_levels=(False, True, True),
    num_head_channels=(0, 64, 64),
)
unet.to(device)


scheduler = DDPMScheduler(num_train_timesteps=1000, schedule="scaled_linear_beta", beta_start=0.0015, beta_end=0.0195)

with torch.no_grad():
    with autocast(enabled=True):
        z = autoencoder.encode_stage_2_inputs(check_data.to(device))

print(f"Scaling factor set to {1/torch.std(z)}")
scale_factor = 1 / torch.std(z)


inferer = LatentDiffusionInferer(scheduler, scale_factor=scale_factor)

optimizer_diff = torch.optim.Adam(params=unet.parameters(), lr=1e-4)


n_epochs = 2500
epoch_loss_list = []
autoencoder.eval()
scaler = GradScaler()

first_batch = first(train_loader)
z = autoencoder.encode_stage_2_inputs(first_batch.to(device))

for epoch in range(n_epochs):
    unet.train()
    epoch_loss = 0
    progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), ncols=70)
    progress_bar.set_description(f"Epoch {epoch}")
    for step, batch in progress_bar:
        images = batch.to(device)
        optimizer_diff.zero_grad(set_to_none=True)

        with autocast(enabled=True):
            # Generate random noise
            noise = torch.randn_like(z).to(device)

            # Create timesteps
            timesteps = torch.randint(
                0, inferer.scheduler.num_train_timesteps, (images.shape[0],), device=images.device
            ).long()

            # Get model prediction
            noise_pred = inferer(
                inputs=images,autoencoder_model=autoencoder, diffusion_model=unet, noise=noise, timesteps=timesteps
            )

            loss = F.mse_loss(noise_pred.float(), noise.float())

        scaler.scale(loss).backward()
        scaler.step(optimizer_diff)
        scaler.update()

        epoch_loss += loss.item()

        progress_bar.set_postfix({"loss": epoch_loss / (step + 1)})
    epoch_loss_list.append(epoch_loss / (step + 1))
# -
model_dir = os.path.join(directory_path, "auto_ddpm_3d_unet_liver_"+time.strftime("%Y-%m-%d", time.localtime())+".pth")

torch.save(unet.state_dict(), model_dir)
plt.plot(epoch_loss_list)
plt.title("Learning Curves", fontsize=20)
plt.plot(epoch_loss_list)
plt.yticks(fontsize=12)
plt.xticks(fontsize=12)
plt.xlabel("Epochs", fontsize=16)
plt.ylabel("Loss", fontsize=16)
plt.legend(prop={"size": 14})
plt.show()

