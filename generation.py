import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from generate_net.autoencoderkl import AutoencoderKL
from generate_net.diffusion_model_unet import DiffusionModelUNet
from generate_net.inferer import LatentDiffusionInferer
from generate_net.ddpm import DDPMScheduler
import torch
from monai.data import DataLoader
import nibabel as nib
from tqdm import tqdm
import torch.nn as nn
import os
from monai.utils import first, set_determinism
import matplotlib.pyplot as plt
import numpy as np
import nibabel
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import random
import cv2
tumor_hu=14
tumor_inspection='/Addresses of real tumours'

directory_path = '/Address of previously saved ldm model'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tumor_size=4
print(f"Using {device}")
def pca_kl(transformed_image1, transformed_image2):
    data1 = transformed_image1.squeeze(0).squeeze(0).numpy()
    data2 = transformed_image2.squeeze(0).squeeze(0).numpy()
    data1_reshaped = data1.reshape(-1, np.prod(data1.shape[1:]))
    data2_reshaped = data2.reshape(-1, np.prod(data2.shape[1:]))
    scaler1 = StandardScaler()
    scaler2 = StandardScaler()

    data1_reshaped_scaled = scaler1.fit_transform(data1_reshaped)
    data2_reshaped_scaled = scaler2.fit_transform(data2_reshaped)

    pca1 = PCA(n_components=3)
    pca2 = PCA(n_components=3)

    data1_pca = pca1.fit_transform(data1_reshaped_scaled)
    data2_pca = pca2.fit_transform(data2_reshaped_scaled)

    # 计算均值和协方差矩阵
    mu1 = np.mean(data1_pca, axis=0)
    mu2 = np.mean(data2_pca, axis=0)
    Sigma1 = np.cov(data1_pca, rowvar=False)
    Sigma2 = np.cov(data2_pca, rowvar=False)
    # 计算KL散度
    kl_divergence = 0.5 * (np.trace(np.linalg.inv(Sigma2) @ Sigma1) +
                           (mu1 - mu2) @ np.linalg.inv(Sigma2) @ (mu1 - mu2) +
                           np.log(np.linalg.det(Sigma2) / np.linalg.det(Sigma1))
                           -2
                           # +data1_pca.shape[1] - data2_pca.shape[1]
                           )
    return kl_divergence
def load_nifti_file0(file_path):
    # 读取.nii或.nii.gz文件
    img_ct = nib.load(file_path)
    # 获取数据
    data = img_ct.get_fdata()

    return data
def load_nifti_file(file_path):
    # 读取.nii或.nii.gz文件
    img_ct = nib.load(file_path)
    # 获取数据
    data = img_ct.get_fdata()
    affine = img_ct.affine
    header=img_ct.header

    # 获取体素尺寸（affine 矩阵的对角线上的元素）
    voxel_sizes = np.diag(affine)[:3]
    return data,affine,voxel_sizes,header
def create_dataset(folder_path,roi_size,spatial_size):
    # 创建数据集列表
    dataset = []
    # 遍历文件夹中的所有文件
    file_paths = [os.path.join(folder_path, file_name) for file_name in os.listdir(folder_path) if file_name.endswith(".nii") or file_name.endswith(".nii.gz")]
    for file_path in tqdm(file_paths,desc="Loading case Nifti files"):
        # 加载NIfTI文件
        image_data= torch.from_numpy(load_nifti_file0(file_path)).float()

        transformed_image =  F.interpolate(image_data.unsqueeze(0).unsqueeze(0), size=spatial_size, mode='trilinear', align_corners=False)
        transformed_image = transformed_image.squeeze(0)
        # 将图像添加到数据集
        dataset.append(transformed_image)
    return dataset
# 文件路径列表
file_paths = '/Tumour ct Get Sample Address'
# 定义新的图像形状，例如：(128, 128, 128)
new_shape = (96,96,64)
# 创建数据集
train_data = create_dataset(file_paths,roi_size=(400, 400, 300),spatial_size=new_shape)
batch_size = 1
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4, persistent_workers=True)
check_data = first(train_loader)
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

autoencoder.load_state_dict(torch.load(os.path.join(directory_path, "auto_ddpm_3d_autoencoder_tumor_2024-07-27.pth")))
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
unet.load_state_dict(torch.load(os.path.join(directory_path, "auto_ddpm_3d_unet_liver_2024-07-27.pth")))
scheduler = DDPMScheduler(num_train_timesteps=1000, schedule="scaled_linear_beta", beta_start=0.0015, beta_end=0.0195)
with torch.no_grad():
    with autocast(enabled=True):
        z = autoencoder.encode_stage_2_inputs(check_data.to(device))

print(f"Scaling factor set to {1/torch.std(z)}")

autoencoder.eval()
unet.eval()
scale_factor = 1 / torch.std(z)
inferer = LatentDiffusionInferer(scheduler, scale_factor=scale_factor)
liver_ct_path = "/Address of the real liver"
liver_mask_path = "/Address of the real liver mask"
synthesis1_mask_path='/Address of the generated t1a tumour mask'
synthesis1_ct_path='/Address of the generated t1a tumour'
synthesis2_mask_path='/Address of the generated t1b tumourmask'
synthesis2_ct_path='/Address of the generated t1b tumour'
synthesis3_mask_path='/Address of the generated t3 tumour mask'
synthesis3_ct_path='/Address of the generated t3 tumour'
def random_select(mask_scan):
    # we first find z index and then sample point with z slice
    z_start, z_end = np.where(np.any(mask_scan, axis=(0, 1)))[0][[0, -1]]

    # we need to strict number z's position (0.3 - 0.7 in the middle of liver)
    z = round(random.uniform(0.3, 0.7) * (z_end - z_start)) + z_start

    liver_mask = mask_scan[..., z]

    # erode the mask (we don't want the edge points)
    kernel = np.ones((5,5), dtype=np.uint8)
    liver_mask = cv2.erode(liver_mask, kernel, iterations=1)
    #腐蚀是图像处理中的一个基本操作，它可以用来去除图像中的小物体或细节，例如图像中的噪声，

    coordinates = np.argwhere(liver_mask >= 1)
    random_index = np.random.randint(0, len(coordinates))
    xyz = coordinates[random_index].tolist() # get x,y
    xyz.append(z)
    potential_points = xyz

    return potential_points
def synthesis(liver_ct, liver_mask,tumor_generate,tumor_mask):
    enlarge_x, enlarge_y, enlarge_z = 160, 160, 160

    liver_ct_img,affine1,voxel_sizes1,header1 = load_nifti_file(liver_ct)
    liver_mask_img ,affine,voxel_sizes,header= load_nifti_file(liver_mask)
    potential_points = random_select(liver_mask_img)
    synthesis_ct = np.zeros(
        (liver_ct_img.shape[0] + enlarge_x, liver_ct_img.shape[1] + enlarge_y, liver_ct_img.shape[2] + enlarge_z), dtype=tumor_mask.dtype)
    synthesis_mask = np.zeros(
        (liver_ct_img.shape[0] + enlarge_x, liver_ct_img.shape[1] + enlarge_y, liver_ct_img.shape[2] + enlarge_z),
        dtype=tumor_mask.dtype)
    new_point = [potential_points[0] + enlarge_x // 2, potential_points[1] + enlarge_y // 2, potential_points[2] + enlarge_z // 2]
    x_low, x_high = new_point[0] - tumor_mask.shape[0] // 2, new_point[0] + tumor_mask.shape[0] // 2
    y_low, y_high = new_point[1] - tumor_mask.shape[1] // 2, new_point[1] + tumor_mask.shape[1] // 2
    z_low, z_high = new_point[2] - tumor_mask.shape[2] // 2, new_point[2] + tumor_mask.shape[2] // 2

    synthesis_mask[x_low:x_high, y_low:y_high, z_low:z_high] += tumor_mask

    synthesis_mask = synthesis_mask[enlarge_x // 2:-enlarge_x // 2, enlarge_y // 2:-enlarge_y // 2, enlarge_z // 2:-enlarge_z // 2]

    liver_mask_img[np.where(synthesis_mask == 2)]=2
    synthesis_mask=liver_mask_img
    synthesis_ct[enlarge_x // 2:-enlarge_x // 2, enlarge_y // 2:-enlarge_y // 2, enlarge_z // 2:-enlarge_z // 2] = liver_ct_img
    synthesis_ct[x_low:x_high, y_low:y_high, z_low:z_high] += tumor_generate
    synthesis_ct = synthesis_ct[enlarge_x // 2:-enlarge_x // 2, enlarge_y // 2:-enlarge_y // 2,
                   enlarge_z // 2:-enlarge_z // 2]
    return synthesis_ct, synthesis_mask,affine,voxel_sizes,header,affine1,voxel_sizes1,header1


# synthesis(liver_ct, liver_mask)
def kl_divergence(mu, sigma):
    kl = (sigma**2 + mu**2 - np.log(sigma**2) - 1) / 2
    return kl
number_g=53
for i in range(20):
    noise = torch.randn((1, 3, z.shape[2], z.shape[3], z.shape[4]))
    noise = noise.to(device)
    synthetic_images = inferer.sample(
        input_noise=noise, autoencoder_model=autoencoder, diffusion_model=unet, scheduler=scheduler
    )
    idx = 0
    channel = 0
    img = synthetic_images[idx, channel].detach().cpu().numpy()

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
    plt.imsave('tumor_mask'+str(i)+'_1.png', img[..., img.shape[2] // 2], cmap="gray")
    plt.imsave('tumor_mask'+str(i)+'_2.png', img[:, img.shape[1] // 2, ...], cmap="gray")
    plt.imsave('tumor_mask'+str(i)+'_3.png', img[img.shape[0] // 2, ...], cmap="gray")

    file_paths = [os.path.join(tumor_inspection, file_name) for file_name in os.listdir(tumor_inspection) if
                  file_name.endswith(".nii") or file_name.endswith(".nii.gz")]
    pca_kl_v=[]
    for file_path in tqdm(file_paths, desc="Loading pca Nifti files"):
        image_data= torch.from_numpy(load_nifti_file0(file_path)).float()
        spatial_size = (96, 96, 64)
        transformed_image1 = F.interpolate(image_data.unsqueeze(0).unsqueeze(0), size=spatial_size, mode='trilinear',
                                           align_corners=False)
        transformed_image1 = transformed_image1
        tumor_mask = img.copy()  # images
        if np.any(tumor_mask < 0):
            # print(np.unique(tumor_mask))
            tumor_mask[tumor_mask < 0] = 0
            if tumor_size == 4:
                transformed_image2 = tumor_mask.reshape(1, 1, 96, 96, 64)
            if tumor_size == 3:
                transformed_image2 = tumor_mask.reshape(1, 1, 96, 96, 64)
            if tumor_size == 2:
                transformed_image2 = tumor_mask.reshape(1, 1, 96, 96, 64)
            if tumor_size == 1:
                transformed_image2 = tumor_mask.reshape(1, 1, 96, 96, 64)

            transformed_image2 = torch.from_numpy(transformed_image2).float()
        pca_kl_value = pca_kl(transformed_image1, transformed_image2)
        pca_kl_v.append(pca_kl_value)
        print(f"PCA KL divergence: {pca_kl_value}")
    print(f"Mean PCA KL divergence: {np.mean(pca_kl_v)}")
    print(f"Std PCA KL divergence: {np.std(pca_kl_v)}")
    image_data = img.copy()
    image_data[np.where(image_data <= 0)] = 0
    image_data= img[np.abs(image_data) > 0.1]
    print(np.mean(image_data))
    if np.std(pca_kl_v) < 0.09 and np.std(pca_kl_v) > 0.05:
    if np.std(pca_kl_v)>0.09 or np.std(pca_kl_v)<0.05:
        if np.mean(pca_kl_v)<0.65:
            if np.mean(image_data)>35:

                if tumor_size==3:
                    print("generate")
                    health_ct_file_paths = [os.path.join(liver_ct_path , file_name) for file_name in os.listdir(liver_ct_path ) if
                                  file_name.endswith(".nii") or file_name.endswith(".nii.gz")]

                    for file_path in tqdm(health_ct_file_paths, desc="Loading Nifti files"):
                        file_name = os.path.basename(file_path)
                        liver_ct_path1 = os.path.join(liver_ct_path, file_name)
                        liver_mask_path1 = os.path.join(liver_mask_path, file_name)
                        tumor_g = img.copy()
                        tumor_mask = img.copy()
                        tumor_mask[tumor_g <= tumor_hu] = 0
                        tumor_mask[tumor_g > tumor_hu] = 2
                        casename = "synthesis_%04.0d" % number_g
                        synthesis_ct, synthesis_mask,affine,voxel_sizes,header,affine1,voxel_sizes1,header1=synthesis(liver_ct_path1, liver_mask_path1, tumor_g, tumor_mask)
                        synthesis_ct = nibabel.Nifti1Image(synthesis_ct, affine1,header1)
                        nibabel.save(synthesis_ct, os.path.join(synthesis3_ct_path, casename + ".nii.gz"))
                        print(f"Generated image saved to {casename}")
                        synthesis_mask = nibabel.Nifti1Image(synthesis_mask, affine, header)
                        nibabel.save(synthesis_mask, os.path.join(synthesis3_mask_path, casename + ".nii.gz"))
                        number_g += 1



                if tumor_size == 2:
                    # 计算每个维度的实际大小
                    spatial_size = (48, 48, 32)

                    synthetic_images= F.interpolate(synthetic_images, size=spatial_size, mode='trilinear',
                                                       align_corners=False)

                    img = synthetic_images[idx, channel].detach().cpu().numpy()

                    print("generate")
                    health_ct_file_paths = [os.path.join(liver_ct_path, file_name) for file_name in
                                            os.listdir(liver_ct_path) if
                                            file_name.endswith(".nii") or file_name.endswith(".nii.gz")]

                    for file_path in tqdm(health_ct_file_paths, desc="Loading Nifti files"):
                        file_name = os.path.basename(file_path)
                        liver_ct_path1 = os.path.join(liver_ct_path, file_name)
                        liver_mask_path1 = os.path.join(liver_mask_path, file_name)

                        tumor_g = img.copy()
                        tumor_mask = img.copy()
                        tumor_mask[tumor_g <= tumor_hu] = 0
                        tumor_mask[tumor_g > tumor_hu] = 2
                        casename = "synthesis_%04.0d" % number_g
                        synthesis_ct, synthesis_mask, affine, voxel_sizes, header, affine1, voxel_sizes1, header1 = synthesis(
                            liver_ct_path1, liver_mask_path1, tumor_g, tumor_mask)
                        synthesis_ct = nibabel.Nifti1Image(synthesis_ct, affine1, header1)
                        nibabel.save(synthesis_ct, os.path.join(synthesis2_ct_path, casename + ".nii.gz"))
                        print(f"Generated image saved to {casename}")
                        synthesis_mask = nibabel.Nifti1Image(synthesis_mask, affine, header)
                        nibabel.save(synthesis_mask, os.path.join(synthesis2_mask_path, casename + ".nii.gz"))
                        number_g += 1

                if tumor_size == 1:
                    # 计算每个维度的实际大小
                    spatial_size = (16, 16, 8)

                    synthetic_images = F.interpolate(synthetic_images, size=spatial_size, mode='trilinear',
                                                     align_corners=False)

                    img = synthetic_images[idx, channel].detach().cpu().numpy()

                    print("generate")
                    health_ct_file_paths = [os.path.join(liver_ct_path, file_name) for file_name in
                                            os.listdir(liver_ct_path) if
                                            file_name.endswith(".nii") or file_name.endswith(".nii.gz")]

                    for file_path in tqdm(health_ct_file_paths, desc="Loading Nifti files"):
                        file_name = os.path.basename(file_path)
                        liver_ct_path1 = os.path.join(liver_ct_path, file_name)
                        liver_mask_path1 = os.path.join(liver_mask_path, file_name)

                        tumor_g = img.copy()
                        tumor_mask = img.copy()
                        tumor_mask[tumor_g <= tumor_hu] = 0
                        tumor_mask[tumor_g > tumor_hu] = 2
                        casename = "synthesis_%04.0d" % number_g
                        synthesis_ct, synthesis_mask, affine, voxel_sizes, header, affine1, voxel_sizes1, header1 = synthesis(
                            liver_ct_path1, liver_mask_path1, tumor_g, tumor_mask)

                        actual_sizes = voxel_sizes * (spatial_size - np.array([1, 1, 1]))
                        print(f"每个维度的实际大小：{actual_sizes}，单位通常是毫米")
                        synthesis_ct = nibabel.Nifti1Image(synthesis_ct, affine1, header1)
                        nibabel.save(synthesis_ct, os.path.join(synthesis1_ct_path, casename + ".nii.gz"))
                        print(f"Generated image saved to {casename}")
                        synthesis_mask = nibabel.Nifti1Image(synthesis_mask, affine, header)
                        nibabel.save(synthesis_mask, os.path.join(synthesis1_mask_path, casename + ".nii.gz"))
                        number_g += 1







