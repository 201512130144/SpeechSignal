# 给你一个直接可用的推理脚本模板（最安全、最简洁）
import torch, os
from speechbrain.utils.checkpoints import Checkpointer
from SimpleSeparator import SimpleSeparator
from speechbrain.dataio.dataio import read_audio
import matplotlib.pyplot as plt
import librosa.display as lrd


fft_size = 1024
N_train = 90
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 初始化模型
model = SimpleSeparator(fft_size=fft_size, hidden_size=300).to(device)

# 加载 FINAL checkpoint
ckpt = Checkpointer(
    checkpoints_dir='exp/checkpoints',
    recoverables={'mdl': model},
)

ckpt.recover_if_possible()
model.eval()

audio_root = 'audio'

# audio_root = '/root/private_data/g813_u1/yzx/CLIP/audio/'
mixture_0 = read_audio(os.path.join(audio_root, 'mixture_0.wav')).squeeze()
source1_0 = read_audio(os.path.join(audio_root, 'source1_0.wav')).squeeze()
source2_0 = read_audio(os.path.join(audio_root, 'source2_0.wav')).squeeze()

estimated_sources, all_masks, mag = model.forward(mixture_0.unsqueeze(0).to(device))
estimated_sources = [src.cpu().detach() for src in estimated_sources]
all_masks = [mask.cpu().detach() for mask in all_masks]
mag = mag.cpu()

# estimated_sources_train, all_masks, mag = model_audio.forward(mixture_0.unsqueeze(0))


plt.figure(figsize=[20, 10], dpi=80)

plt.subplot(331)
mag = mag[0].t().numpy()
# lrd.specshow(mag, y_axis='log')
lrd.specshow(mag, y_axis='log')
plt.title('Mixture')
plt.colorbar()

plt.subplot(334)
mask1 = all_masks[0][0].detach().t().numpy()
# mask1 = all_masks[0][0].detach().t().numpy()[N_train].reshape(1, -1)
lrd.specshow(mask1, y_axis='log')
plt.title('Mask for source 1')
plt.colorbar()

plt.subplot(335)
masked1 = mask1 * mag
lrd.specshow(masked1, y_axis='log')
# lrd.specshow(masked1[N_train].reshape(1, -1), y_axis='log')
plt.title('Estimated Source 1')
plt.colorbar()

plt.subplot(336)
source1_gt = source1_0
# 本质求模
# source1_spec = torch.sqrt((torch.view_as_real(torch.stft(source1_gt, n_fft=fft_size, return_complex=True))**2).sum(-1))
source1_spec = torch.sqrt((torch.view_as_real(torch.stft(source1_gt, n_fft=fft_size, return_complex=True))**2).sum(-1))[N_train].view(1, -1)
lrd.specshow(source1_spec.numpy(), y_axis='log')
plt.title('Ground Truth Source 1')
plt.colorbar()

plt.subplot(337)
# mask2 = all_masks[1][0].detach().t().numpy()
mask2 = all_masks[1][0].detach().t().numpy()[N_train].reshape(1, -1),
lrd.specshow(mask2, y_axis='log')
plt.title('Mask for Source 2')
plt.colorbar()

plt.subplot(338)
masked2 = mask2 * mag
# lrd.specshow(masked2, y_axis='log')
lrd.specshow(masked2[N_train].reshape(1, -1), y_axis='log')
plt.title('Estimated Source 2')
plt.colorbar()

plt.subplot(339)
source2_gt = source2_0
# torch.Size([513, 252]) 全部频率带的模值
# source2_spec = torch.sqrt((torch.view_as_real(torch.stft(source2_gt, n_fft=fft_size, return_complex=True)**2)).sum(-1))
# 其中一个频率带的模值
source2_spec = torch.sqrt((torch.view_as_real(torch.stft(source2_gt, n_fft=fft_size, return_complex=True)**2)).sum(-1))[N_train].view(1, -1)
lrd.specshow(source2_spec.numpy(), y_axis='log')
plt.title('Ground Truth Source 2')
plt.colorbar()

plt.show()