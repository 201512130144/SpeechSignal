import matplotlib.pyplot as  plt
import librosa.display as lrd
import torchaudio.transforms
from separator_yzx.SeparationBrain import SeparationBrain
from separator_yzx.SimpleSeparator import SimpleSeparator
import torch, os
from speechbrain.dataio.dataio import read_audio
from speechbrain.utils.checkpoints import Checkpointer

fft_size = 1024
N_train = 90
optimizer = lambda x: torch.optim.Adam(x, lr=0.0002)
run_opts = {
    "device": "cuda:0" if torch.cuda.is_available() else "cpu",
    "ckpt_interval_steps": 1
}

# 创建检查点目录
ckpt_path = 'exp/checkpoints'
os.makedirs(ckpt_path, exist_ok=True)

model_audio = SimpleSeparator(fft_size=fft_size, hidden_size=300)
# 创建 Checkpointer
myCheckpointer = Checkpointer(
    checkpoints_dir=ckpt_path,
    recoverables={
        "model": model_audio,
        "optimizer": optimizer
    },
    # 可选配置参数 # 每个epoch保存一次检查点
    # checkpoints_minute_interval=30,  # 每30分钟保存一次
    # max_checkpoints=5,  # 最多保留5个检查点
    # checkpoint_ready_to_save_fn=lambda: True,  # 自定义保存条件
)
separator = SeparationBrain(
    train_loss='si-snr',
    modules={'mdl': model_audio},
    opt_class=optimizer,
    run_opts=run_opts,
    checkpointer=myCheckpointer
)

separator.modules.mdl.eval()

audio_root = '../audio'
# audio_root = '/root/private_data/g813_u1/yzx/CLIP/audio/'
mixture_0 = read_audio(os.path.join(audio_root, 'mixture_0.wav')).squeeze()
source1_0 = read_audio(os.path.join(audio_root, 'source1_0.wav')).squeeze()
source2_0 = read_audio(os.path.join(audio_root, 'source2_0.wav')).squeeze()

mixture_1 = read_audio(os.path.join(audio_root, 'mixture_1.wav')).squeeze()
source1_1 = read_audio(os.path.join(audio_root, 'source1_1.wav')).squeeze()
source2_1 = read_audio(os.path.join(audio_root, 'source2_1.wav')).squeeze()

mixture_2 = read_audio(os.path.join(audio_root, 'mixture_2.wav')).squeeze()
source1_2 = read_audio(os.path.join(audio_root, 'source1_2.wav')).squeeze()
source2_2 = read_audio(os.path.join(audio_root, 'source2_2.wav')).squeeze()

mixture_3 = read_audio(os.path.join(audio_root, 'mixture_3.wav')).squeeze()
source1_3 = read_audio(os.path.join(audio_root, 'source1_3.wav')).squeeze()
source2_3 = read_audio(os.path.join(audio_root, 'source2_3.wav')).squeeze()

# Audio(mixture_0, rate=16000)

# estimated_sources, all_masks, mag = separator.modules.mdl.forward(mixture_3[N_train:].unsqueeze(0).to(run_opts['device']))
estimated_sources, all_masks, mag = separator.modules.mdl.forward(mixture_3.unsqueeze(0).to(run_opts['device']))
estimated_sources = [src.cpu().detach() for src in estimated_sources]
all_masks = [mask.cpu().detach() for mask in all_masks]
mag = mag.cpu()

# estimated_sources_train, all_masks, mag = model_audio.forward(mixture_0.unsqueeze(0))


plt.figure(figsize=[20, 10], dpi=80)

plt.subplot(331)
mag = mag[0].t().numpy()
lrd.specshow(mag, y_axis='log')
plt.title('Mixture')
plt.colorbar()

plt.subplot(334)
mask1 = all_masks[0][0].detach().t().numpy()
lrd.specshow(mask1, y_axis='log')
plt.title('Mask for source 1')
plt.colorbar()

plt.subplot(335)
masked1 = mask1 * mag
lrd.specshow(masked1, y_axis='log')
plt.title('Estimated Source 1')
plt.colorbar()

plt.subplot(336)
# source1_gt = source1_3[N_train]
source1_gt = source1_3
# 本质求模
source1_spec = torch.sqrt((torch.view_as_real(torch.stft(source1_gt, n_fft=fft_size, return_complex=True))**2).sum(-1))
lrd.specshow(source1_spec.numpy(), y_axis='log')
plt.title('Ground Truth Source 1')
plt.colorbar()

plt.subplot(337)
mask2 = all_masks[1][0].detach().t().numpy()
lrd.specshow(mask2, y_axis='log')
plt.title('Mask for Source 2')
plt.colorbar()

plt.subplot(338)
masked2 = mask2 * mag
lrd.specshow(masked2, y_axis='log')
plt.title('Estimated Source 2')
plt.colorbar()

plt.subplot(339)
# source2_gt = source2_3[N_train]
source2_gt = source2_3
torchaudio.transforms.Spectrogram()
source2_spec = torch.sqrt((torch.view_as_real(torch.stft(source2_gt, n_fft=fft_size, return_complex=True)**2)).sum(-1))
lrd.specshow(source2_spec.numpy(), y_axis='log')
plt.title('Ground Truth Source 2')
plt.colorbar()

plt.show()