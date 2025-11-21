import librosa.display as lrd
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.utils.data as data_utils

N = 100
f_th = 200
fs = 8000

T = 10000
t = torch.arange(0, T).unsqueeze(0)
f1 = torch.randint(5, f_th, (N, 1))
f2 = torch.randint(f_th, 400, (N, 1))
batch_size = 10

source1 = torch.sin(2 * np.pi * (f1 / fs) * t)
source2 = torch.sin(2 * np.pi * (f2 / fs) * t)
mixture = source1 + source2
N_train = 90
train_dataset = data_utils.TensorDataset(source1[:N_train], source2[:N_train], mixture[:N_train])
test_dataset = data_utils.TensorDataset(source1[N_train:], source2[N_train:], mixture[N_train:])

train_loader = data_utils.DataLoader(train_dataset, batch_size=batch_size)
test_loader = data_utils.DataLoader(test_dataset, batch_size=batch_size)

# now let's visualize the frequency spectra for the dataset
fft_size = 200

plt.figure(figsize=[20, 10], dpi=50)

plt.subplot(131)
mix_gt = mixture[N_train]
mix_spec = torch.sqrt((torch.view_as_real(torch.stft(mix_gt, n_fft=fft_size, return_complex=True)) ** 2).sum(-1))
lrd.specshow(mix_spec.numpy(), y_axis='log')
plt.title('Mixture Spectrogram')

plt.subplot(132)
source1_gt = source1[N_train]
source1_spec = torch.sqrt(
    (torch.view_as_real(torch.stft(source1_gt, n_fft=fft_size, return_complex=True)) ** 2).sum(-1))
lrd.specshow(source1_spec.numpy(), y_axis='log')
plt.title('Source 1 Spectrogram')

plt.subplot(133)
source2_gt = source2[N_train]
source2_spec = torch.sqrt(
    (torch.view_as_real(torch.stft(source2_gt, n_fft=fft_size, return_complex=True)) ** 2).sum(-1))
lrd.specshow(source2_spec.numpy(), y_axis='log')
plt.title('Source 2 Spectrogram')

plt.show()
