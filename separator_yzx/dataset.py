import os.path

from IPython.display import Audio
from speechbrain.dataio.dataio import read_audio
from torch.utils.data import Dataset, DataLoader

audio_root = 'audio'
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

train_mixs = [mixture_0, mixture_1, mixture_2]
train_source1s = [source1_0, source1_1, source1_2]
train_source2s = [source2_0, source2_1, source2_2]


# Audio(mixture_0, rate=16000)


class source_separation_dataset(Dataset):
    def __init__(self, train_mixs, train_source1s, train_source2s, device='cpu'):
        self.mixs = train_mixs
        self.train_source1s = train_source1s
        self.train_source2s = train_source2s
        self.device = device

    def __len__(self):
        return len(self.mixs)

    def __getitem__(self, idx):
        mix = self.mixs[idx].to(self.device)
        source1 = self.train_source1s[idx].to(self.device)
        source2 = self.train_source2s[idx].to(self.device)
        return mix, source1, source2


def load_audio(device='cpu'):
    train_dataset_audio = source_separation_dataset(train_mixs, train_source1s, train_source2s, device=device)
    valid_dataset_audio = source_separation_dataset([mixture_2], [source1_2], [source2_2], device=device)

    train_loader_audio = DataLoader(train_dataset_audio, batch_size=1)
    valid_loader_audio = DataLoader(valid_dataset_audio, batch_size=1)
    return train_loader_audio, valid_loader_audio
