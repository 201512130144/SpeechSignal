import torch
import torch.nn as nn


# define the model
class SimpleSeparator(nn.Module):
    def __init__(self, fft_size, hidden_size, num_sources=2):
        super(SimpleSeparator, self).__init__()
        self.masking = nn.LSTM(input_size=fft_size // 2 + 1, hidden_size=hidden_size, batch_first=True,
                               bidirectional=True)
        self.output_layer = nn.Linear(in_features=hidden_size * 2, out_features=num_sources * (fft_size // 2 + 1))
        self.fft_size = fft_size
        self.num_sources = num_sources

    def forward(self, inp):
        # batch x freq x time x realim
        y = torch.view_as_real(torch.stft(inp, n_fft=self.fft_size, return_complex=True))

        # batch X freq x time
        mag = torch.sqrt((y ** 2).sum(-1))
        phase = torch.atan2(y[:, :, :, 1], y[:, :, :, 0])

        # batch x time x freq
        mag = mag.permute(0, 2, 1)

        # batch x time x feature
        rnn_out = self.masking(mag)[0]

        # batch x time x (nfft*num_sources)
        lin_out = self.output_layer(rnn_out)

        # batch x time x nfft x num_sources
        lin_out = nn.functional.relu(lin_out.reshape(lin_out.size(0), lin_out.size(1), -1, self.num_sources))

        # reconstruct in time domain
        sources = []
        all_masks = []
        for n in range(self.num_sources):
            sourcehat_mask = (lin_out[:, :, :, n])
            all_masks.append(sourcehat_mask)

            # multiply with mask and magnitude
            sourcehat_dft = (sourcehat_mask * mag).permute(0, 2, 1) * torch.exp(1j * phase)

            # reconstruct in time domain with istft
            sourcehat = torch.istft(sourcehat_dft, n_fft=self.fft_size)
            sources.append(sourcehat)
        return sources, all_masks, mag
