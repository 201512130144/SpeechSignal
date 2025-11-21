from torch import nn as nn


class audioseparator(nn.Module):
    def __init__(self, fft_size, hidden_size, num_sources=2, kernel_size=16):
        super(audioseparator, self).__init__()
        self.encoder = nn.Conv1d(in_channels=1, out_channels=fft_size, kernel_size=16, stride=kernel_size // 2)

        # MaskNet
        self.rnn = nn.LSTM(input_size=fft_size, hidden_size=hidden_size, batch_first=True, bidirectional=True)
        self.output_layer = nn.Linear(in_features=hidden_size * 2, out_features=num_sources * (fft_size))

        self.decoder = nn.ConvTranspose1d(in_channels=fft_size, out_channels=1, kernel_size=kernel_size,
                                          stride=kernel_size // 2)

        self.fft_size = fft_size
        self.hidden_size = hidden_size
        self.num_sources = num_sources

    def forward(self, inp):
        # batch x channels x time
        y = nn.functional.relu(self.encoder(inp.unsqueeze(0)))

        # batch x time x nfft
        y = y.permute(0, 2, 1)

        # batch x time x feature
        rnn_out = self.rnn(y)[0]

        # batch x time x (nfft*num_sources)
        lin_out = self.output_layer(rnn_out)

        # batch x time x nfft x num_sources
        lin_out = lin_out.reshape(lin_out.size(0), lin_out.size(1), -1, self.num_sources)

        # reconstruct in time domain
        sources = []
        all_masks = []
        for n in range(self.num_sources):
            sourcehat_mask = nn.functional.relu(lin_out[:, :, :, n])
            all_masks.append(sourcehat_mask)

            # multiply with mask and magnitude
            T = sourcehat_mask.size(1)
            sourcehat_latent = (sourcehat_mask * y[:, :T, :]).permute(0, 2, 1)

            # reconstruct in time domain with istft
            sourcehat = self.decoder(sourcehat_latent).squeeze(0)
            sources.append(sourcehat)

        return sources, all_masks, y
