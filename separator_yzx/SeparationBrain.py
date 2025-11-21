import speechbrain as sb
import torch
import torch.nn as nn


class SeparationBrain(sb.Brain):
    def __init__(self, train_loss, modules, opt_class, run_opts, checkpointer):
        super(SeparationBrain, self).__init__(modules=modules, opt_class=opt_class,
                                              run_opts=run_opts, checkpointer=checkpointer)
        self.train_loss = train_loss

    def compute_forward(self, mix):
        """Forward computations from the mixture to the separated signals."""

        # Get the estimates for the sources
        est_sources, _, _ = self.modules.mdl(mix)

        est_sources = torch.stack(est_sources, dim=-1)

        # T changed after conv1d in encoder, fix it here
        T_origin = mix.size(1)
        T_est = est_sources.size(1)
        # 截取等长的长度
        if T_origin > T_est:
            est_sources = nn.functional.pad(est_sources, (0, 0, 0, T_origin - T_est))
        else:
            est_sources = est_sources[:, :T_origin, :]

        return est_sources

    def compute_objectives(self, targets, est_sources):
        """Computes the loss functions between estimated and ground truth sources"""
        if self.train_loss == 'l1':
            return (est_sources - targets).abs().mean()
        elif self.train_loss == 'si-snr':
            return sb.nnet.losses.get_si_snr_with_pitwrapper(targets, est_sources).mean()

    def fit_batch(self, batch):
        """Trains one batch"""
        # Unpacking batch list
        source1, source2, mix = batch
        targets = torch.stack([source1, source2], dim=-1)

        est_sources = self.compute_forward(mix)
        # 逻辑重点  yzx 2025年11月18日03:58:03
        loss = self.compute_objectives(targets, est_sources)

        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        return loss.detach().cpu()

    def evaluate_batch(self, batch, stage):
        """Computations needed for test batches"""

        source1, source2, mix = batch
        targets = torch.stack([source1, source2], dim=-1)

        est_sources = self.compute_forward(mix)

        si_snr = sb.nnet.losses.get_si_snr_with_pitwrapper(targets, est_sources)
        si_snr_mean = si_snr.mean().item()
        print('VALID SI-SNR = {}'.format(-si_snr_mean))
        return si_snr.mean().detach()

# from functools import partial
#
# optimizer = lambda x: torch.optim.Adam(x, lr=0.0001)
# N_epochs = 10
# epoch_counter = sb.utils.epoch_loop.EpochCounter(limit=N_epochs)
#
# separator = SeparationBrain(
#         train_loss='l1',
#         modules={'mdl': model},
#         opt_class=optimizer
#
#     )
#
#
# separator.fit(
#             epoch_counter,
#             train_loader,
#             test_loader)
