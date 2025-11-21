"""Audio Source Separation Training Script Using SpeechBrain
Improvements applied:
- Clean imports and formatting
- Safer checkpointer usage and recovery before creating Brain
- Use run_on_main for filesystem operations in distributed setups
- Clearer exception handling for saving and cleanup
- Move seeds and device handling into run_opts
- Helpful logging

Notes:
- This script assumes `SimpleSeparator`, `SeparationBrain`, and `dataset.load_audio`
  exist and follow SpeechBrain conventions.
- You still need to verify SpeechBrain version compatibility in your environment.
"""

import os
import sys
import logging
import random
import torch
import numpy as np

import speechbrain as sb
from speechbrain.utils.checkpoints import Checkpointer
from speechbrain.utils.distributed import run_on_main

from SimpleSeparator import SimpleSeparator
from SeparationBrain import SeparationBrain
import dataset as dataset
# import ftfy as ftfy
#
# ftfy.fix_text('The Mona Lisa doesnÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢t have eyebrows.')
# -------------------------
# Logging
# -------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("train_separation")

# -------------------------
# Reproducibility
# -------------------------
def set_seed(seed: int = 1):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    # Deterministic CUDNN may slow things down — keep intentionally explicit
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.info(f"已设置随机种子为 {seed}，保证实验可复现")


# -------------------------
# Hyperparameters
# -------------------------
class HParams:
    def __init__(self):
        # Model params
        self.fft_size = 1024
        self.hidden_size = 300

        # Training params
        self.n_epochs = 5
        self.learning_rate = 2e-4
        self.weight_decay = 1e-5

        # DataLoader params
        self.batch_size = 2
        self.num_workers = 7
        self.pin_memory = True

        # Loss
        self.train_loss = "si-snr"

        # Checkpointing
        self.checkpoint_dir = "exp/checkpoints"
        self.save_interval = 1
        self.keep_checkpoint_max = 5

        # Device (SpeechBrain's run_opts will also control device placement)
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

        # Output
        self.output_dir = "exp/results"


# -------------------------
# Optional custom hooks
# -------------------------
def custom_save_hook(instance, path):
    logger.info(f"[CUSTOM SAVE] Saving model state to: {path}")
    torch.save({"state_dict": instance.state_dict(), "meta": {"note": "custom"}}, path)


# end_of_epoch ?
def custom_load_hook(instance, path, end_of_epoch=True):
    logger.info(f"[CUSTOM LOAD] Loading model state from: {path}")
    ckpt = torch.load(path, map_location="cpu")
    # 与 load state_dict 不同，这里直接加载 ckpt   但是效果一样
    instance.load_state_dict(ckpt)


# -------------------------
# Training
# -------------------------
def train_separation_model(hparams: HParams = None, use_custom_hooks: bool = False):
    if hparams is None:
        hparams = HParams()

    set_seed(2)

    # Create directories (only on main process)
    run_on_main(lambda: os.makedirs(hparams.checkpoint_dir, exist_ok=True))
    run_on_main(lambda: os.makedirs(hparams.output_dir, exist_ok=True))

    logger.info("=" * 70)
    logger.info("音频源分离模型训练")
    logger.info("=" * 70)
    logger.info(f"运行设备: {hparams.device}")
    logger.info(f"批大小: {hparams.batch_size}")
    logger.info(f"训练轮数: {hparams.n_epochs}")
    logger.info(f"学习率: {hparams.learning_rate}")
    logger.info(f"检查点目录: {hparams.checkpoint_dir}")

    # -------------------------
    # Model and optimizer
    # -------------------------
    logger.info("正在初始化模型...")
    model = SimpleSeparator(fft_size=hparams.fft_size, hidden_size=hparams.hidden_size)

    def opt_class(params):
        return torch.optim.Adam(params, lr=hparams.learning_rate, weight_decay=hparams.weight_decay)

    # -------------------------
    # Data loaders
    # -------------------------
    logger.info("正在加载数据集（dataset.load_audio）...")
    train_loader, valid_loader = dataset.load_audio(device=hparams.device)

    train_loader_kwargs = {
        "batch_size": hparams.batch_size,
        "num_workers": hparams.num_workers,
        "shuffle": True,
        "pin_memory": hparams.pin_memory,
        "drop_last": True,
    }

    valid_loader_kwargs = {
        "batch_size": hparams.batch_size,
        "num_workers": hparams.num_workers,
        "shuffle": False,
        "pin_memory": hparams.pin_memory,
        "drop_last": False,
    }

    # -------------------------
    # Checkpointer (recover BEFORE creating Brain where possible)
    # -------------------------
    logger.info("正在初始化检查点管理器（Checkpointer）...")
    recoverables = {"mdl": model}

    custom_hooks = {}
    if use_custom_hooks:
        custom_hooks = {
            "custom_save_hooks": {"mdl": custom_save_hook},
            "custom_load_hooks": {"mdl": custom_load_hook},
        }
        logger.info("已启用自定义检查点钩子")

    checkpointer = Checkpointer(checkpoints_dir=hparams.checkpoint_dir,
                                recoverables=recoverables, **custom_hooks)

    # Try recover from checkpointer directly (safe to do before creating Brain)
    try:
        recovered = checkpointer.recover_if_possible()
    except Exception as e:
        logger.warning(f"从检查点恢复时发生异常: {e}")
        recovered = None

    if recovered is not None:
        logger.info(f"已恢复的检查点信息: {recovered}")
    else:
        logger.info("未找到已有检查点或恢复失败，将从头开始训练。")

    # -------------------------
    # Epoch counter and run options
    # -------------------------
    epoch_counter = sb.utils.epoch_loop.EpochCounter(limit=hparams.n_epochs)

    run_opts = {"device": hparams.device, "auto_mix_prec": False}

    # -------------------------
    # SeparationBrain
    # -------------------------
    logger.info("正在初始化 SeparationBrain...")
    separator = SeparationBrain(
        train_loss=hparams.train_loss,
        modules={"mdl": model},
        opt_class=opt_class,
        run_opts=run_opts,
        checkpointer=checkpointer,
    )

    # If recovery happened above, ensure modules/optimizer states loaded into the Brain
    try:
        # Brain may expose a convenience method to recover; attempt it as well.
        if hasattr(separator.checkpointer, "recover_if_possible"):
            separator.checkpointer.recover_if_possible()
    except Exception as e:
        logger.warning(f"通过 Brain 的 checkpointer 恢复时出现异常: {e}")

    # -------------------------
    # Training loop
    # -------------------------
    logger.info("=" * 70)
    logger.info("开始训练")
    logger.info("=" * 70)

    try:
        separator.fit(
            epoch_counter=epoch_counter,
            train_set=train_loader,
            valid_set=valid_loader,
            train_loader_kwargs=train_loader_kwargs,
            valid_loader_kwargs=valid_loader_kwargs,
        )

        logger.info("训练已完成")

    except Exception as e:
        logger.exception("训练过程中发生异常")
        raise

    # -------------------------
    # Final checkpoint save (main process only)
    # -------------------------
    def save_final():
        try:
            final_ckpt = separator.checkpointer.save_checkpoint(name="FINAL")
            logger.info(f"最终模型检查点已保存: {final_ckpt}")
        except Exception as e:
            logger.exception(f"保存最终检查点失败: {e}")

        try:
            separator.checkpointer.save_and_keep_only(num_to_keep=hparams.keep_checkpoint_max)
            logger.info("已清理旧检查点")
        except Exception as e:
            logger.warning(f"清理旧检查点时出现问题: {e}")

    save_final()

    # List remaining checkpoints (main only)
    def list_ckpts():
        ckpts = separator.checkpointer.list_checkpoints()
        logger.info(f"Remaining checkpoints: {len(ckpts)}")
        for c in ckpts:
            logger.info(f"  - {c}")

    list_ckpts()

    return separator


# -------------------------
# Inference loader
# -------------------------
def load_trained_model(checkpoint_dir: str, hparams: HParams = None):
    if hparams is None:
        hparams = HParams()

    if not os.path.isdir(checkpoint_dir):
        raise ValueError(f"Checkpoint dir does not exist: {checkpoint_dir}")

    model = SimpleSeparator(fft_size=hparams.fft_size, hidden_size=hparams.hidden_size)

    checkpointer = Checkpointer(checkpoints_dir=checkpoint_dir, recoverables={"mdl": model})
    recovered = checkpointer.recover_if_possible()
    if recovered is None:
        raise ValueError(f"检查点目录不存在: {checkpoint_dir}")

    logger.info(f"Recovered checkpoint info: {recovered}")

    run_opts = {"device": hparams.device}
    separator = SeparationBrain(
        train_loss=hparams.train_loss,
        modules={"mdl": model},
        opt_class=lambda params: torch.optim.Adam(params, lr=hparams.learning_rate),
        run_opts=run_opts,
        checkpointer=checkpointer,
    )

    return separator


# -------------------------
# Utility: print checkpoint info
# -------------------------
def print_checkpoint_info(checkpoint_dir: str):
    checkpointer = Checkpointer(checkpoints_dir=checkpoint_dir, recoverables={})
    ckpts = checkpointer.list_checkpoints()
    logger.info("=" * 70)
    logger.info(f"CHECKPOINTS IN {checkpoint_dir}")
    logger.info("=" * 70)
    logger.info(f"Total checkpoints: {len(ckpts)}")
    for i, ck in enumerate(ckpts, 1):
        logger.info(f"{i}. {ck}")
        if hasattr(ck, "meta"):
            logger.info(f"   meta: {ck.meta}")
    logger.info("=" * 70)


# -------------------------
# Main
# -------------------------
if __name__ == "__main__":
    hparams = HParams()

    logger.info("Training with default SpeechBrain checkpoint hooks...")
    separator = train_separation_model(hparams, use_custom_hooks=True)

    # Uncomment to use custom hooks
    # separator = train_separation_model(hparams, use_custom_hooks=True)

    print_checkpoint_info(hparams.checkpoint_dir)

    # Example of loading for inference:
    # loaded = load_trained_model(checkpoint_dir=hparams.checkpoint_dir, hparams=hparams)
