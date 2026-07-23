import os
import json
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import wandb

from config import *
from utils import *
from dataset import MultimodalPersonalityDataset, collate_fn, frame_transform
from model import MultimodalPersonalityModel
from training import train_epoch, evaluate_multiclip, tukey_biweight_loss

# Code-1 LRs
BASE_LR_HEADS = 1e-4
BASE_LR_BACKB = 1e-5
WEIGHT_DECAY  = 1e-4

TRAIT_NAMES = ["Extraversion","Neuroticism","Agreeableness","Conscientiousness","Openness"]

# ---------- checkpoint helpers ----------
def save_ckpt(path, model, optimizer, scheduler, scaler, epoch, best_val, config):
    state = {
        "model_state": (model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()),
        "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
        "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
        "scaler_state": scaler.state_dict() if scaler is not None else None,
        "epoch": epoch,
        "best_val": best_val,
        "config": config,
    }
    torch.save(state, path)

def load_ckpt(path, model, optimizer=None, scheduler=None, scaler=None, strict=True, map_location="cpu"):
    ckpt = torch.load(path, map_location=map_location)
    model_state = ckpt.get("model_state", ckpt)
    (model.module if isinstance(model, nn.DataParallel) else model).load_state_dict(model_state, strict=strict)
    if optimizer is not None and ckpt.get("optimizer_state") is not None:
        optimizer.load_state_dict(ckpt["optimizer_state"])
    if scheduler is not None and ckpt.get("scheduler_state") is not None:
        scheduler.load_state_dict(ckpt["scheduler_state"])
    if scaler is not None and ckpt.get("scaler_state") is not None:
        scaler.load_state_dict(ckpt["scaler_state"])
    return ckpt.get("epoch", 0), ckpt.get("best_val", float("inf")), ckpt.get("config", {})

def _append_log(path, lines):
    with open(path, "a") as f:
        if isinstance(lines, (list, tuple)):
            for ln in lines: f.write(str(ln) + "\n")
        else:
            f.write(str(lines) + "\n")

def main(config=None):
    # wandb init
    wandb_mode = os.environ.get("WANDB_MODE", "")
    if wandb_mode != "disabled":
        try:
            wandb.init(project="multimodal-personality-recognition",
                       config=config if config else DEFAULT_CONFIG,
                       dir=WANDB_DIR,
                       mode=wandb_mode if wandb_mode else "online")
            print("W&B initialized")
        except Exception as e:
            print(f"[wandb] {e}; disabling")
            os.environ["WANDB_MODE"] = "disabled"

    # base config defaults (Code-1)
    if config is None: config = DEFAULT_CONFIG
    cfg = config
    cfg.setdefault("seq_len", 100)
    cfg.setdefault("batch_size", 4)
    cfg.setdefault("num_epochs", 12)
    cfg.setdefault("warmup_epochs", 1)
    cfg.setdefault("accumulation_steps", 1)
    cfg.setdefault("max_grad_norm", 1.0)
    cfg.setdefault("mc_clips", 7)
    cfg.setdefault("device", "cuda" if torch.cuda.is_available() else "cpu")
    cfg.setdefault("bert_path", "/home/su/SIJ/waqas/models/bert-base-uncased")
    cfg.setdefault("vision_w", 0.6)
    cfg.setdefault("audio_w", 0.3)
    cfg.setdefault("text_w",  0.1)
    cfg.setdefault("ckpt_every", 0)

    resume_from = cfg.get("resume_from", os.environ.get("RESUME_FROM", "")) or ""
    strict_load = bool(cfg.get("strict_load", os.environ.get("STRICT_LOAD", "0") == "1"))

    device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}, GPUs: {torch.cuda.device_count()}")

    exp_name = f"experiment_seq{cfg['seq_len']}_batch{cfg['batch_size']}_e{cfg['num_epochs']}"
    exp_dir  = os.path.join(EXPERIMENTS_DIR, exp_name)
    os.makedirs(exp_dir, exist_ok=True)

    log_path = os.path.join(exp_dir, "training_log.txt")
    with open(log_path, "w") as f:
        f.write("Training Log\n")
        f.write("=" * 60 + "\n")
        f.write(f"Experiment: {exp_name}\n")
        f.write(f"Started:   {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Config:\n{json.dumps(cfg, indent=2)}\n")
        f.write("=" * 60 + "\n")

    # data
    train_ann = filter_annotations(load_annotations(os.path.join(ANNOTATIONS_DIR, "annotation_training.pkl")))
    val_ann   = filter_annotations(load_annotations(os.path.join(ANNOTATIONS_DIR, "annotation_validation.pkl")))
    train_tr  = load_annotations(os.path.join(TRANSCRIPTIONS_DIR, "transcription_training.pkl"))
    val_tr    = load_annotations(os.path.join(TRANSCRIPTIONS_DIR, "transcription_validation.pkl"))

    train_vids = [vid + ".mp4" for vid in get_video_ids_from_folder(os.path.join(FRAMES_DIR, "train"))]
    val_vids   = [vid + ".mp4" for vid in get_video_ids_from_folder(os.path.join(FRAMES_DIR, "val"))]
    tr_ann = filter_annotations_for_all_traits(train_vids, train_ann)
    va_ann = filter_annotations_for_all_traits(val_vids,   val_ann)

    train_frame_dirs = [os.path.join(FRAMES_DIR, "train", d) for d in os.listdir(os.path.join(FRAMES_DIR, "train"))]
    val_frame_dirs   = [os.path.join(FRAMES_DIR, "val",   d) for d in os.listdir(os.path.join(FRAMES_DIR, "val"))]
    train_audio_paths = [os.path.join(AUDIO_DIR, "train", os.path.basename(d) + ".wav") for d in train_frame_dirs]
    val_audio_paths   = [os.path.join(AUDIO_DIR, "val",   os.path.basename(d) + ".wav") for d in val_frame_dirs]

    _append_log(log_path, [
        f"Num train samples: {len(train_frame_dirs)}",
        f"Num val   samples: {len(val_frame_dirs)}",
        f"Batch size: {cfg['batch_size']}",
        f"Seq len:    {cfg['seq_len']}",
        "-" * 60
    ])

    train_ds = MultimodalPersonalityDataset(train_frame_dirs, train_audio_paths, train_tr, tr_ann,
                                            seq_len=cfg["seq_len"], transform=frame_transform,
                                            device=device, is_train=True,  bert_path=cfg["bert_path"])
    val_ds   = MultimodalPersonalityDataset(val_frame_dirs, val_audio_paths, val_tr, va_ann,
                                            seq_len=cfg["seq_len"], transform=frame_transform,
                                            device=device, is_train=False, bert_path=cfg["bert_path"])

    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True,
                              num_workers=0, pin_memory=False, collate_fn=collate_fn)
    val_loader   = DataLoader(val_ds, batch_size=cfg["batch_size"], shuffle=False,
                              num_workers=0, pin_memory=False, collate_fn=collate_fn)

    # model
    model = MultimodalPersonalityModel(
        seq_len=cfg["seq_len"],
        bert_path=cfg["bert_path"],
        fusion_weights=(cfg["vision_w"], cfg["audio_w"], cfg["text_w"])
    ).to(device)
    if torch.cuda.device_count() > 1:
        print("Using multi-GPU (DataParallel)"); model = nn.DataParallel(model)
    m = model.module if isinstance(model, nn.DataParallel) else model

    # ======= FREEZE STRATEGY (UPDATED AS REQUESTED) =======
    # Freeze most ResNet; train layer3 + layer4
    for p in m.resnet.parameters(): p.requires_grad = False
    for p in m.resnet.layer3.parameters(): p.requires_grad = True
    for p in m.resnet.layer4.parameters(): p.requires_grad = True
    # Freeze most ViT; train last 3 blocks
    for p in m.vit.parameters(): p.requires_grad = False
    try:
        for p in m.vit.blocks[-3:].parameters(): p.requires_grad = True
    except AttributeError:
        for p in m.vit.transformer.encoder.layers[-3:].parameters(): p.requires_grad = True
    # Freeze BERT encoders + embeddings; keep pooler + text_mlp trainable
    for p in m.bert.embeddings.parameters(): p.requires_grad = False
    for p in m.bert.encoder.parameters():    p.requires_grad = False

    # optimizer
    heads = list(m.gru.parameters()) + list(m.audio_mlp.parameters()) + list(m.text_mlp.parameters()) + \
            list(m.vision_head.parameters()) + list(m.audio_head.parameters()) + list(m.text_head.parameters()) + \
            list(m.bert.pooler.parameters())
    # ===== backbones match the new freeze plan =====
    backbones = list(m.resnet.layer3.parameters()) + list(m.resnet.layer4.parameters())
    try:
        backbones += list(m.vit.blocks[-3:].parameters())
    except AttributeError:
        backbones += list(m.vit.transformer.encoder.layers[-3:].parameters())

    optimizer = torch.optim.AdamW([
        {"params": heads,     "lr": BASE_LR_HEADS, "weight_decay": WEIGHT_DECAY},
        {"params": backbones, "lr": BASE_LR_BACKB, "weight_decay": WEIGHT_DECAY/10.0},
    ])

    # warmup + cosine
    from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
    warm = max(1, int(cfg["warmup_epochs"]))
    total_epochs = int(cfg["num_epochs"])
    linear = LinearLR(optimizer, start_factor=0.1, total_iters=warm)
    cosine = CosineAnnealingLR(optimizer, T_max=max(1, total_epochs - warm))
    scheduler = SequentialLR(optimizer, schedulers=[linear, cosine], milestones=[warm])

    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))
    criterion = tukey_biweight_loss

    # checkpoints
    best_val = float("inf")
    start_epoch = 0
    best_weights = os.path.join(exp_dir, "best_model.pth")
    best_ckpt    = os.path.join(exp_dir, "best_ckpt.pth")
    last_ckpt    = os.path.join(exp_dir, "last_ckpt.pth")
    ckpt_dir     = os.path.join(exp_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    # resume
    if resume_from and os.path.isfile(resume_from):
        try:
            ep, best_loaded, _ = load_ckpt(resume_from, model, optimizer, scheduler, scaler,
                                           strict=strict_load, map_location=device)
            start_epoch = ep
            if best_loaded is not None: best_val = best_loaded
            _append_log(log_path, f"[RESUME] Loaded '{resume_from}' at epoch {start_epoch}, best_val={best_val:.6f}")
            print(f"[RESUME] epoch={start_epoch}, best_val={best_val:.6f}")
        except Exception as e:
            _append_log(log_path, f"[RESUME] Failed: {e}")
            print(f"[RESUME] Failed to load: {e}")

    # train
    for epoch in range(start_epoch + 1, total_epochs + 1):
        print(f"\n==> Epoch {epoch}/{total_epochs} | LRs: {[g['lr'] for g in optimizer.param_groups]}")
        tr_loss, tr_macc, tr_tacc, *_ = train_epoch(
            model, train_loader, criterion, optimizer, device, scaler,
            accumulation_steps=cfg.get("accumulation_steps", 1),
            max_grad_norm=cfg.get("max_grad_norm", 1.0)
        )
        scheduler.step()

        va_loss, va_macc, va_tacc = evaluate_multiclip(
            model, val_ds, val_loader, criterion, device, clips=cfg.get("mc_clips", 5)
        )

        print(f"Train Loss: {tr_loss:.4f} | Train Mean Acc: {tr_macc:.4f}")
        for n, a in zip(TRAIT_NAMES, tr_tacc): print(f"  Train {n}: {a:.4f}")
        print(f"Val   Loss: {va_loss:.4f} | Val   Mean Acc: {va_macc:.4f}")
        for n, a in zip(TRAIT_NAMES, va_tacc): print(f"  Val   {n}: {a:.4f}")

        _append_log(log_path, [
            f"Epoch {epoch}/{total_epochs}",
            f"  Train Loss: {tr_loss:.6f}",
            f"  Train Mean Acc: {tr_macc:.6f}",
            "  Train Trait Accuracies:",
        ])
        for n, a in zip(TRAIT_NAMES, tr_tacc): _append_log(log_path, f"    {n}: {a:.6f}")
        _append_log(log_path, [
            f"  Val   Loss: {va_loss:.6f}",
            f"  Val   Mean Acc: {va_macc:.6f}",
            "  Val   Trait Accuracies:",
        ])
        for n, a in zip(TRAIT_NAMES, va_tacc): _append_log(log_path, f"    {n}: {a:.6f}")
        try:
            _append_log(log_path, f"  LRs -> heads: {optimizer.param_groups[0]['lr']:.6e} | backbones: {optimizer.param_groups[1]['lr']:.6e}")
        except Exception:
            pass
        _append_log(log_path, "-" * 60)

        if os.environ.get("WANDB_MODE", "") != "disabled":
            wandb.log({
                "epoch": epoch,
                "train_loss": tr_loss, "val_loss": va_loss,
                "train_mean_acc": tr_macc, "val_mean_acc": va_macc,
                "lr_heads": optimizer.param_groups[0]["lr"],
                "lr_backb": optimizer.param_groups[1]["lr"],
            })
            for n, a in zip(TRAIT_NAMES, tr_tacc): wandb.log({f"train_{n.lower()}_acc": a})
            for n, a in zip(TRAIT_NAMES, va_tacc): wandb.log({f"val_{n.lower()}_acc": a})

        # checkpoints
        save_ckpt(last_ckpt, model, optimizer, scheduler, scaler, epoch, best_val, cfg)
        if cfg.get("ckpt_every", 0) and (epoch % cfg["ckpt_every"] == 0):
            ep_ckpt = os.path.join(ckpt_dir, f"ckpt_epoch_{epoch:03d}.pth")
            save_ckpt(ep_ckpt, model, optimizer, scheduler, scaler, epoch, best_val, cfg)
            _append_log(log_path, f"[CKPT] Saved epoch checkpoint -> {ep_ckpt}")

        if va_loss < best_val:
            best_val = va_loss
            torch.save(m.state_dict(), best_weights)
            save_ckpt(best_ckpt, model, optimizer, scheduler, scaler, epoch, best_val, cfg)
            print(f"[SAVE] New BEST -> weights: {best_weights} | ckpt: {best_ckpt}")
            _append_log(log_path, f"[SAVE] New BEST -> weights: {best_weights} | ckpt: {best_ckpt}")

    # final saves
    final_weights = os.path.join(exp_dir, "final_model.pth")
    torch.save(m.state_dict(), final_weights)
    save_ckpt(last_ckpt, model, optimizer, scheduler, scaler, total_epochs, best_val, cfg)

    print(f"[SAVE] Final weights -> {final_weights}")
    _append_log(log_path, [
        "=" * 60, "TRAINING SUMMARY", "=" * 60,
        f"Best Validation Loss: {best_val:.6f}",
        f"Final weights path: {final_weights}",
        f"Best weights path:  {best_weights}",
        f"Best ckpt path:     {best_ckpt}",
        f"Last ckpt path:     {last_ckpt}",
    ])

    if os.environ.get("WANDB_MODE", "") != "disabled":
        wandb.finish()

if __name__ == "__main__":
    main()
