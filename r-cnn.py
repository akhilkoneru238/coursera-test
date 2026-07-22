"""
Faster/Mask R-CNN training for the crater dataset (single class, dense small objects).

Tuned for this data: small anchors, high max detections per image, raised RPN
proposal caps, grayscale->3ch. Includes early stopping, a wall-clock time budget
(for Kaggle's session limit), and resume.

Examples
--------
Single GPU, train with early stopping + a 11.5h budget:
    python train_rcnn.py \
        --train-images /kaggle/input/graphted_dataset/graphted_dataset/images/train \
        --val-images   /kaggle/input/graphted_dataset/graphted_dataset/images/val \
        --epochs 50 --patience 6 --max-hours 11.5 --out /kaggle/working/rcnn_runs

Resume next session (after adding the previous output as an input dataset):
    python train_rcnn.py ... --resume /kaggle/input/<prev-output>/rcnn_runs/last_frcnn.pt

Multi-GPU:
    torchrun --nproc_per_node=2 train_rcnn.py ...
"""
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import argparse
import glob
import math
import random
import time

import cv2
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torchmetrics.detection import MeanAveragePrecision
from torchvision.models.detection import (
    FasterRCNN_ResNet50_FPN_V2_Weights,
    MaskRCNN_ResNet50_FPN_V2_Weights,
    fasterrcnn_resnet50_fpn_v2,
    maskrcnn_resnet50_fpn_v2,
)
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor


def label_path_for(img_path):
    p = img_path.replace("/images/", "/labels/")
    return os.path.splitext(p)[0] + ".txt"


def load_yolo_boxes(txt, w, h):
    if not os.path.exists(txt):
        return np.zeros((0, 4), np.float32)
    out = []
    with open(txt) as f:
        for lin in f:
            parts = lin.split()
            if len(parts) < 5:
                continue
            cx, cy, bw, bh = map(float, parts[1:5])
            x1 = max((cx - bw / 2) * w, 0.0)
            y1 = max((cy - bh / 2) * h, 0.0)
            x2 = min((cx + bw / 2) * w, w - 1.0)
            y2 = min((cy + bh / 2) * h, h - 1.0)
            if x2 - x1 >= 2.0 and y2 - y1 >= 2.0:
                out.append([x1, y1, x2, y2])
    return np.asarray(out, np.float32).reshape(-1, 4)


def rot90_boxes(boxes, k, w, h):
    # np.rot90 rotates counter-clockwise: a point (x, y) in a W x H tile maps to
    # (y, w-1-x), so a box [x1, y1, x2, y2] -> [y1, w-x2, y2, w-x1]. Both vertical
    # terms use the CURRENT width w; (w, h) swap after each 90 deg step.
    for _ in range(k % 4):
        if len(boxes):
            x1, y1 = boxes[:, 0].copy(), boxes[:, 1].copy()
            x2, y2 = boxes[:, 2].copy(), boxes[:, 3].copy()
            boxes = np.stack([y1, w - x2, y2, w - x1], axis=1)
        w, h = h, w
    return boxes


def augment(img, boxes):
    h, w = img.shape
    if random.random() < 0.5:
        img = img[:, ::-1]
        if len(boxes):
            boxes[:, [0, 2]] = w - boxes[:, [2, 0]]
    if random.random() < 0.5:
        img = img[::-1, :]
        if len(boxes):
            boxes[:, [1, 3]] = h - boxes[:, [3, 1]]
    k = random.randint(0, 3)
    if k:
        img = np.rot90(img, k)
        boxes = rot90_boxes(boxes, k, w, h)
    return np.ascontiguousarray(img), boxes


def ellipse_masks(boxes, h, w):
    m = np.zeros((len(boxes), h, w), np.uint8)
    for i, (x1, y1, x2, y2) in enumerate(boxes):
        c = (int(round((x1 + x2) / 2)), int(round((y1 + y2) / 2)))
        ax = (max(int((x2 - x1) / 2), 1), max(int((y2 - y1) / 2), 1))
        cv2.ellipse(m[i], c, ax, 0, 0, 360, 1, -1)
    return torch.from_numpy(m)


class CraterTiles(Dataset):
    def __init__(self, img_dir, train, use_masks):
        self.paths = sorted(glob.glob(os.path.join(img_dir, "*.png")))
        if not self.paths:
            raise FileNotFoundError(f"no .png files under {img_dir}")
        self.train = train
        self.use_masks = use_masks

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = None
        for _ in range(5):
            img = cv2.imread(self.paths[idx], cv2.IMREAD_GRAYSCALE)
            if img is not None:
                break
            idx = random.randrange(len(self.paths))
        if img is None:
            raise IOError("too many IO errors in a row")
        h, w = img.shape
        boxes = load_yolo_boxes(label_path_for(self.paths[idx]), w, h)
        if self.train:
            img, boxes = augment(img, boxes)
            h, w = img.shape
        x = torch.from_numpy(np.repeat(img[..., None], 3, axis=2))
        x = x.permute(2, 0, 1).float() / 255.0
        target = {
            "boxes": torch.from_numpy(boxes),
            "labels": torch.ones((len(boxes),), dtype=torch.int64),
            "image_id": torch.tensor([idx]),
        }
        if self.use_masks:
            target["masks"] = ellipse_masks(boxes, h, w)
        return x, target


def collate(batch):
    return tuple(zip(*batch))


def build_model(args):
    anchors = AnchorGenerator(
        sizes=((8,), (16,), (32,), (64,), (128,)),
        aspect_ratios=((0.75, 1.0, 1.33),) * 5,
    )
    kw = dict(
        rpn_anchor_generator=anchors,
        min_size=args.imgsz,
        max_size=args.imgsz,
        box_detections_per_img=args.max_det,
        box_nms_thresh=args.nms,
        box_score_thresh=0.05,
        rpn_pre_nms_top_n_train=4000,
        rpn_post_nms_top_n_train=2000,
        rpn_pre_nms_top_n_test=3000,
        rpn_post_nms_top_n_test=1500,
        rpn_batch_size_per_image=512,
        box_batch_size_per_image=1024,
        trainable_backbone_layers=args.trainable_layers,
    )
    if args.arch == "frcnn":
        model = fasterrcnn_resnet50_fpn_v2(weights=FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT, **kw)
    else:
        model = maskrcnn_resnet50_fpn_v2(weights=MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT, **kw)
    in_f = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_f, num_classes=2)
    if args.arch == "maskrcnn":
        in_m = model.roi_heads.mask_predictor.conv5_mask.in_channels
        model.roi_heads.mask_predictor = MaskRCNNPredictor(in_m, 256, num_classes=2)
    return model


def warmup_cosine(opt, total_iters, warmup_iters):
    def fn(i):
        if i < warmup_iters:
            return (i + 1) / max(warmup_iters, 1)
        p = (i - warmup_iters) / max(total_iters - warmup_iters, 1)
        return 0.5 * (1 + math.cos(math.pi * p))

    return torch.optim.lr_scheduler.LambdaLR(opt, fn)


def train_one_epoch(model, loader, opt, scaler, sched, device, ep, epochs, rank):
    model.train()
    running, n, t0 = 0.0, 0, time.time()
    for it, (images, targets) in enumerate(loader):
        images = [im.to(device) for im in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        with torch.autocast("cuda", dtype=torch.float16):
            loss_dict = model(images, targets)
            loss = sum(loss_dict.values())
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        scaler.step(opt)
        scaler.update()
        sched.step()
        running += loss.item()
        n += 1
        if rank == 0 and it % 50 == 0:
            elapsed = max(time.time() - t0, 1e-6)
            ips = (n * loader.batch_size) / elapsed
            comp = " ".join(f"{k.replace('loss_', '')}={v.item():.3f}" for k, v in loss_dict.items())
            print(
                f"  ep {ep + 1}/{epochs} it {it}/{len(loader)} "
                f"loss={loss.item():.3f} ({comp}) "
                f"{ips:.1f} img/s lr={opt.param_groups[0]['lr']:.2e}",
                flush=True,
            )
    return running / max(1, n)


@torch.no_grad()
def evaluate(model, loader, max_det, device):
    model.eval()
    metric = MeanAveragePrecision(
        iou_type="bbox",
        iou_thresholds=[0.5],
        max_detection_thresholds=[1, 100, max_det],
    )
    for images, targets in loader:
        images = [im.to(device, non_blocking=True) for im in images]
        with torch.autocast("cuda", dtype=torch.float16):
            preds = model(images)
        p_cpu = [
            {
                "boxes": p["boxes"].float().cpu(),
                "scores": p["scores"].float().cpu(),
                "labels": p["labels"].cpu(),
            }
            for p in preds
        ]
        t_cpu = [{"boxes": t["boxes"], "labels": t["labels"]} for t in targets]
        metric.update(p_cpu, t_cpu)
    res = metric.compute()
    keep = ("map_50", "mar_100", f"mar_{max_det}")
    return {k: float(v) for k, v in res.items() if k in keep}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-images", required=True)
    ap.add_argument("--val-images", required=True)
    ap.add_argument("--arch", choices=["frcnn", "maskrcnn"], default="frcnn")
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--max-det", type=int, default=1500)
    ap.add_argument("--nms", type=float, default=0.5)
    ap.add_argument("--trainable-layers", type=int, default=3)
    ap.add_argument("--warmup-iters", type=int, default=500)
    ap.add_argument("--patience", type=int, default=0,
                    help="stop if val map_50 does not improve for this many epochs (0 = disabled)")
    ap.add_argument("--min-delta", type=float, default=0.0,
                    help="minimum map_50 gain to count as an improvement for patience")
    ap.add_argument("--max-hours", type=float, default=0.0,
                    help="stop after this many wall-clock hours and save (0 = no limit)")
    ap.add_argument("--resume", default="",
                    help="path to a last_*.pt checkpoint to resume from (model+opt+sched+epoch)")
    ap.add_argument("--out", default="/kaggle/working/rcnn_runs")
    ap.add_argument("--seed", type=int, default=0)
    args, _ = ap.parse_known_args()

    ddp = "RANK" in os.environ
    if ddp:
        dist.init_process_group("nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        rank = dist.get_rank()
    else:
        local_rank, rank = 0, 0
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    random.seed(args.seed + rank)
    np.random.seed(args.seed + rank)
    torch.manual_seed(args.seed + rank)

    use_masks = args.arch == "maskrcnn"
    train_ds = CraterTiles(args.train_images, train=True, use_masks=use_masks)
    val_ds = CraterTiles(args.val_images, train=False, use_masks=False)
    if rank == 0:
        print(f"train tiles: {len(train_ds)} val tiles: {len(val_ds)}", flush=True)

    sampler = DistributedSampler(train_ds) if ddp else None
    train_ld = DataLoader(
        train_ds,
        batch_size=args.batch,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=args.workers,
        collate_fn=collate,
        pin_memory=True,
        persistent_workers=args.workers > 0,
        drop_last=True,
    )
    val_ld = DataLoader(
        val_ds,
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.workers,
        collate_fn=collate,
        pin_memory=True,
    )

    model = build_model(args).to(device)
    model_eval = model

    # resume: load weights before wrapping in DDP; opt/sched loaded after they exist
    start_epoch, best, epochs_no_improve = 0, 0.0, 0
    ckpt_state = None
    if args.resume and os.path.exists(args.resume):
        ckpt_state = torch.load(args.resume, map_location=device, weights_only=False)
        model_eval.load_state_dict(ckpt_state["model"])
        start_epoch = ckpt_state.get("epoch", -1) + 1
        best = ckpt_state.get("best", 0.0)
        epochs_no_improve = ckpt_state.get("epochs_no_improve", 0)
        if rank == 0:
            print(f"resumed from {args.resume}: start_epoch={start_epoch} "
                  f"best_map_50={best:.4f}", flush=True)

    if ddp:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    sched = warmup_cosine(opt, args.epochs * len(train_ld), args.warmup_iters)
    scaler = torch.amp.GradScaler("cuda")
    if ckpt_state is not None:
        opt.load_state_dict(ckpt_state["optimizer"])
        sched.load_state_dict(ckpt_state["scheduler"])
        scaler.load_state_dict(ckpt_state["scaler"])

    os.makedirs(args.out, exist_ok=True)
    t_start = time.time()
    for ep in range(start_epoch, args.epochs):
        if sampler is not None:
            sampler.set_epoch(ep)
        avg = train_one_epoch(model, train_ld, opt, scaler, sched, device, ep, args.epochs, rank)
        stop = torch.zeros(1, device=device)  # early-stop / time-budget flag, shared across ranks
        if rank == 0:
            stats = evaluate(model_eval, val_ld, args.max_det, device)
            line = " ".join(f"{k}={v:.4f}" for k, v in stats.items())
            print(f"epoch {ep + 1}/{args.epochs} loss={avg:.3f} {line}", flush=True)

            cur = stats.get("map_50", 0.0)
            improved = cur > best + args.min_delta  # counts toward patience
            if cur > best:                          # always keep the literal best
                best = cur
                ckpt = os.path.join(args.out, f"best_{args.arch}.pt")
                torch.save(
                    {"model": model_eval.state_dict(), "map_50": best, "args": vars(args)},
                    ckpt,
                )
                print(f"new best map_50={best:.4f} saved to {ckpt}", flush=True)
            if improved:
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                print(
                    f"no map_50 improvement for {epochs_no_improve} epoch(s) (best={best:.4f})",
                    flush=True,
                )
                if args.patience > 0 and epochs_no_improve >= args.patience:
                    stop[0] = 1.0
                    print(
                        f"early stopping: val map_50 stalled for {args.patience} "
                        f"epochs; best map_50={best:.4f}",
                        flush=True,
                    )

            # always save latest full state -> a session kill loses at most one epoch
            last = os.path.join(args.out, f"last_{args.arch}.pt")
            torch.save(
                {
                    "model": model_eval.state_dict(),
                    "optimizer": opt.state_dict(),
                    "scheduler": sched.state_dict(),
                    "scaler": scaler.state_dict(),
                    "epoch": ep,
                    "best": best,
                    "epochs_no_improve": epochs_no_improve,
                    "args": vars(args),
                },
                last,
            )
            # wall-clock budget: stop cleanly BEFORE Kaggle kills the GPU session
            if args.max_hours > 0 and (time.time() - t_start) / 3600.0 >= args.max_hours:
                stop[0] = 1.0
                print(
                    f"time budget of {args.max_hours}h reached; stopping after "
                    f"epoch {ep + 1}. Resume next session with --resume {last}",
                    flush=True,
                )

        if ddp:
            dist.broadcast(stop, src=0)  # every rank learns rank-0's decision
            dist.barrier()
        if stop.item() > 0:
            break

    if ddp:
        dist.destroy_process_group()
    if rank == 0:
        print(f"done. best map_50={best:.4f}", flush=True)


if __name__ == "__main__":
    main()
