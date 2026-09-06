import os
import sys
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np
import torch
from tqdm import tqdm

project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from config import TrainConfig, ArtifactUploadConfig
from dataset import FishMultimodalDataLoader
from models import LiteFFIANet

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def extract_multimodal_features(
    model: torch.nn.Module,
    data_loader: Any,
    output_dir: str,
    device: torch.device
) -> Dict[str, str]:
    """
    Extracts all 3 feature groups (Spatial, Motion, Audio Frequency & Rhythm, and Fused MBT tokens)
    and persists them as .npy arrays.
    """
    os.makedirs(output_dir, exist_ok=True)
    model.eval()
    model.to(device)

    features = {
        'f_spatial': [],
        'f_motion': [],
        'f_frequency': [],
        'f_rhythm': [],
        'f_fused': [],
        'clipwise_output': [],
        'gating_alpha': [],
        'targets': []
    }

    logger.info("Extracting multimodal features across dataset...")
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Extracting features"):
            video = batch['video_form'].to(device)
            audio = batch['audio_form'].to(device)
            targets = batch['target']

            out = model(video, audio)

            for key in ['f_spatial', 'f_motion', 'f_frequency', 'f_rhythm', 'f_fused', 'clipwise_output', 'gating_alpha']:
                features[key].append(out[key].detach().cpu().numpy())
            features['targets'].append(targets.numpy() if hasattr(targets, 'numpy') else np.array(targets))

    saved_paths = {}
    for key, data_list in features.items():
        if len(data_list) == 0:
            logger.warning(f"No samples extracted for {key}. Skipping save.")
            continue
        arr = np.concatenate(data_list, axis=0)
        file_path = os.path.join(output_dir, f"{key}.npy")
        np.save(file_path, arr)
        saved_paths[key] = file_path
        logger.info(f"Saved {key} features to: '{file_path}' (Shape: {arr.shape})")

    return saved_paths


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Multimodal Feature Extractor for LiteFFIA-Net")
    parser.add_argument("--config", type=str, default=None, help="Path to train_config.json")
    parser.add_argument("--output-dir", type=str, default="outputs/features", help="Output directory for extracted .npy features")
    parser.add_argument("--checkpoint", type=str, default=None, help="Optional model checkpoint path")
    args = parser.parse_args()

    pkg_dir = Path(__file__).resolve().parent.parent
    cfg_path = args.config if args.config else str(pkg_dir / "config" / "train_config.json")
    train_cfg = TrainConfig.from_json(cfg_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = LiteFFIANet(
        classes_num=train_cfg.model.classes_num,
        embed_dim=train_cfg.model.embed_dim,
        num_bottlenecks=train_cfg.model.num_bottlenecks,
        num_heads=train_cfg.model.num_heads,
        pretrained_video=train_cfg.model.pretrained_video
    )
    if args.checkpoint and os.path.isfile(args.checkpoint):
        ckpt = torch.load(args.checkpoint, map_location=device)
        state_dict = ckpt.get("model_state_dict", ckpt)
        model.load_state_dict(state_dict)
        logger.info(f"Loaded weights from checkpoint: {args.checkpoint}")
        
    loader_manager = FishMultimodalDataLoader(
        batch_size=train_cfg.batch_size,
        dataloader_workers=0,
        image_size=train_cfg.video_features.image_size,
        frame_policy=train_cfg.video_features.frame_policy,
        num_frames=train_cfg.video_features.num_frames,
        sample_rate=train_cfg.audio_features.sample_rate,
        splitter_config=train_cfg.dataset_splitter
    )
    
    test_loader = loader_manager.get_data_loader("test", shuffle=False)
    out_paths = extract_multimodal_features(model, test_loader, args.output_dir, device)
    print("Feature extraction completed successfully.")
