"""
NavAssist: Object Detection Model Training Script
Optimized and cleaned version with improved efficiency
"""

import os
import yaml
import logging
from pathlib import Path
from typing import Dict, Optional
from ultralytics import YOLO

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TrainingConfig:
    """Configuration class for training parameters"""
    MODEL_NAME = 'yolov8n.pt'
    OUTPUT_NAME = 'navassist_custom_model'
    EPOCHS = 50
    BATCH_SIZE = 8
    IMAGE_SIZE = 640
    DEVICE = 'cpu'
    DATASET_PATH = 'blind-1/data.yaml'


def train_model(config: TrainingConfig = None) -> Dict:
    """Train YOLO model with specified configuration"""
    config = config or TrainingConfig()
    
    try:
        logger.info(f"Loading pretrained model: {config.MODEL_NAME}")
        model = YOLO(config.MODEL_NAME)
        
        logger.info("Starting Training...")
        logger.info(f"  Dataset: {config.DATASET_PATH}")
        logger.info(f"  Epochs: {config.EPOCHS} | Batch: {config.BATCH_SIZE}")
        logger.info(f"  Image Size: {config.IMAGE_SIZE} | Device: {config.DEVICE}")
        
        results = model.train(
            data=config.DATASET_PATH,
            epochs=config.EPOCHS,
            imgsz=config.IMAGE_SIZE,
            batch=config.BATCH_SIZE,
            name=config.OUTPUT_NAME,
            device=config.DEVICE,
            verbose=True
        )
        
        logger.info("✓ Training Complete!")
        logger.info(f"Model saved: runs/detect/{config.OUTPUT_NAME}/weights/best.pt")
        return results
    
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise


if __name__ == '__main__':
    train_model()