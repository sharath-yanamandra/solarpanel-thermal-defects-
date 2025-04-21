from dotenv import load_dotenv

load_dotenv()

class Config:
    
    DETECTION_MODEL_PATH = 'train/weights/best.pt'
    # Processing Configuration
    MAX_RETRIES = 3
    PROCESSING_TIMEOUT = 3600 # 1 hour in seconds
    FRAMES_OUTPUT_DIR = 'frames'

    TRACKING_THRESHOLD = 5
    MAX_AGE = 50
    MIN_MA = 1