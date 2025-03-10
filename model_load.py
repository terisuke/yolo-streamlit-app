"""
YOLOモデルの読み込みと処理を行うモジュール
"""
import os
import sys
import traceback
# NumPy互換性問題を解決する環境変数設定
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

# まずNumPyをインポート
try:
    print("Importing numpy...")
    import numpy as np
    print(f"NumPy version: {np.__version__}")
except Exception as e:
    print(f"NumPy import error: {e}")
    traceback.print_exc()

# 次にOpenCVをインポート
try:
    print("Importing OpenCV...")
    import cv2
    print(f"OpenCV version: {cv2.__version__}")
except Exception as e:
    print(f"OpenCV import error: {e}")
    traceback.print_exc()

# 次にPILをインポート
try:
    print("Importing PIL...")
    from PIL import Image
    print(f"PIL version: {Image.__version__}")
except Exception as e:
    print(f"PIL import error: {e}")
    traceback.print_exc()

# 最後にUltralyticsをインポート
try:
    print("Importing YOLO...")
    from ultralytics import YOLO
    print("YOLO imported successfully")
except Exception as e:
    print(f"YOLO import error: {e}")
    traceback.print_exc()

try:
    print("Importing GCS...")
    from google.cloud import storage
    print("GCS imported successfully")
except Exception as e:
    print(f"GCS import error: {e}")
    traceback.print_exc()

import tempfile
import io

def initialize_gcs_client() -> storage.Client:
    """
    Google Cloud Storageクライアントを初期化します。
    """
    try:
        print("Initializing GCS client...")
        service_account_path = os.path.join("config", "service_account.json")
        if os.path.exists(service_account_path):
            print(f"Using service account file: {service_account_path}")
            return storage.Client.from_service_account_json(service_account_path)
        else:
            print("Service account file not found, using Streamlit secrets")
            import streamlit as st
            creds = st.secrets["gcp_service_account"]
            return storage.Client.from_service_account_info(creds)
    except Exception as e:
        print(f"GCS init error: {e}")
        traceback.print_exc()
        return None

def download_model(storage_client: storage.Client, bucket: str, blob_name: str) -> str:
    """
    指定されたバケットからモデルをダウンロードします。
    """
    try:
        print(f"Downloading model from bucket: {bucket}, blob: {blob_name}")
        bucket_obj = storage_client.bucket(bucket)
        blob = bucket_obj.blob(blob_name)
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
            print(f"Downloading to temporary file: {tmp.name}")
            blob.download_to_filename(tmp.name)
            print(f"Download complete: {os.path.getsize(tmp.name)} bytes")
            return tmp.name
    except Exception as e:
        print(f"Model download error: {e}")
        traceback.print_exc()
        return None

def load_yolo_model() -> YOLO:
    """
    YOLOモデルをロードします。
    """
    try:
        print("Starting YOLO model load process")
        client = initialize_gcs_client()
        if client:
            print("GCS client initialized successfully")
            path = download_model(client, "yolo-v8-training", "trained_models/best.pt")
            if path:
                print(f"Model downloaded to {path}, loading YOLO...")
                model = YOLO(path)
                print("YOLO model loaded successfully")
                os.unlink(path)  # 一時ファイルを削除
                print(f"Temporary file {path} deleted")
                return model
            else:
                print("Failed to download model")
        else:
            print("Failed to initialize GCS client")
    except Exception as e:
        print(f"Error in load_yolo_model: {e}")
        traceback.print_exc()
    return None 