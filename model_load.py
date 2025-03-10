"""
YOLOモデルの読み込みと処理を行うモジュール
"""
import os
# NumPy互換性問題を解決する環境変数設定
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

# まずNumPyをインポート
import numpy as np
# 次にOpenCVをインポート
import cv2
# 次にPILをインポート
from PIL import Image
# 最後にUltralyticsをインポート
from ultralytics import YOLO

from google.cloud import storage
import tempfile
import io

def initialize_gcs_client() -> storage.Client:
    """
    Google Cloud Storageクライアントを初期化します。
    """
    try:
        service_account_path = os.path.join("config", "service_account.json")
        if os.path.exists(service_account_path):
            return storage.Client.from_service_account_json(service_account_path)
        else:
            import streamlit as st
            creds = st.secrets["gcp_service_account"]
            return storage.Client.from_service_account_info(creds)
    except Exception as e:
        print(f"GCS init error: {e}")
        return None

def download_model(storage_client: storage.Client, bucket: str, blob_name: str) -> str:
    """
    指定されたバケットからモデルをダウンロードします。
    """
    try:
        bucket_obj = storage_client.bucket(bucket)
        blob = bucket_obj.blob(blob_name)
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
            blob.download_to_filename(tmp.name)
            return tmp.name
    except Exception as e:
        print(f"Model download error: {e}")
        return None

def load_yolo_model() -> YOLO:
    """
    YOLOモデルをロードします。
    """
    client = initialize_gcs_client()
    if client:
        path = download_model(client, "yolo-v8-training", "trained_models/best.pt")
        if path:
            model = YOLO(path)
            os.unlink(path)  # 一時ファイルを削除
            return model
    return None 