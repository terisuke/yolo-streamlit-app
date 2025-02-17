# app.py
import streamlit as st
import os
import tempfile
from google.cloud import storage
from ultralytics import YOLO
import shutil

# アプリケーションのタイトルを設定
st.title("YOLO セグメンテーションアプリ")
st.write("Google Cloud Storageのモデルを使用して画像のセグメンテーションを行います")

# セッション状態の初期化
if 'model' not in st.session_state:
    st.session_state.model = None

def initialize_gcs_client():
    """Google Cloud Storageクライアントの初期化"""
    try:
        # Streamlit Secretsから認証情報を取得
        credentials_info = st.secrets["gcp_service_account"]
        # GCSクライアントの初期化
        storage_client = storage.Client.from_service_account_info(credentials_info)
        return storage_client
    except Exception as e:
        st.error(f"GCSクライアントの初期化に失敗しました: {str(e)}")
        return None

def download_model_from_gcs(storage_client, bucket_name, blob_name):
    """GCSからモデルをダウンロード"""
    try:
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as temp_model_file:
            blob.download_to_filename(temp_model_file.name)
            return temp_model_file.name
    except Exception as e:
        st.error(f"モデルのダウンロードに失敗しました: {str(e)}")
        return None

def load_yolo_model():
    """YOLOモデルのロード"""
    if st.session_state.model is None:
        storage_client = initialize_gcs_client()
        if storage_client:
            model_path = download_model_from_gcs(
                storage_client,
                "your-bucket-name",  # あなたのバケット名に変更してください
                "trained_models/best.pt"  # モデルのパスを適切に設定してください
            )
            if model_path:
                st.session_state.model = YOLO(model_path)
                os.unlink(model_path)  # 一時ファイルを削除

def process_image(image_file):
    """画像の処理と推論の実行"""
    try:
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp_image_file:
            temp_image_file.write(image_file.getvalue())
            temp_image_file.close()
            
            # 推論の実行
            results = st.session_state.model(temp_image_file.name, save=True)
            
            # 一時ファイルの削除
            os.unlink(temp_image_file.name)
            
            return results
    except Exception as e:
        st.error(f"画像の処理中にエラーが発生しました: {str(e)}")
        return None

def main():
    # モデルのロード
    load_yolo_model()
    
    if st.session_state.model is None:
        st.warning("モデルのロードに失敗しました。設定を確認してください。")
        return
    
    # 画像のアップロード
    uploaded_file = st.file_uploader("セグメンテーションする画像をアップロードしてください", type=["jpg", "jpeg", "png"])
    
    if uploaded_file:
        # 元の画像を表示
        st.image(uploaded_file, caption="アップロードされた画像", use_column_width=True)
        
        # 推論の実行
        results = process_image(uploaded_file)
        
        if results:
            # 結果の表示
            if os.path.exists("runs/detect/predict"):
                for file in os.listdir("runs/detect/predict"):
                    if file.endswith((".jpg", ".jpeg", ".png")):
                        result_image_path = os.path.join("runs", "detect", "predict", file)
                        st.image(result_image_path, caption="セグメンテーション結果", use_column_width=True)
                
                # 後片付け
                shutil.rmtree("runs")

if __name__ == "__main__":
    main()