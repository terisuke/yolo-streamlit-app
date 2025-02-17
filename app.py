import streamlit as st
import os
import json
import tempfile
from google.cloud import storage
from ultralytics import YOLO
import shutil
import numpy as np
from PIL import Image
import io

# アプリケーションのタイトルを設定
st.title("YOLO セグメンテーションアプリ")
st.write("Google Cloud Storageのモデルを使用して画像のセグメンテーションを行います")

# セッション状態の初期化
if 'model' not in st.session_state:
    st.session_state.model = None

def initialize_gcs_client():
    """Google Cloud Storageクライアントの初期化"""
    try:
        # サービスアカウントJSONファイルのパスを指定
        service_account_path = os.path.join('config', 'service_account.json')
        
        if os.path.exists(service_account_path):
            # JSONファイルから認証情報を読み込む
            storage_client = storage.Client.from_service_account_json(service_account_path)
        else:
            # バックアップとしてStreamlit Secretsを使用
            credentials_info = st.secrets["gcp_service_account"]
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
                "yolo-v8-training",
                "trained_models/best.pt"
            )
            if model_path:
                st.session_state.model = YOLO(model_path)
                os.unlink(model_path)  # 一時ファイルを削除

def process_image(image_file):
    """画像の処理と推論の実行"""
    try:
        # 画像をバイトデータとして読み込む
        image_bytes = image_file.getvalue()
        
        # 一時ファイルを作成
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp_image_file:
            temp_image_file.write(image_bytes)
            temp_image_file.close()
            
            # 推論の実行（plot=Trueで結果を描画）
            results = st.session_state.model(temp_image_file.name, task='segment')
            
            # 結果の画像を取得
            result_image = results[0].plot()  # この行で検出結果が描画された画像を取得
            
            # NumPy配列をPIL Imageに変換
            if isinstance(result_image, np.ndarray):
                result_image = Image.fromarray(result_image[..., ::-1])  # BGR to RGB
            
            # 一時ファイルの削除
            os.unlink(temp_image_file.name)
            
            return result_image
            
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
        st.image(uploaded_file, caption="アップロードされた画像", use_container_width=True)
        
        # 進捗状態を表示
        with st.spinner('セグメンテーションを実行中...'):
            # 推論の実行と結果の表示
            result_image = process_image(uploaded_file)
            
            if result_image is not None:
                st.image(result_image, caption="セグメンテーション結果", use_container_width=True)
                
                # 結果の画像をダウンロード可能にする
                buf = io.BytesIO()
                result_image.save(buf, format="PNG")
                st.download_button(
                    label="結果をダウンロード",
                    data=buf.getvalue(),
                    file_name="segmentation_result.png",
                    mime="image/png"
                )

if __name__ == "__main__":
    main()