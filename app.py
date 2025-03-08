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
import cv2  # Newly added import

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

def mm_to_pixels(mm: float, dpi: float = 300.0, scale: float = 0.01) -> int:
    """
    Convert a real-world length (in mm) to the number of pixels given the DPI and scale.
    scale=0.01 means 1:100 (1 mm in the diagram is 100 mm in real life).
    """
    inch = mm / 25.4  # 1 inch = 25.4 mm
    px = inch * dpi
    # Apply scale factor for 1:100
    px *= (1.0 / scale)
    return int(round(px))

def offset_mask_by_distance(mask: np.ndarray, offset_px: int) -> np.ndarray:
    """
    Given a binary segmentation mask and an offset in pixels,
    shrink the mask inward by 'offset_px' using distance transform.
    """
    # Ensure mask is binary (0 or 1)
    binary_mask = (mask > 0).astype(np.uint8)

    # Distance transform (L2)
    dist = cv2.distanceTransform(binary_mask, cv2.DIST_L2, 5)

    # Keep only points where distance >= offset_px
    shrunk_mask = (dist >= offset_px).astype(np.uint8)
    return shrunk_mask

def draw_910mm_grid(
    image: np.ndarray,
    shrunk_mask: np.ndarray,
    cell_mm: float = 910.0,
    dpi: float = 300.0,
    scale: float = 0.01
) -> np.ndarray:
    """
    Draw 910mm (半畳) grid lines on the image, but only in regions
    where shrunk_mask indicates valid buildable area (1).

    Args:
        image (np.ndarray): The original result image (BGR or after YOLO plot).
        shrunk_mask (np.ndarray): Binary mask (0/1) for buildable area.
        cell_mm (float): Grid cell size in mm (910mm by default).
        dpi (float): DPI used for unit conversion.
        scale (float): The scale factor for the diagram (e.g., 1:100).

    Returns:
        np.ndarray: Image with the 910mm grid lines drawn where mask=1.
    """
    out_img = image.copy()
    h, w = shrunk_mask.shape[:2]

    # Convert 910mm to pixels
    cell_px = mm_to_pixels(cell_mm, dpi, scale)

    # Step through in increments of cell_px
    for y in range(0, h, cell_px):
        for x in range(0, w, cell_px):
            # We'll define the cell's center
            center_x = x + cell_px // 2
            center_y = y + cell_px // 2

            if center_x >= w or center_y >= h:
                continue

            # If the center is inside shrunk_mask, draw the cell boundary
            if shrunk_mask[center_y, center_x] > 0:
                p1 = (x, y)
                p2 = (x + cell_px, y + cell_px)
                cv2.rectangle(out_img, p1, p2, (255, 0, 0), 1)  # Blue lines

    return out_img

def process_image(image_file, offset_m: float = 5.0):
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
            
            # 推論結果のプロット画像 (OpenCV BGR形式のNumPy配列)
            result_image = results[0].plot()
            # (height, width, 3)
            h_img, w_img = result_image.shape[:2]

            # セグメンテーションマスクを取り出し、一括で合わせる
            if results[0].masks is not None:
                # Initialize a combined mask with the same size as the result_image
                combined_mask = np.zeros((h_img, w_img), dtype=np.uint8)

                for seg_data in results[0].masks.data:
                    mask_np = seg_data.cpu().numpy().astype(np.uint8)

                    # The YOLO mask is typically the shape the model used for inference (e.g., 480x640).
                    # Resize it to match the result_image shape so we can combine them properly.
                    mask_resized = cv2.resize(
                        mask_np,
                        (w_img, h_img),  # (width, height) order
                        interpolation=cv2.INTER_NEAREST
                    )
                    combined_mask = np.maximum(combined_mask, mask_resized)

                # ピクセルオフセットを計算 (例: 5m)
                offset_px = mm_to_pixels(offset_m * 1000.0, dpi=300.0, scale=0.01)

                # マスクを縮小（セットバック）
                shrunk_mask = offset_mask_by_distance(combined_mask, offset_px=offset_px)

                # 縮小後の輪郭を緑色で描画（線幅太め）
                contours, _ = cv2.findContours(shrunk_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(result_image, contours, -1, (0, 255, 0), 3)

                # 910mmグリッドを描画
                result_image = draw_910mm_grid(
                    image=result_image,
                    shrunk_mask=shrunk_mask,
                    cell_mm=910.0,   # 半畳(910mm) 
                    dpi=300.0,
                    scale=0.01
                )
            
            # 一時ファイルの削除
            os.unlink(temp_image_file.name)

            # NumPy配列をPIL Imageに変換 (BGR -> RGB)
            if isinstance(result_image, np.ndarray):
                result_image = Image.fromarray(result_image[..., ::-1])
            
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

    # セットバック距離を指定（メートル単位）
    offset_m = st.number_input("セットバック距離 (m)", min_value=0.0, max_value=50.0, value=5.0, step=1.0)
    
    # 画像のアップロード
    uploaded_file = st.file_uploader("セグメンテーションする画像をアップロードしてください", type=["jpg", "jpeg", "png"])
    
    if uploaded_file:
        # 元の画像を表示
        st.image(uploaded_file, caption="アップロードされた画像", use_container_width=True)
        
        # 進捗状態を表示
        with st.spinner('セグメンテーションを実行中...'):
            # 推論の実行と結果の表示
            result_image = process_image(uploaded_file, offset_m=offset_m)
            
            if result_image is not None:
                st.image(result_image, caption="セグメンテーション結果（セットバック＆910mmグリッド）", use_container_width=True)
                
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