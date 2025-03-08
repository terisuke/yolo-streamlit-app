import streamlit as st
import os
import tempfile
from google.cloud import storage
from ultralytics import YOLO
import numpy as np
import cv2
from PIL import Image
import io

def initialize_gcs_client():
    try:
        service_account_path = os.path.join("config", "service_account.json")
        if os.path.exists(service_account_path):
            return storage.Client.from_service_account_json(service_account_path)
        else:
            creds = st.secrets["gcp_service_account"]
            return storage.Client.from_service_account_info(creds)
    except Exception as e:
        st.error(f"GCS init error: {e}")
        return None

def download_model(storage_client, bucket, blob_name):
    try:
        bucket_obj = storage_client.bucket(bucket)
        blob = bucket_obj.blob(blob_name)
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
            blob.download_to_filename(tmp.name)
            return tmp.name
    except Exception as e:
        st.error(f"Model download error: {e}")
        return None

def load_yolo_model():
    if "model" not in st.session_state or st.session_state.model is None:
        client = initialize_gcs_client()
        if client:
            path = download_model(client, "yolo-v8-training", "trained_models/best.pt")
            if path:
                st.session_state.model = YOLO(path)

def draw_bounding_box_offset(
    image: np.ndarray,
    box: tuple[int,int,int,int],
    offset_px: int = 10,
    fill_color=(0,0,255),
    alpha=0.3,
    grid_cell_px: int = 50
) -> np.ndarray:
    """
    1) バウンディングボックスをそのまま描画(緑枠)
    2) 内側にoffset_pxだけ縮めた矩形を赤塗り潰し
    3) そこに910mm相当のグリッド(ここでは固定50pxとしてサンプル)
    """
    out = image.copy()
    
    x1,y1,x2,y2 = box
    # 枠を描画(緑)
    cv2.rectangle(out, (x1,y1), (x2,y2), (0,255,0), 2)
    
    # 内側オフセット
    nx1, ny1 = x1+offset_px, y1+offset_px
    nx2, ny2 = x2-offset_px, y2-offset_px
    if nx1>=nx2 or ny1>=ny2:
        # オフセットが大きすぎる場合
        return out
    
    # 半透明塗り潰し
    overlay = out.copy()
    cv2.rectangle(overlay, (nx1,ny1), (nx2,ny2), fill_color, cv2.FILLED)
    cv2.addWeighted(overlay, alpha, out, 1-alpha, 0, out)
    
    # グリッド(青)を描画: 単純に50pxごとにラインを引くだけ
    for yy in range(ny1, ny2, grid_cell_px):
        cv2.line(out, (nx1,yy), (nx2,yy), (255,0,0), 2)
    for xx in range(nx1, nx2, grid_cell_px):
        cv2.line(out, (xx,ny1), (xx,ny2), (255,0,0), 2)
    
    return out

def process_image(image_file, offset_px=50):
    """
    シンプルに YOLO -> Houseのバウンディングボックスを取り、
    そこを内側にオフセットして赤塗り潰し+グリッド描画。
    """
    try:
        image_bytes = image_file.getvalue()
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            tmp.write(image_bytes)
            tmp.close()
            
            results = st.session_state.model(tmp.name, task="detect")
            # detectモード: バウンディングボックスが確実に取れる
            
            # orig_img (BGR)
            orig = results[0].orig_img
            if orig is None:
                st.error("orig_img not found.")
                return None
            
            h, w = orig.shape[:2]
            # 変換用にコピー
            out_bgr = orig.copy()
            
            # 各検出のバウンディングボックスを確認
            boxes = results[0].boxes
            if boxes is None or len(boxes)==0:
                st.warning("No detection found.")
            else:
                for box, cls_id in zip(boxes.xyxy, boxes.cls):
                    # box: [x1,y1,x2,y2]
                    # cls_id: クラスID (ex: 0=house?)
                    # confidence= boxes.conf
                    if int(cls_id) == 1:  # 1をHouseクラスにしている例
                        x1,y1,x2,y2 = box.int().tolist()  # int()でピクセル座標に
                        
                        # Draw offset
                        out_bgr = draw_bounding_box_offset(
                            out_bgr,
                            (x1,y1,x2,y2),
                            offset_px=offset_px,
                            fill_color=(0,0,255),
                            alpha=0.4,
                            grid_cell_px=50 # 簡易固定
                        )
            
            # convert BGR->RGB
            rgb = cv2.cvtColor(out_bgr, cv2.COLOR_BGR2RGB)
            return Image.fromarray(rgb)

    except Exception as e:
        st.error(f"process_image error: {e}")
        return None

def main():
    load_yolo_model()
    if st.session_state.model is None:
        st.warning("モデルロード失敗")
        return
    
    offset_px = st.number_input("オフセット(px)", 0, 500, 50, 10)
    upfile = st.file_uploader("画像アップロード", ["jpg","jpeg","png"])
    if upfile:
        st.image(upfile, caption="アップロード画像", use_container_width=True)
        with st.spinner("Detectモード推論中..."):
            res = process_image(upfile, offset_px=offset_px)
            if res:
                st.image(res, caption="バウンディングボックス+オフセット領域+グリッド", use_container_width=True)
                buf = io.BytesIO()
                res.save(buf, format="PNG")
                st.download_button("結果DL", buf.getvalue(), "result.png","image/png")

if __name__=="__main__":
    main()