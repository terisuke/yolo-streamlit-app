import streamlit as st
import os
import tempfile
from google.cloud import storage
from ultralytics import YOLO
import numpy as np
import cv2
from PIL import Image
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
            creds = st.secrets["gcp_service_account"]
            return storage.Client.from_service_account_info(creds)
    except Exception as e:
        st.error(f"GCS init error: {e}")
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
        st.error(f"Model download error: {e}")
        return None

def load_yolo_model() -> None:
    """
    YOLOモデルをロードします。
    """
    if "model" not in st.session_state or st.session_state.model is None:
        client = initialize_gcs_client()
        if client:
            path = download_model(client, "yolo-v8-training", "trained_models/best.pt")
            if path:
                st.session_state.model = YOLO(path)

def offset_mask_by_distance(mask: np.ndarray, offset_px: int) -> np.ndarray:
    """
    マスクを距離変換して内側にオフセットしたバイナリマスクを返します。
    """
    bin_mask = (mask > 0).astype(np.uint8)
    dist = cv2.distanceTransform(bin_mask, cv2.DIST_L2, 5)
    shrunk = (dist >= offset_px).astype(np.uint8)
    return shrunk

def draw_910mm_grid_on_rect(
    image: np.ndarray,
    rect: tuple[int,int,int,int],
    grid_mm: float = 910.0,
    dpi: float = 300.0,
    scale: float = 1.0,
    fill_color=(255,0,0),
    alpha=0.4,
    line_color=(0,0,255),
    line_thickness=2
) -> np.ndarray:
    """
    rect=(x,y,w,h)に半透明塗り潰し & 指定mm格子を描画。
    ここでは簡単に 1mm= (dpi/25.4) px * scale という換算。
    """
    out = image.copy()

    x, y, w_, h_ = rect
    x2, y2 = x + w_, y + h_

    # 半透明塗り潰し
    overlay = out.copy()
    cv2.rectangle(overlay, (x, y), (x2, y2), fill_color, cv2.FILLED)
    cv2.addWeighted(overlay, alpha, out, 1 - alpha, 0, out)

    # 外枠
    cv2.rectangle(out, (x, y), (x2, y2), (0, 0, 255), 2)

    def mm_to_px(mm: float) -> int:
        inch = mm / 25.4
        px = inch * dpi * scale
        return int(round(px))

    cell_px = mm_to_px(grid_mm)

    # フォールバック処理
    if cell_px > w_ or cell_px > h_:
        fallback = max(1, min(w_, h_) // 5)
        st.warning(f"cell_px={cell_px} が大きすぎるため、{fallback}pxに調整します。")
        cell_px = fallback

    # 格子線を描画
    for gy in range(y, y2, cell_px):
        cv2.line(out, (x, gy), (x2, gy), line_color, line_thickness)
    for gx in range(x, x2, cell_px):
        cv2.line(out, (gx, y), (gx, y2), line_color, line_thickness)

    return out

def process_image(
    image_file,
    near_offset_px: int = 100,
    far_offset_px: int = 50,
    grid_mm: float = 910.0,
    dpi: float = 300.0,
    scale: float = 1.0
) -> Image.Image | None:
    """
    1) YOLOセグメンテーションで "House" と "Road" マスクを合成
    2) Roadから近いHouseは near_offset_px でオフセット、
       それ以外は far_offset_px でオフセット
    3) boundingRectをとり、そこに grid_mm 間隔の格子を描画
    4) Houseマスク(元サイズ)は緑色半透明で表示
    """
    try:
        image_bytes = image_file.getvalue()

        # 1) 推論実行
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            tmp.write(image_bytes)
            tmp.close()
            results = st.session_state.model(tmp.name, task="segment")

        orig = results[0].orig_img  # shape e.g. (1755, 2481, 3)
        if orig is None:
            st.error("orig_img not found.")
            return None

        # 2) YOLOマスクは小さい(480x640など)なので、後で元サイズ(2481x1755)に合わせる
        h, w = orig.shape[:2]  # h=1755, w=2481

        HOUSE_CLASS_ID = 1
        ROAD_CLASS_ID = 2

        house_mask = np.zeros((h, w), dtype=np.uint8)  # (1755,2481)
        road_mask  = np.zeros((h, w), dtype=np.uint8)

        # 3) セグメンテーションマスクを合成
        if results[0].masks is not None:
            for seg_data, cls_id in zip(results[0].masks.data, results[0].boxes.cls):
                m = seg_data.cpu().numpy().astype(np.uint8)
                # YOLO推論で得られるm.shape は(480,640)など
                # → 幅=640, 高さ=480
                # origは 幅=2481, 高さ=1755
                # なので (w, h)=(2481,1755) でリサイズ
                # NOTE: cv2.resize expects (width, height) as second argument
                resized = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)

                if int(cls_id) == HOUSE_CLASS_ID:
                    house_mask = np.maximum(house_mask, resized)
                elif int(cls_id) == ROAD_CLASS_ID:
                    road_mask = np.maximum(road_mask, resized)

        # (A) Houseを緑色半透明で表示
        out_bgr = orig.copy()
        overlay = out_bgr.copy()
        overlay[house_mask==1] = (0,255,0)
        cv2.addWeighted(overlay, 0.3, out_bgr, 0.7, 0, out_bgr)

        # (B) Road距離で「near/far」に分けて別オフセット
        bin_road = (road_mask>0).astype(np.uint8)
        dist_road = cv2.distanceTransform(bin_road, cv2.DIST_L2, 5)
        near_threshold = 20
        near_road = (dist_road < near_threshold).astype(np.uint8)

        near_house = (house_mask & near_road)
        far_house  = (house_mask & (1 - near_road))

        shrunk_near = offset_mask_by_distance(near_house, near_offset_px)
        shrunk_far  = offset_mask_by_distance(far_house,  far_offset_px)
        final_house = np.maximum(shrunk_near, shrunk_far)

        # (C) boundingRect + グリッド
        contours, _ = cv2.findContours(final_house, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours)==0:
            st.warning("No house after offset.")
        else:
            big_contour = max(contours, key=cv2.contourArea)
            x,y,wc,hc = cv2.boundingRect(big_contour)

            out_bgr = draw_910mm_grid_on_rect(
                image=out_bgr,
                rect=(x,y,wc,hc),
                grid_mm=grid_mm,
                dpi=dpi,
                scale=scale,
                fill_color=(255,0,0),
                alpha=0.4,
                line_color=(0,0,255),
                line_thickness=2
            )

        rgb = cv2.cvtColor(out_bgr, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb)

    except Exception as e:
        st.error(f"process_image error: {e}")
        return None

def main() -> None:
    load_yolo_model()
    if st.session_state.model is None:
        st.warning("モデルロード失敗")
        return

    st.title("Fix shape mismatch: YOLO masks(480x640) → Original(1755x2481)")

    offset_near = st.number_input("Roadに近い領域のオフセット(px)", 0, 5000, 100, 10)
    offset_far  = st.number_input("Road以外領域のオフセット(px)", 0, 5000, 50, 10)
    grid_mm     = st.number_input("グリッド間隔(mm)", 1.0, 10000.0, 910.0, 10.0)
    dpi_val     = st.number_input("DPI", 1.0, 1200.0, 300.0, 1.0)
    scale_val   = st.number_input("スケール(例:1.0)", 0.01, 10.0, 1.0, 0.01)

    upfile = st.file_uploader("画像アップロード", ["jpg","jpeg","png"])
    if upfile:
        st.image(upfile, "アップロード画像", use_container_width=True)
        with st.spinner("処理中..."):
            result = process_image(
                image_file=upfile,
                near_offset_px=offset_near,
                far_offset_px=offset_far,
                grid_mm=grid_mm,
                dpi=dpi_val,
                scale=scale_val
            )
            if result:
                st.image(result, "結果画像", use_container_width=True)
                buf = io.BytesIO()
                result.save(buf, "PNG")
                st.download_button("結果をダウンロード", buf.getvalue(), "result.png", "image/png")

if __name__=="__main__":
    main()