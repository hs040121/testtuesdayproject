import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import joblib
import numpy as np
from PIL import Image, ImageChops, ImageEnhance
import cv2
from skimage.feature import local_binary_pattern
import io
from pathlib import Path
import base64
from transformers import AutoImageProcessor, AutoModelForImageClassification
import torch
import torch.nn.functional as F  # Softmax 사용을 위해 추가

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 모델 및 스케일러 전역 변수 ---
model = None
scaler = None
hf_model = None
hf_processor = None
ELA_QUALITIES = [90, 85, 80, 75]  # ELA 품질 리스트 (train_model.py와 일치해야 함)
LBP_RADII = [1, 2, 3, 4]  # LBP 반경 리스트 (train_model.py와 일치해야 함)

try:
    BASE_DIR = Path(__file__).resolve().parent.parent

    # 1. Custom 모델 로드 (딕셔너리 형태 처리)
    MODEL_PATH = BASE_DIR / "model_training" / "classifier.pkl"
    if MODEL_PATH.exists():
        loaded_data = joblib.load(MODEL_PATH)

        # 저장된 파일이 딕셔너리인지 확인
        if isinstance(loaded_data, dict):
            model = loaded_data.get("model")
            scaler = loaded_data.get("scaler")
            print("사용자 정의 모델(ELA+LBP) 및 스케일러 로딩 성공! (Dict Type)")
        else:
            # 딕셔너리가 아니라 모델 객체 자체만 저장된 경우 (예전 방식 호환)
            model = loaded_data
            scaler = None
            print("사용자 정의 모델(ELA+LBP) 로딩 성공! (Model Only Type)")

    else:
        print(f"경고: 사용자 정의 모델 파일을 찾을 수 없습니다: {MODEL_PATH}")

    # 2. Hugging Face 모델 로드 (로컬 경로)
    hf_model_path = BASE_DIR / "models" / "vit-ai-or-real"
    if hf_model_path.exists():
        hf_processor = AutoImageProcessor.from_pretrained(hf_model_path)
        hf_model = AutoModelForImageClassification.from_pretrained(hf_model_path)
        print(f"성공: 로컬 경로 '{hf_model_path}'에서 허깅페이스 모델(ViT)을 로딩했습니다.")
    else:
        print(f"경고: 허깅페이스 모델 폴더를 찾을 수 없습니다: {hf_model_path}")

except Exception as e:
    print(f"모델 로딩 중 오류 발생: {e}")


# --- ELA 특징 추출 (4개 품질로 확장, train_model.py와 완벽히 동기화) ---
def extract_ela_features(image_bytes, qualities=ELA_QUALITIES):
    """Pillow를 사용하여 ELA 특징 추출 (4개 품질로 확장, train_model.py와 동기화)"""
    features = []
    base64_image = ""
    try:
        original_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        for idx, quality in enumerate(qualities):
            resaved_buffer = io.BytesIO()
            original_image.save(resaved_buffer, "JPEG", quality=quality)
            resaved_image = Image.open(resaved_buffer)

            ela_image = ImageChops.difference(original_image, resaved_image)

            extrema = ela_image.getextrema()
            max_diff = max([ex[1] for ex in extrema])
            if max_diff == 0:
                max_diff = 1
            scale = 255.0 / max_diff

            ela_image_visual = ImageEnhance.Brightness(ela_image).enhance(scale)

            # 시각화용 이미지 (가장 첫 번째 품질만 저장)
            if idx == 0:
                buffered = io.BytesIO()
                ela_image_visual.save(buffered, format="PNG")
                base64_image = "data:image/png;base64," + base64.b64encode(
                    buffered.getvalue()
                ).decode("utf-8")

            # ★★★ 핵심 수정: ela_image_visual에서 히스토그램 추출 (train_model.py와 일치) ★★★
            hist, _ = np.histogram(
                np.array(ela_image_visual).flatten(), bins=10, range=[0, 255]
            )

            if np.sum(hist) == 0:
                features.extend(np.zeros(10))
            else:
                features.extend(hist / np.sum(hist))

        # features는 numpy array로 변환되어 반환 (40차원)
        return np.array(features), base64_image

    except Exception as e:
        print(f"ELA 특징 추출 오류: {e}")
    # 오류 시 fallback: 40차원 0 벡터 반환
    return np.zeros(len(qualities) * 10), base64_image


# --- LBP 특징 추출 (4개 반경, train_model.py와 완벽히 동기화) ---
def extract_lbp_features(image_bytes, radii=LBP_RADII):
    features = []
    try:
        image_np = np.frombuffer(image_bytes, np.uint8)
        gray_image = cv2.imdecode(image_np, cv2.IMREAD_GRAYSCALE)

        if gray_image is None:
            raise ValueError("OpenCV가 이미지를 디코딩할 수 없습니다.")

        gray_image = cv2.resize(gray_image, (256, 256))

        for r in radii:
            n_points = 8 * r
            lbp = local_binary_pattern(gray_image, n_points, r, method="uniform")

            hist, _ = np.histogram(
                lbp.ravel(), bins=np.arange(0, n_points + 3), range=(0, n_points + 2)
            )
            if np.sum(hist) == 0:
                features.extend(np.zeros(n_points + 2))
            else:
                features.extend(hist / np.sum(hist))

        # features는 numpy array로 변환되어 반환 (88차원)
        return np.array(features)
    except Exception as e:
        print(f"LBP 특징 추출 오류: {e}")
        # 오류 시 fallback: 88차원 0 벡터 반환
        return np.zeros(sum(8 * r + 2 for r in radii))


def analyze_image_grid(ela_image_bytes):
    try:
        header, encoded = ela_image_bytes.split(",", 1)
        ela_img_data = base64.b64decode(encoded)
        ela_img = Image.open(io.BytesIO(ela_img_data)).convert("L")
        ela_array = np.array(ela_img)
        h, w = ela_array.shape
        grid_size = 4
        cell_h, cell_w = h // grid_size, w // grid_size
        max_mean = -1
        hotspot_coords = (0, 0)
        for r in range(grid_size):
            for c in range(grid_size):
                cell = ela_array[
                    r * cell_h : (r + 1) * cell_h, c * cell_w : (c + 1) * cell_w
                ]
                mean_val = np.mean(cell)
                if mean_val > max_mean:
                    max_mean = mean_val
                    hotspot_coords = (r, c)
        row_map = {0: "상단", 1: "중상단", 2: "중하단", 3: "하단"}
        col_map = {0: "좌측", 1: "중좌측", 2: "중우측", 3: "우측"}
        location_text = (
            f"{row_map[hotspot_coords[0]]} {col_map[hotspot_coords[1]]}"
        )
        return {
            "grid_size": grid_size,
            "hotspot_row": hotspot_coords[0],
            "hotspot_col": hotspot_coords[1],
            "location_text": location_text,
        }
    except Exception as e:
        print(f"그리드 분석 오류: {e}")
        return None


# --- API 엔드포인트 ---
@app.get("/")
def read_root():
    return {"message": "AI 이미지 분석기 백엔드"}


@app.post("/analyze-with-custom-model/")
async def analyze_image_custom(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(
            status_code=500, detail="서버에 사용자 정의 모델이 로드되지 않았습니다."
        )
    try:
        contents = await file.read()

        # 특징 추출
        ela_features, ela_image_base64 = extract_ela_features(contents)
        lbp_features = extract_lbp_features(contents)
        combined_features = np.concatenate([ela_features, lbp_features]).reshape(1, -1)

        # --- 스케일러 적용 및 차원 보정 ---
        if scaler:
            try:
                # 스케일러가 기대하는 입력 차원 확인
                if hasattr(scaler, "mean_"):
                    expected_dim = scaler.mean_.shape[0]
                elif hasattr(scaler, "n_features_in_"):
                    expected_dim = scaler.n_features_in_
                else:
                    expected_dim = combined_features.shape[1]  # fallback

                current_dim = combined_features.shape[1]

                if current_dim < expected_dim:
                    # 부족하면 0으로 패딩
                    padding = np.zeros((1, expected_dim - current_dim))
                    combined_features = np.hstack((combined_features, padding))
                elif current_dim > expected_dim:
                    # 초과하면 잘라냄
                    combined_features = combined_features[:, :expected_dim]

                # 스케일러 적용
                scaled_features = scaler.transform(combined_features)
            except Exception as e:
                print(f"스케일러 적용 중 오류 (무시하고 원본 사용): {e}")
                scaled_features = combined_features
        else:
            # 스케일러가 없으면 원본 특징 사용
            scaled_features = combined_features

        # 예측 수행
        probability = model.predict_proba(scaled_features)[0]
        prediction = np.argmax(probability)

        result_label = "AI 생성 이미지" if prediction == 1 else "실제 사진"
        confidence = probability[prediction] * 100
        real_prob = probability[0] * 100
        fake_prob = probability[1] * 100

        grid_analysis = analyze_image_grid(ela_image_base64)

        insights = []
        hotspot_location = (
            grid_analysis["location_text"] if grid_analysis else "특정 영역"
        )

        if result_label == "AI 생성 이미지":
            insights.append(
                f"이미지의 **{hotspot_location}**에서 상대적으로 높은 압축 오류(ELA)가 발견되었습니다."
            )
            insights.append(
                "이는 해당 영역이 다른 부분과 다르게 처리되었을 가능성을 나타냅니다."
            )
        else:
            insights.append("이미지 전반적으로 압축 오류 레벨이 낮고 일관되게 나타납니다.")
            insights.append(
                f"가장 변화가 많은 **{hotspot_location}** 영역조차 자연스러운 사진의 범주에 속합니다."
            )

        return {
            "prediction": result_label,
            "confidence": f"{confidence:.2f}%",
            "details": {
                "real_prob": f"{real_prob:.2f}%",
                "fake_prob": f"{fake_prob:.2f}%",
            },
            "suspicious_area": grid_analysis,
            "insights": insights,
        }

    except Exception as e:
        print(f"Custom 모델 분석 중 오류: {e}")
        # 상세한 오류 메시지 반환
        raise HTTPException(
            status_code=500, detail=f"Custom 모델 분석 중 오류 발생: {str(e)}"
        )


@app.post("/analyze-with-huggingface-model/")
async def analyze_image_hf(file: UploadFile = File(...)):
    if not hf_model or not hf_processor:
        raise HTTPException(
            status_code=500, detail="서버에 허깅페이스 모델이 로드되지 않았습니다."
        )
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        inputs = hf_processor(images=image, return_tensors="pt")
        with torch.no_grad():
            outputs = hf_model(**inputs)

        logits = outputs.logits
        # ★★★ 핵심 수정: Softmax를 적용하여 합이 1인 확률 분포 얻기 (dim=1) ★★★
        probabilities = F.softmax(logits, dim=1)[0]

        # 모델 레이블 순서 (0: real, 1: fake 가정)
        real_prob_percent = probabilities[0].item() * 100
        fake_prob_percent = probabilities[1].item() * 100

        if fake_prob_percent > real_prob_percent:
            result_label, confidence = "AI 생성 이미지", fake_prob_percent
        else:
            result_label, confidence = "실제 사진", real_prob_percent

        insights = [
            "이 결과는 로컬에 저장된 **Vision Transformer(ViT) 딥러닝 모델**에 기반합니다.",
        ]

        return {
            "prediction": result_label,
            "confidence": f"{confidence:.2f}%",
            "details": {
                "real_prob": f"{real_prob_percent:.2f}%",
                "fake_prob": f"{fake_prob_percent:.2f}%",
            },
            "insights": insights,
        }
    except Exception as e:
        print(f"HF 모델 분석 중 오류: {e}")
        raise HTTPException(status_code=500, detail="HF 모델 분석 중 오류 발생")


if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
