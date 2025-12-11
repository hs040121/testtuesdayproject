import os
import io
import cv2
import numpy as np
from PIL import Image, ImageChops, ImageEnhance
from skimage.feature import local_binary_pattern
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from lightgbm import LGBMClassifier
import joblib
from tqdm import tqdm


# --- 1️⃣ 다단계 ELA 특징 추출 ---
def extract_ela_features(image_path, qualities=[90, 85, 80, 75]):
    """여러 JPEG 품질로 ELA 히스토그램 추출"""
    try:
        original_image = Image.open(image_path).convert('RGB')
        features = []

        for q in qualities:
            buffer = io.BytesIO()
            original_image.save(buffer, 'JPEG', quality=q)
            resaved = Image.open(buffer)
            ela = ImageChops.difference(original_image, resaved)

            extrema = ela.getextrema()
            max_diff = max(ex[1] for ex in extrema)
            if max_diff == 0:
                max_diff = 1
            scale = 255.0 / max_diff
            ela_vis = ImageEnhance.Brightness(ela).enhance(scale)

            arr = np.array(ela_vis).flatten()
            hist, _ = np.histogram(arr, bins=10, range=[0, 255])
            hist = hist / np.sum(hist)
            features.extend(hist)

        return np.array(features)

    except Exception as e:
        # print(f"ELA 오류 ({image_path}): {e}")
        return None


# --- 2️⃣ 다중반경 LBP 특징 추출 ---
def extract_lbp_features(image_path, radii=[1, 2, 3, 4]):
    """여러 반경으로 LBP 히스토그램 추출"""
    try:
        with open(image_path, "rb") as f:
            image_bytes = f.read()

        image_np = np.frombuffer(image_bytes, np.uint8)
        gray = cv2.imdecode(image_np, cv2.IMREAD_GRAYSCALE)

        if gray is None:
            raise ValueError("OpenCV 디코딩 실패")

        gray = cv2.resize(gray, (256, 256))
        features = []

        for r in radii:
            n_points = 8 * r
            lbp = local_binary_pattern(gray, n_points, r, method='uniform')
            hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, n_points + 3), range=(0, n_points + 2))
            hist = hist / np.sum(hist)
            features.extend(hist)

        return np.array(features)

    except Exception as e:
        # print(f"LBP 오류 ({image_path}): {e}")
        return None


# --- 3️⃣ 데이터셋 로드 ---
def load_dataset(dataset_path):
    features = []
    labels = []

    valid_exts = ('.png', '.jpg', '.jpeg', '.webp', '.bmp')
    skipped = 0

    for label_name in ['real', 'fake']:
        label_dir = os.path.join(dataset_path, label_name)
        if not os.path.isdir(label_dir):
            print(f"경고: {label_dir} 폴더가 없습니다. 건너뜀.")
            continue

        print(f"{label_name} 이미지 처리 중...")
        image_files = [
            os.path.join(root, f)
            for root, _, files in os.walk(label_dir)
            for f in files if f.lower().endswith(valid_exts)
        ]
        print(f"  - {len(image_files)}개 이미지 발견.")

        for image_path in tqdm(image_files, desc=f"  {label_name} 폴더"):
            ela = extract_ela_features(image_path)
            lbp = extract_lbp_features(image_path)

            if ela is None or lbp is None:
                skipped += 1
                continue

            combined = np.concatenate([ela, lbp])
            features.append(combined)
            labels.append(1 if label_name == 'fake' else 0)

    print(f"\n총 {skipped}개 파일은 손상 또는 처리 실패로 건너뜀.")
    return np.array(features), np.array(labels)


# --- 4️⃣ 모델 학습 및 평가 ---
if __name__ == "__main__":
    DATASET_PATH = '../dataset'
    MODEL_SAVE_PATH = 'classifier.pkl'

    print("데이터셋 로드 및 특징 추출 중...")
    X, y = load_dataset(DATASET_PATH)

    if len(X) < 10:
        print(f"데이터셋이 너무 적습니다 ({len(X)}개). 최소 10개 이상 필요.")
        exit()

    print(f"\n총 {len(X)}개 유효 이미지 처리 완료.")
    print(f"특징 벡터 차원: {X.shape[1]}")

    # 데이터 분할
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 특징 정규화
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # 모델 구성 (LightGBM)
    print("\nLightGBM 모델 학습 시작...")
    model = LGBMClassifier(
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=64,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )

    model.fit(X_train, y_train)
    print("모델 학습 완료.")

    # 평가
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n테스트 정확도: {acc * 100:.2f}%")

    # 저장 (모델 + 스케일러)
    joblib.dump({"model": model, "scaler": scaler}, MODEL_SAVE_PATH)
    print(f"\n모델이 '{MODEL_SAVE_PATH}'로 저장되었습니다.")
