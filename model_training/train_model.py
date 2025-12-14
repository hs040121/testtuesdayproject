import os
import cv2
import numpy as np
from PIL import Image, ImageChops, ImageEnhance
from skimage.feature import local_binary_pattern
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler  # StandardScaler import 추가
import joblib
from tqdm import tqdm
import io


# --- 1. 특징 추출 함수들 (ELA 4개 품질로 확장) ---

def extract_ela_features(image_path, qualities=[90, 85, 80, 75]):
    """Pillow를 사용하여 ELA 특징 추출 (4개 품질로 확장)"""
    features = []
    try:
        original_image = Image.open(image_path).convert('RGB')

        for quality in qualities:
            resaved_buffer = io.BytesIO()
            original_image.save(resaved_buffer, 'JPEG', quality=quality)
            resaved_image = Image.open(resaved_buffer)

            ela_image = ImageChops.difference(original_image, resaved_image)

            extrema = ela_image.getextrema()
            max_diff = max([ex[1] for ex in extrema])
            if max_diff == 0: max_diff = 1
            scale = 255.0 / max_diff

            ela_image_visual = ImageEnhance.Brightness(ela_image).enhance(scale)

            # 히스토그램 추출 (각 품질당 10차원)
            hist, _ = np.histogram(np.array(ela_image_visual).flatten(), bins=10, range=[0, 255])

            if np.sum(hist) == 0:
                features.extend(np.zeros(10))
            else:
                features.extend(hist / np.sum(hist))

        return np.array(features)

    except Exception as e:
        # print(f"ELA 오류 ({image_path}): {e}")
        return None  # ★★★ 오류 발생 시 None 반환 ★★★


def extract_lbp_features(image_path):
    """한글/특수문자 경로 문제를 해결한 LBP 특징 추출"""
    radius = 3
    n_points = 8 * radius
    try:
        with open(image_path, "rb") as f:
            image_bytes = f.read()

        image_np = np.frombuffer(image_bytes, np.uint8)
        gray_image = cv2.imdecode(image_np, cv2.IMREAD_GRAYSCALE)

        if gray_image is None:
            raise ValueError("OpenCV가 이미지를 디코딩할 수 없습니다.")

        gray_image = cv2.resize(gray_image, (256, 256))

        lbp = local_binary_pattern(gray_image, n_points, radius, method='uniform')

        hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, n_points + 3), range=(0, n_points + 2))

        if np.sum(hist) == 0:
            return np.zeros(n_points + 2)
        return hist / np.sum(hist)

    except Exception as e:
        # print(f"LBP 오류 ({image_path}): {e}")
        return None  # ★★★ 오류 발생 시 None 반환 ★★★


# --- 2. 데이터셋 로드 및 전처리 ---
def load_dataset(dataset_path):
    features = []
    labels = []

    valid_extensions = ('.png', '.jpg', '.jpeg', '.webp', '.bmp')
    skipped_files = 0

    for label_name in ['real', 'fake']:
        print(f"{label_name} 이미지 처리 중...")
        image_dir = os.path.join(dataset_path, label_name)

        if not os.path.isdir(image_dir):
            print(f"경고: '{image_dir}' 폴더를 찾을 수 없습니다. 건너뜁니다.")
            continue

        file_list = []
        for root, dirs, files in os.walk(image_dir):
            for file_name in files:
                if file_name.lower().endswith(valid_extensions):
                    file_list.append(os.path.join(root, file_name))

        print(f"  - 총 {len(file_list)}개의 {label_name} 이미지 파일을 찾았습니다. 특징 추출을 시작합니다...")

        for image_path in tqdm(file_list, desc=f"  {label_name} 폴더 처리 중"):
            ela = extract_ela_features(image_path)  # 40차원
            lbp = extract_lbp_features(image_path)  # 26차원

            # ELA 또는 LBP 추출 실패 시(None 반환 시) 데이터셋에 추가하지 않음
            if ela is None or lbp is None:
                skipped_files += 1
                continue

            combined_features = np.concatenate([ela, lbp])  # 총 66차원

            features.append(combined_features)
            labels.append(1 if label_name == 'fake' else 0)  # fake=1, real=0

    print(f"\n총 {skipped_files}개의 손상되거나 읽을 수 없는 파일을 건너뛰었습니다.")
    return np.array(features), np.array(labels)


# --- 3. 모델 학습 및 평가 ---
if __name__ == "__main__":
    DATASET_PATH = '../dataset'
    MODEL_SAVE_PATH = 'classifier.pkl'

    print("데이터셋을 로드하고 특징을 추출합니다...")
    X, y = load_dataset(DATASET_PATH)

    if len(X) < 10:
        print(f"\n오류: 데이터셋에 이미지가 너무 적습니다 ({len(X)}개). 최소 10개 이상의 이미지가 필요합니다.")
        print("'dataset/real'과 'dataset/fake' 폴더에 이미지를 추가해주세요.")
    else:
        print(f"\n총 {len(X)}개의 유효한 이미지에서 특징 추출 완료.")
        print(f"특징 벡터의 크기: {X.shape[1]}")  # 66차원

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        # 특징 표준화 (Standardization)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        print("\n모델 학습을 시작합니다 (데이터가 많아 시간이 걸릴 수 있습니다)...")
        model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, verbose=1)
        model.fit(X_train_scaled, y_train)
        print("모델 학습 완료.")

        print("\n모델 성능을 평가합니다...")
        y_pred = model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"\n모델 테스트 정확도: {accuracy * 100:.2f}%")

        # 모델 저장 시 스케일러도 같이 저장 (딕셔너리 형태)
        joblib.dump({'model': model, 'scaler': scaler}, MODEL_SAVE_PATH)
        print(f"\n학습된 모델과 스케일러를 '{MODEL_SAVE_PATH}' 파일로 저장했습니다.")
