import streamlit as st
import rasterio
import numpy as np
import matplotlib.pyplot as plt
import tempfile
import os

# 1. CRISM MTRDR 파일 로드 함수 (.img + .hdr 지원)
def load_crism(files):
    # 임시 디렉토리 생성
    tmp_dir = tempfile.mkdtemp()
    img_path = None
    # 업로드된 파일들 저장
    for uploaded in files:
        suffix = os.path.splitext(uploaded.name)[1]
        out_path = os.path.join(tmp_dir, uploaded.name)
        with open(out_path, "wb") as f:
            f.write(uploaded.read())
        # .img 파일 경로 기록
        if suffix.lower() == ".img":
            img_path = out_path
    if img_path is None:
        st.error(".img 파일이 필요합니다.")
        return None, None
    # rasterio로 .img 파일 열기 (.hdr가 같은 디렉토리에 있어야 함)
    dataset = rasterio.open(img_path)
    data = dataset.read()  # shape: (bands, rows, cols)
    return dataset, data

# 2. 시각화 함수 (예시: RGB 합성)
def visualize(data):
    # 밴드 인덱스는 CRISM wavelength에 따라 조정 필요 (1-based -> 0-based)
    band_r, band_g, band_b = 10, 30, 50
    rgb = np.stack([data[band_r-1], data[band_g-1], data[band_b-1]], axis=2)
    fig, ax = plt.subplots()
    ax.imshow(rgb)
    ax.axis('off')
    st.pyplot(fig)

# 3. 수분/염분 지표 계산 함수
def analyze_h2o_salt(data):
    # 예시: 수분 흡수 밴드, 염분 흡수 밴드 인덱스 (조정 필요)
    h2o_band = 100
    salt_band = 150
    h2o_ref = data[h2o_band-1].astype(float)
    salt_ref = data[salt_band-1].astype(float)
    # 간단한 평균 반사율 지표
    h2o_index = np.nanmean(h2o_ref)
    salt_index = np.nanmean(salt_ref)
    return h2o_index, salt_index

# 4. 생존 가능성 확률 계산 함수
def calculate_habitability(h2o_idx, salt_idx):
    # 로지스틱 모델 기반 (임의 기준치: 0.5)
    h2o_score = 1 / (1 + np.exp(- (h2o_idx - 0.5)))
    salt_score = 1 / (1 + np.exp(salt_idx - 0.5))
    # 가중 평균: 수분 0.6, 염분 0.4
    prob = 0.6 * h2o_score + 0.4 * (1 - salt_score)
    return prob

# Streamlit 앱 메인 함수
def main():
    st.title('CRISM MTRDR Viewer & Habitability Estimator')
    st.markdown("""
CRISM hyperspectral 데이터로부터 수분/염분 지표 및 생존 확률을 계산합니다.
.img 파일과 .hdr 파일을 함께 업로드해주세요.
""")
    uploaded_files = st.file_uploader(
        'CRISM MTRDR 파일 업로드 (.img + .hdr)',
        type=['img', 'hdr'],
        accept_multiple_files=True
    )
    if uploaded_files and len(uploaded_files) >= 2:
        dataset, data = load_crism(uploaded_files)
        if dataset is None:
            return

        st.header('1. 데이터 시각화')
        visualize(data)

        st.header('2. 수분/염분 지표 분석')
        h2o_idx, salt_idx = analyze_h2o_salt(data)
        st.write(f'- 수분 지표 (H2O Index): {h2o_idx:.3f}')
        st.write(f'- 염분 지표 (Salt Index): {salt_idx:.3f}')

        st.header('3. 생존 가능성 확률')
        prob = calculate_habitability(h2o_idx, salt_idx)
        st.write(f'- 예상 생존 가능성 확률: **{prob * 100:.2f}%**')
    else:
        st.info('*.img* 파일과 해당하는 *.hdr* 파일을 함께 업로드하세요.')

if __name__ == '__main__':
    main()
