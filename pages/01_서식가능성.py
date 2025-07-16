# app.py
# requirements.txt
# streamlit
# rasterio
# numpy
# matplotlib

import streamlit as st
import rasterio
import numpy as np
import matplotlib.pyplot as plt
import tempfile
import os

# 1. CRISM MTRDR 파일 로드 함수 (.img + .hdr 지원)
def load_crism(files):
    tmp_dir = tempfile.mkdtemp()
    img_path = None
    for uploaded in files:
        suffix = os.path.splitext(uploaded.name)[1]
        out_path = os.path.join(tmp_dir, uploaded.name)
        with open(out_path, "wb") as f:
            f.write(uploaded.read())
        if suffix.lower() == ".img":
            img_path = out_path
    if img_path is None:
        st.error(".img 파일이 필요합니다.")
        return None, None

    dataset = rasterio.open(img_path)
    data = dataset.read().astype(float)
    # 원본 I/F 값을 그대로 반환
    return dataset, data

# 2. 시각화 함수 (원본 데이터 사용)
def visualize(data):
    n_bands = data.shape[0]
    if n_bands < 3:
        st.error(f"밴드 수가 {n_bands}개로 RGB 합성이 불가능합니다.")
        return

    st.sidebar.header('RGB 밴드 선택')
    default_r, default_g, default_b = 1, n_bands//2, n_bands
    r = st.sidebar.slider('Red 밴드', 1, n_bands, default_r)
    g = st.sidebar.slider('Green 밴드', 1, n_bands, default_g)
    b = st.sidebar.slider('Blue 밴드', 1, n_bands, default_b)

    rgb = np.stack([data[r-1], data[g-1], data[b-1]], axis=2)
    vmin = np.percentile(rgb, 2)
    vmax = np.percentile(rgb, 98)

    fig, ax = plt.subplots(figsize=(6,6))
    ax.imshow(rgb, vmin=vmin, vmax=vmax)
    ax.set_title(f"RGB 합성 (R={r}, G={g}, B={b})")
    ax.axis('off')
    st.pyplot(fig)

# 3. 수분/염분 지표 계산 및 맵 생성 함수
# 밴드 맵은 원본 데이터 사용
def compute_indices(data):
    n_bands = data.shape[0]
    h2o_band = min(100, n_bands)
    salt_band = min(150, n_bands)

    h2o_map = data[h2o_band-1]
    salt_map = data[salt_band-1]
    h2o_idx = np.nanmean(h2o_map)
    salt_idx = np.nanmean(salt_map)

    return h2o_map, salt_map, h2o_idx, salt_idx, h2o_band, salt_band

# 4. 생존 가능성 확률 계산(픽셀 단위)
def habitability_map(h2o_map, salt_map):
    h2o_score = 1 / (1 + np.exp(-(h2o_map - 0.5)))
    salt_score = 1 / (1 + np.exp(salt_map - 0.5))
    return 0.6 * h2o_score + 0.4 * (1 - salt_score)

# 메인 함수
def main():
    st.title('CRISM MTRDR Viewer & Habitability Estimator')
    st.markdown("""
CRISM hyperspectral 파일(.img + .hdr)을 업로드하면:

1. 원본 기반 RGB 시각화
2. 수분/염분 지표 맵 및 평균값
3. 픽셀 단위 생존 확률 맵 및 임계치 비교
""")

    files = st.file_uploader('파일 업로드 (.img + .hdr)', type=['img','hdr'], accept_multiple_files=True)
    if not files or len(files) < 2:
        st.info('*.img* 파일과 대응하는 *.hdr* 파일을 함께 업로드하세요.')
        return

    dataset, data = load_crism(files)
    if dataset is None:
        return

    # 1. 시각화
    st.header('1. 원본 기반 RGB 시각화')
    visualize(data)

    # 2. 수분/염분 지표
    st.header('2. 수분/염분 지표 맵')
    h2o_map, salt_map, h2o_idx, salt_idx, h2o_band, salt_band = compute_indices(data)
    col1, col2 = st.columns(2)
    # H2O 맵 컬러 스트레칭 및 컬러맵 적용
    with col1:
        fig1, ax1 = plt.subplots()
        vmin1, vmax1 = np.percentile(h2o_map, 2), np.percentile(h2o_map, 98)
        cax1 = ax1.imshow(h2o_map, vmin=vmin1, vmax=vmax1, cmap='Blues')
        ax1.set_title(f'H2O 맵 (밴드 {h2o_band})')
        ax1.axis('off')
        fig1.colorbar(cax1, ax=ax1, fraction=0.046, pad=0.04)
        st.pyplot(fig1)
        st.write(f'- 평균 H2O 지표: {h2o_idx:.3f}')
    # Salt 맵 컬러 스트레칭 및 컬러맵 적용
    with col2:
        fig2, ax2 = plt.subplots()
        vmin2, vmax2 = np.percentile(salt_map, 2), np.percentile(salt_map, 98)
        cax2 = ax2.imshow(salt_map, vmin=vmin2, vmax=vmax2, cmap='Reds')
        ax2.set_title(f'Salt 맵 (밴드 {salt_band})')
        ax2.axis('off')
        fig2.colorbar(cax2, ax=ax2, fraction=0.046, pad=0.04)
        st.pyplot(fig2)
        st.write(f'- 평균 Salt 지표: {salt_idx:.3f}')

    # 3. 생존 확률 및 임계치 적용
    st.header('3. 생존 확률 맵 & 임계치 비교')
    prob_map = habitability_map(h2o_map, salt_map)
    thresh_h2o = st.slider('Water Threshold', float(h2o_map.min()), float(h2o_map.max()), 0.75)
    thresh_salt = st.slider('Salt Threshold', float(salt_map.min()), float(salt_map.max()), 0.20)
    fig3, ax3 = plt.subplots()
    cax3 = ax3.imshow(prob_map, cmap='viridis')
    ax3.set_title('생존 확률 맵')
    ax3.axis('off')
    fig3.colorbar(cax3, ax=ax3, fraction=0.046, pad=0.04)
    st.pyplot(fig3)

    mask = (h2o_map >= thresh_h2o) & (salt_map <= thresh_salt)
    survival_ratio = np.sum(mask) / mask.size
    st.write(f'- 임계치 조건을 만족하는 픽셀 비율: {survival_ratio*100:.2f}%')

if __name__ == '__main__':
    main()

# requirements.txt:
# streamlit
# rasterio
# numpy
# matplotlib
