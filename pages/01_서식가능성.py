# app.py
# requirements.txt:
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

# 1. 파일 로드 함수
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
    return dataset, data

# 2. RGB 시각화 (원본 기반)
def visualize_rgb(data):
    n_bands = data.shape[0]
    if n_bands < 3:
        st.error(f"밴드 수 {n_bands}개로 RGB 합성이 불가능합니다.")
        return
    st.sidebar.header('RGB 밴드 선택')
    r = st.sidebar.slider('Red 밴드', 1, n_bands, 1)
    g = st.sidebar.slider('Green 밴드', 1, n_bands, n_bands//2)
    b = st.sidebar.slider('Blue 밴드', 1, n_bands, n_bands)
    rgb = np.stack([data[r-1], data[g-1], data[b-1]], axis=2)
    vmin, vmax = np.percentile(rgb, 2), np.percentile(rgb, 98)
    fig, ax = plt.subplots(figsize=(6,6))
    ax.imshow(rgb, vmin=vmin, vmax=vmax)
    ax.set_title(f"RGB 합성 (R={r}, G={g}, B={b})")
    ax.axis('off')
    st.pyplot(fig)

# 3. Hydration & Salt ratio 계산
def compute_ratios(data):
    n_bands = data.shape[0]
    st.sidebar.header('지표 밴드 선택')
    # 수분 함유 지표 밴드 (NIR/SWIR)
    nir = st.sidebar.slider('NIR 밴드 (~1.0µm)', 1, n_bands, min(31, n_bands))
    swir_h2o = st.sidebar.slider('SWIR 밴드 (2.8µm)', 1, n_bands, min(256, n_bands))
    # 염분 지표 밴드 (SWIR/Blue)
    swir_salt = st.sidebar.slider('SWIR 밴드 (~2.4µm)', 1, n_bands, min(184, n_bands))
    blue = st.sidebar.slider('Blue 밴드 (~0.5µm)', 1, n_bands, min(6, n_bands))
    # 지표 계산
    with np.errstate(divide='ignore', invalid='ignore'):
        hydration = data[nir-1] / data[swir_h2o-1]
        salt_ratio = data[swir_salt-1] / data[blue-1]
    # 평균
    h2o_idx = np.nanmean(hydration)
    salt_idx = np.nanmean(salt_ratio)
    return hydration, salt_ratio, h2o_idx, salt_idx, (nir, swir_h2o), (swir_salt, blue)

# 4. 생존 확률 계산
def habitability_mask(hydration, salt_ratio, thr_h2o, thr_salt):
    return (hydration > thr_h2o) & (salt_ratio < thr_salt)

# 메인
def main():
    st.title('CRISM Hydration & Salt Mineral Ratio App')
    st.markdown("""
CRISM MTRDR 데이터를 업로드하면:

1. 원본 RGB 시각화
2. Hydration (NIR/SWIR) 및 Salt (SWIR/Blue) 비율 맵
3. 임계치 기반 마스크 및 비율 계산
""")
    files = st.file_uploader('.img + .hdr 파일 업로드', type=['img','hdr'], accept_multiple_files=True)
    if not files or len(files)<2:
        st.info('*.img* 와 대응하는 *.hdr* 파일을 함께 업로드하세요.')
        return

    dataset, data = load_crism(files)
    if dataset is None:
        return

    # 1. RGB
    st.header('1. 원본 RGB 시각화')
    visualize_rgb(data)

    # 2. 지표 맵
    st.header('2. Hydration & Salt Ratio Maps')
    hydration, salt_ratio, h2o_idx, salt_idx, bands_h2o, bands_salt = compute_ratios(data)
    col1, col2 = st.columns(2)
    with col1:
        fig1, ax1 = plt.subplots()
        vmin, vmax = np.nanpercentile(hydration, [5,95])
        cax1 = ax1.imshow(hydration, cmap='viridis', vmin=vmin, vmax=vmax)
        ax1.set_title(f'Hydration Ratio (Band {bands_h2o[0]}/{bands_h2o[1]})')
        ax1.axis('off')
        fig1.colorbar(cax1, ax=ax1, fraction=0.046, pad=0.04)
        st.pyplot(fig1)
        st.write(f'- 평균 Hydration Ratio: {h2o_idx:.3f}')
    with col2:
        fig2, ax2 = plt.subplots()
        vmin2, vmax2 = np.nanpercentile(salt_ratio, [5,95])
        cax2 = ax2.imshow(salt_ratio, cmap='magma', vmin=vmin2, vmax=vmax2)
        ax2.set_title(f'Salt Ratio (Band {bands_salt[0]}/{bands_salt[1]})')
        ax2.axis('off')
        fig2.colorbar(cax2, ax=ax2, fraction=0.046, pad=0.04)
        st.pyplot(fig2)
        st.write(f'- 평균 Salt Ratio: {salt_idx:.3f}')

    # 3. 임계치 및 마스크
    st.header('3. Threshold & Mask')
    # Hydration: 최소값을 넘어야 함
    thr_h2o = st.slider('Hydration Minimum Threshold', 0.6, 2.0, 1.1, 0.01,
                        help='Hydration ratio이 이 값 이상인 픽셀만 선택')
    # Salt: 특정 구간 내에 있어야 함
    thr_salt = st.slider('Salt Acceptable Range', 0.3, 1.5, (0.3, 0.6), 0.01,
                         help='Salt ratio이 이 범위 내에 있는 픽셀만 선택')
    # 임계치 적용: hydration >= thr_h2o, thr_salt[0] <= salt_ratio <= thr_salt[1]
    mask = (hydration >= thr_h2o) & (salt_ratio >= thr_salt[0]) & (salt_ratio <= thr_salt[1])
    fig3, ax3 = plt.subplots()
    ax3.imshow(mask, cmap='gray')
    ax3.set_title('Survivability Mask')
    ax3.axis('off')
    st.pyplot(fig3)
    ratio = np.sum(mask) / mask.size
    st.write(f'- 임계치 만족 픽셀 비율: {ratio*100:.2f}%')

if __name__=='__main__':
    main()
