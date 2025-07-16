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
# .img/.hdr 로드 및 파장 정보(wavelengths) 파싱

def load_crism(files):
    tmp_dir = tempfile.mkdtemp()
    img_path = None
    for uploaded in files:
        ext = os.path.splitext(uploaded.name)[1].lower()
        out_path = os.path.join(tmp_dir, uploaded.name)
        with open(out_path, 'wb') as f:
            f.write(uploaded.read())
        if ext == '.img':
            img_path = out_path
    if img_path is None:
        st.error(".img 파일이 필요합니다.")
        return None, None, None
    ds = rasterio.open(img_path)
    data = ds.read().astype(float)
    # 파장 정보 추출 시도
    wavelengths = None
    tags = ds.tags()
    if 'wavelength' in tags:
        try:
            wavelengths = np.array(tags['wavelength'].split(), dtype=float)
        except:
            pass
    if wavelengths is None and ds.descriptions:
        try:
            wavelengths = np.array([float(d) for d in ds.descriptions], dtype=float)
        except:
            pass
    return ds, data, wavelengths

# 2. 원본 RGB 시각화

def visualize_rgb(data):
    bands = data.shape[0]
    if bands < 3:
        st.error(f"밴드 수 {bands}개로 RGB 합성이 불가능합니다.")
        return
    st.sidebar.header('RGB 밴드 선택')
    r = st.sidebar.slider('Red 밴드', 1, bands, 1)
    g = st.sidebar.slider('Green 밴드', 1, bands, bands//2)
    b = st.sidebar.slider('Blue 밴드', 1, bands, bands)
    img = np.stack([data[r-1], data[g-1], data[b-1]], axis=2)
    vmin, vmax = np.percentile(img, 2), np.percentile(img, 98)
    fig, ax = plt.subplots(figsize=(6,6))
    ax.imshow(img, vmin=vmin, vmax=vmax)
    ax.set_title(f"RGB 합성 (R={r}, G={g}, B={b})")
    ax.axis('off')
    st.pyplot(fig)

# 3. Hydration & Salt Ratio 계산

def compute_ratios(data, wavelengths=None):
    bands = data.shape[0]
    st.sidebar.header('지표 밴드 선택 및 자동 탐색')
    def find_band(target):
        if wavelengths is None:
            return None
        idx = np.argmin(np.abs(wavelengths - target)) + 1
        return int(idx)
    # 기본값: 자동 또는 전형적
    default_nir = find_band(1.027) or min(31, bands)
    default_swir_h2o = find_band(2.780) or min(256, bands)
    default_swir_salt = find_band(2.445) or min(184, bands)
    default_blue = find_band(0.535) or min(6, bands)
    # 슬라이더
    nir = st.sidebar.slider('NIR 밴드 (~1.0µm)', 1, bands, default_nir)
    swir_h2o = st.sidebar.slider('SWIR H2O 밴드 (2.8µm)', 1, bands, default_swir_h2o)
    swir_salt = st.sidebar.slider('SWIR Salt 밴드 (~2.4µm)', 1, bands, default_swir_salt)
    blue = st.sidebar.slider('Blue 밴드 (~0.5µm)', 1, bands, default_blue)
    # 계산
    with np.errstate(divide='ignore', invalid='ignore'):
        hydration = data[nir-1] / data[swir_h2o-1]
        salt_ratio = data[swir_salt-1] / data[blue-1]
    hydration[np.isinf(hydration)] = np.nan
    salt_ratio[np.isinf(salt_ratio)] = np.nan
    return hydration, salt_ratio, np.nanmean(hydration), np.nanmean(salt_ratio), (nir, swir_h2o), (swir_salt, blue)

# 4. 메인 앱

def main():
    st.title('CRISM Hydration & Salt Ratio Analyzer')
    st.markdown("""
CRISM MTRDR 데이터를 업로드하면:

1. 원본 RGB 시각화
2. Hydration (NIR/SWIR) 및 Salt (SWIR/Blue) 비율 맵
3. 임계치 기반 마스크 및 비율 계산
""")
    files = st.file_uploader('Upload .img + .hdr files', type=['img','hdr'], accept_multiple_files=True)
    if not files or len(files) < 2:
        st.info('*.img*와 대응하는 *.hdr* 파일을 함께 업로드하세요.')
        return
    ds, data, wavelengths = load_crism(files)
    if ds is None:
        return
    # 1
    st.header('1. 원본 RGB 시각화')
    visualize_rgb(data)
    # 2
    st.header('2. Hydration & Salt Ratio Maps')
    hydration, salt_ratio, h2o_mean, salt_mean, bands_h2o, bands_salt = compute_ratios(data, wavelengths)
    c1, c2 = st.columns(2)
    with c1:
        fig1, ax1 = plt.subplots()
        v1, v2 = np.nanpercentile(hydration, [5,95])
        c1_img = ax1.imshow(hydration, cmap='viridis', vmin=v1, vmax=v2)
        ax1.set_title(f'Hydration Ratio (Band {bands_h2o[0]}/{bands_h2o[1]})')
        ax1.axis('off')
        fig1.colorbar(c1_img, ax=ax1, fraction=0.046, pad=0.04)
        st.pyplot(fig1)
        st.write(f'- 평균 Hydration Ratio: {h2o_mean:.3f}')
    with c2:
        fig2, ax2 = plt.subplots()
        u1, u2 = np.nanpercentile(salt_ratio, [5,95])
        c2_img = ax2.imshow(salt_ratio, cmap='magma', vmin=u1, vmax=u2)
        ax2.set_title(f'Salt Ratio (Band {bands_salt[0]}/{bands_salt[1]})')
        ax2.axis('off')
        fig2.colorbar(c2_img, ax=ax2, fraction=0.046, pad=0.04)
        st.pyplot(fig2)
        st.write(f'- 평균 Salt Ratio: {salt_mean:.3f}')
    # 3
    st.header('3. Threshold & Mask')
    thr_h2o = st.slider('Hydration Min Threshold', 0.6, 2.0, 1.1, 0.01)
    thr_salt = st.slider('Salt Acceptable Range', 0.3, 1.5, (0.3, 0.6), 0.01)
    mask = (hydration >= thr_h2o) & (salt_ratio >= thr_salt[0]) & (salt_ratio <= thr_salt[1])
    fig3, ax3 = plt.subplots()
    ax3.imshow(mask, cmap='gray')
    ax3.set_title('Survivability Mask')
    ax3.axis('off')
    st.pyplot(fig3)
    ratio = np.nansum(mask) / mask.size
    st.write(f'- 임계치 만족 픽셀 비율: {ratio*100:.2f}%')

if __name__ == '__main__':
    main()
