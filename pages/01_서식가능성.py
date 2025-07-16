# app.py
# requirements.txt 예시:
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
    data = dataset.read()
    return dataset, data

# 2. 시각화 함수 (동적으로 R/G/B 밴드 선택)
def visualize(data):
    n_bands = data.shape[0]
    if n_bands < 3:
        st.error(f"밴드 수가 {n_bands}개로 RGB 합성이 불가능합니다.")
        return
    st.sidebar.header('시각화 옵션')
    default_r, default_g, default_b = 1, n_bands//2, n_bands
    r = st.sidebar.slider('Red 밴드', 1, n_bands, default_r)
    g = st.sidebar.slider('Green 밴드', 1, n_bands, default_g)
    b = st.sidebar.slider('Blue 밴드', 1, n_bands, default_b)
    rgb = np.stack([data[r-1], data[g-1], data[b-1]], axis=2)
    fig, ax = plt.subplots()
    ax.imshow(rgb)
    ax.set_title(f"RGB 합성 (R={r}, G={g}, B={b})")
    ax.axis('off')
    st.pyplot(fig)

# 3. 수분/염분 지표 계산 함수
def analyze_h2o_salt(data):
    n_bands = data.shape[0]
    # 최대 밴드 범위 내에서 선택
    h2o_band = min(100, n_bands)
    salt_band = min(150, n_bands)
    h2o_idx = np.nanmean(data[h2o_band-1].astype(float))
    salt_idx = np.nanmean(data[salt_band-1].astype(float))
    return h2o_idx, salt_idx, h2o_band, salt_band

# 4. 생존 가능성 확률 계산 함수
def calculate_habitability(h2o_idx, salt_idx):
    h2o_score = 1 / (1 + np.exp(-(h2o_idx - 0.5)))
    salt_score = 1 / (1 + np.exp(salt_idx - 0.5))
    return 0.6 * h2o_score + 0.4 * (1 - salt_score)

# 메인 함수
def main():
    st.title('CRISM MTRDR Viewer & Habitability Estimator')
    st.markdown("""
CRISM hyperspectral 파일(.img + .hdr)을 업로드하면:

1. 동적 RGB 합성 시각화
2. 수분/염분 지표 계산
3. 생존 가능성 확률 제공
""")
    files = st.file_uploader('파일 업로드', type=['img', 'hdr'], accept_multiple_files=True)
    if not files or len(files) < 2:
        st.info('*.img* 파일과 대응하는 *.hdr* 파일을 함께 업로드하세요.')
        return

    dataset, data = load_crism(files)
    if dataset is None:
        return

    st.header('1. 데이터 시각화')
    visualize(data)

    st.header('2. 수분/염분 지표 분석')
    h2o_idx, salt_idx, h2o_band, salt_band = analyze_h2o_salt(data)
    st.write(f"• H2O 인덱스 (밴드 {h2o_band}): {h2o_idx:.3f}")
    st.write(f"• Salt 인덱스 (밴드 {salt_band}): {salt_idx:.3f}")

    st.header('3. 생존 가능성 확률')
    prob = calculate_habitability(h2o_idx, salt_idx)
    st.write(f"**생존 가능성: {prob*100:.2f}%**")

if __name__ == '__main__':
    main()
