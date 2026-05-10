import streamlit as st
import requests
import os
from pathlib import Path
import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
# Cấu hình Streamlit
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Audio Search Engine",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────────────────────────────────────
# CSS Custom
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
    <style>
    .main { padding: 2rem; }
    .title { 
        font-size: 2.5rem; 
        font-weight: bold;
        color: #FF6B6B;
        margin-bottom: 0.5rem;
    }
    .subtitle {
        font-size: 1rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .result-card {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border-left: 4px solid #FF6B6B;
        margin-bottom: 1rem;
    }
    .rank-badge {
        display: inline-block;
        background-color: #FF6B6B;
        color: white;
        padding: 0.3rem 0.6rem;
        border-radius: 50%;
        font-weight: bold;
        margin-right: 0.5rem;
    }
    .similarity-bar {
        background-color: #e9ecef;
        height: 1.5rem;
        border-radius: 0.25rem;
        overflow: hidden;
    }
    .similarity-fill {
        background: linear-gradient(90deg, #FF6B6B, #FFE66D);
        height: 100%;
        display: flex;
        align-items: center;
        justify-content: center;
        color: white;
        font-weight: bold;
        font-size: 0.8rem;
    }
    </style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Tiêu đề
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<p class="title">🎵 Audio Search Engine</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Tìm kiếm các file âm thanh tương đồng bằng AI</p>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Di Sidebar - Cấu hình
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Cấu hình")
    
    api_url = st.text_input(
        "API URL",
        value="http://localhost:8000",
        help="Địa chỉ của FastAPI server"
    )
    
    top_k = st.slider(
        "Số kết quả cần trả về",
        min_value=1,
        max_value=20,
        value=5,
        help="Chọn số lượng kết quả tương đồng"
    )
    
    # Kiểm tra server
    st.divider()
    st.subheader("📊 Trạng thái Server")
    
    try:
        response = requests.get(f"{api_url}/health", timeout=5)
        if response.status_code == 200:
            st.success("✓ Server đang hoạt động")
            health_data = response.json()
            
            with st.expander("📈 Chi tiết"):
                st.json(health_data)
                
                # Stats
                try:
                    stats = requests.get(f"{api_url}/stats", timeout=5).json()
                    st.divider()
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Points", stats.get('point_count', 0))
                    with col2:
                        st.metric("Vector Size", stats.get('vector_size', 0))
                except:
                    pass
        else:
            st.error("✗ Server trả về lỗi")
    except Exception as e:
        st.error(f"✗ Không thể kết nối: {str(e)}")

# ─────────────────────────────────────────────────────────────────────────────
# Main Content - Upload và Search
# ─────────────────────────────────────────────────────────────────────────────
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📤 Upload File Audio")
    uploaded_file = st.file_uploader(
        "Chọn file audio (WAV, MP3, etc.)",
        type=["wav", "mp3", "flac", "ogg"]
    )

with col2:
    st.subheader("🔍 Tìm kiếm")
    search_button = st.button("🚀 Tìm kiếm", use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# Xử lý tìm kiếm
# ─────────────────────────────────────────────────────────────────────────────
if search_button:
    if uploaded_file is None:
        st.error("❌ Vui lòng chọn file audio trước")
    else:
        try:
            # Hiển thị thông tin file
            st.info(f"📁 File: {uploaded_file.name} ({uploaded_file.size / 1024:.1f} KB)")
            
            # Gửi request tới API
            with st.spinner("🔄 Đang xử lý..."):
                files = {'file': uploaded_file}
                params = {'top_k': top_k}
                
                response = requests.post(
                    f"{api_url}/search",
                    files=files,
                    params=params,
                    timeout=30
                )
            
            if response.status_code == 200:
                data = response.json()
                
                st.divider()
                st.success(f"✓ Tìm thấy {data['results_count']} kết quả tương đồng")
                st.divider()
                
                # Hiển thị kết quả
                for result in data['results']:
                    with st.container():
                        col1, col2 = st.columns([0.5, 2.5])
                        
                        with col1:
                            st.markdown(f"<span class='rank-badge'>#{result['rank']}</span>", 
                                      unsafe_allow_html=True)
                        
                        with col2:
                            st.markdown(f"**{result['file_name']}**")
                            col_a, col_b, col_c = st.columns(3)
                            with col_a:
                                st.metric("Nhãn", result['label'], label_visibility="collapsed")
                            with col_b:
                                st.metric("Distance", f"{result['distance']:.4f}", label_visibility="collapsed")
                            with col_c:
                                st.metric("Tương đồng", f"{result['similarity']*100:.1f}%", label_visibility="collapsed")
                        
                        # Similarity bar
                        similarity_pct = result['similarity'] * 100
                        bar_html = f"""
                            <div class='similarity-bar'>
                                <div class='similarity-fill' style='width: {similarity_pct}%'>
                                    {similarity_pct:.1f}%
                                </div>
                            </div>
                        """
                        st.markdown(bar_html, unsafe_allow_html=True)
                        
                        # Đường dẫn file
                        with st.expander("📂 Đường dẫn & Chi tiết"):
                            st.code(result['file_path'], language="text")
                        
                        st.divider()
                
                # Bảng tóm tắt
                st.subheader("📋 Bảng kết quả")
                df_results = pd.DataFrame(data['results'])
                df_display = df_results[['rank', 'file_name', 'label', 'similarity']].copy()
                df_display['similarity'] = df_display['similarity'].apply(lambda x: f"{x*100:.2f}%")
                df_display.columns = ['Hạng', 'Tên file', 'Nhãn', 'Tương đồng']
                
                st.dataframe(df_display, use_container_width=True, hide_index=True)
                
            else:
                st.error(f"❌ Lỗi từ server: {response.status_code}")
                st.error(response.json() if response.text else "Không có thông báo lỗi")
        
        except requests.exceptions.ConnectionError:
            st.error("❌ Không thể kết nối đến server. Vui lòng kiểm tra API URL")
        except Exception as e:
            st.error(f"❌ Lỗi: {str(e)}")

# ─────────────────────────────────────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────────────────────────────────────
st.divider()
st.caption("🎵 Audio Search Engine v1.0 | Powered by Qdrant Vector DB + FastAPI")
