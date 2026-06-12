import streamlit as st
import os
import pandas as pd
import tempfile
import numpy as np
from audio_processor_v2 import AudioPreprocessor
from qdrant_manager import QdrantVectorDB

# Cấu hình trang với giao diện hiện đại
st.set_page_config(
    page_title="Hệ thống Tìm kiếm Âm thanh Thông minh",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS để tạo giao diện premium (Dark mode vibes)
st.markdown("""
<style>
    /* Google Font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .main {
        background-color: #0e1117;
    }
    
    /* Card style cho kết quả */
    .audio-card {
        background-color: #1e2128;
        padding: 24px;
        border-radius: 16px;
        border: 1px solid #30363d;
        margin-bottom: 20px;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .audio-card:hover {
        transform: translateY(-4px);
        border-color: #58a6ff;
        box-shadow: 0 8px 15px rgba(0, 0, 0, 0.2);
    }
    
    .similarity-badge {
        background: linear-gradient(135deg, #238636 0%, #2ea043 100%);
        color: white;
        padding: 6px 14px;
        border-radius: 50px;
        font-size: 0.85rem;
        font-weight: 600;
    }
    
    .label-badge {
        background: linear-gradient(135deg, #1f6feb 0%, #58a6ff 100%);
        color: white;
        padding: 6px 14px;
        border-radius: 50px;
        font-size: 0.85rem;
        font-weight: 600;
        margin-left: 8px;
    }
    
    .rank-number {
        font-size: 1.5rem;
        font-weight: 700;
        color: #8b949e;
        margin-right: 15px;
    }
    
    .file-name {
        color: #c9d1d9;
        font-weight: 600;
        font-size: 1.1rem;
    }

    /* Tùy chỉnh thanh upload */
    .stFileUploader {
        border: 2px dashed #30363d;
        border-radius: 12px;
        padding: 20px;
    }
</style>
""", unsafe_allow_html=True)

# Các hằng số cấu hình
# Lưu ý: Thay đổi đường dẫn này nếu cần thiết để khớp với môi trường của bạn
AUDIO_DATA_PATH = r"c:\Users\ADZ\Downloads\wind_instruments_2s"
CSV_PATH = 'wind_instruments_2s.csv'

@st.cache_resource
def load_resources():
    """Khởi tạo các thành phần core một lần duy nhất"""
    try:
        preprocessor = AudioPreprocessor()
        qdrant_db = QdrantVectorDB()
        
        # Load labels từ CSV
        if os.path.exists(CSV_PATH):
            df = pd.read_csv(CSV_PATH)
            fname_to_label = dict(zip(df['fname'], df['label']))
        else:
            st.warning(f"Không tìm thấy file mapping {CSV_PATH}")
            fname_to_label = {}
            
        return preprocessor, qdrant_db, fname_to_label
    except Exception as e:
        st.error(f"Lỗi khởi tạo hệ thống: {e}")
        return None, None, {}

preprocessor, qdrant_db, fname_to_label = load_resources()

# --- SIDEBAR ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3039/3039391.png", width=80)
    st.title("Cấu hình")
    st.markdown("---")
    
    top_k = st.slider("Số lượng kết quả trả về", 1, 20, 5)
    
    # st.markdown("### Về hệ thống")
    # st.info("""
    # Ứng dụng này sử dụng trích xuất đặc trưng MFCC & F0 kết hợp với 
    # Vector Database (Qdrant) để tìm kiếm các đoạn âm thanh nhạc cụ 
    # hơi tương đồng dựa trên nội dung âm thanh.
    # """)
    
    # if st.button("Làm mới DB"):
    #     st.cache_resource.clear()
    #     st.rerun()

# --- MAIN CONTENT ---
st.title("🎵 Hệ thống tìm kiếm âm thanh nhạc cụ")
st.markdown("Tải lên một file audio (.WAV).")

# Phần Upload
upload_col, info_col = st.columns([2, 1])

with upload_col:
    uploaded_file = st.file_uploader("Kéo thả hoặc chọn file audio .WAV", type=["wav"])

with info_col:
    if uploaded_file:
        st.success("File đã sẵn sàng!")
        st.audio(uploaded_file)
        st.write(f"**Tên file:** `{uploaded_file.name}`")
        st.write(f"**Kích thước:** `{uploaded_file.size / 1024:.1f} KB`")
        # Lấy label từ file CSV
        file_label = fname_to_label.get(uploaded_file.name, "Không xác định")
        st.write(f"**Loại nhạc cụ:** `{file_label}`")

st.divider()

# Xử lý tìm kiếm
if uploaded_file:
    if st.button("🔍 Bắt đầu Tìm kiếm", type="primary", use_container_width=True):
        if preprocessor is None or qdrant_db is None:
            st.error("Hệ thống chưa được khởi tạo đúng cách.")
        else:
            with st.spinner("Đang phân tích đặc trưng âm thanh và truy vấn database..."):
                # 1. Lưu file tạm
                with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp:
                    tmp.write(uploaded_file.getvalue())
                    tmp_path = tmp.name
                
                try:
                    # 2. Trích xuất đặc trưng
                    features = preprocessor.preprocess(tmp_path)
                    
                    if features is not None:
                        # 3. Truy vấn Qdrant
                        # Sử dụng query_points như trong api.py
                        search_results = qdrant_db.client.query_points(
                            collection_name=qdrant_db.collection_name,
                            query=features.tolist(),
                            with_payload=True,
                            limit=top_k
                        ).points
                        
                        # 4. Hiển thị kết quả
                        st.subheader(f"Kết quả tìm kiếm ({len(search_results)})")
                        
                        if not search_results:
                            st.warning("Không tìm thấy kết quả phù hợp.")
                        else:
                            for i, point in enumerate(search_results):
                                payload = point.payload or {}
                                file_name = payload.get('file_name', 'Unknown')
                                label = fname_to_label.get(file_name, 'Unknown')
                                score = point.score
                                # Trong Qdrant với Cosine similarity, score càng cao càng giống
                                similarity = score 
                                
                                # Đường dẫn file audio kết quả
                                result_audio_path = os.path.join(AUDIO_DATA_PATH, file_name)
                                
                                # Layout cho card kết quả
                                st.markdown(f"""
                                <div class="audio-card">
                                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px;">
                                        <div style="display: flex; align-items: center;">
                                            <span class="rank-number">#{i+1}</span>
                                            <span class="file-name"> Tên file: {file_name}</span>
                                        </div>
                                        <div>
                                            <span class="similarity-badge">Độ tương đồng: {similarity:.2%}</span>
                                            <span class="label-badge">Loại nhạc cụ: {label}</span>
                                        </div>
                                    </div>
                                </div>
                                """, unsafe_allow_html=True)
                                
                                # Hiển thị trình phát nhạc bên dưới card (Streamlit không cho phép lồng widget vào HTML markdown)
                                if os.path.exists(result_audio_path):
                                    st.audio(result_audio_path)
                                else:
                                    st.caption(f"⚠️ Không tìm thấy file tại: {result_audio_path}")
                                st.markdown("<br>", unsafe_allow_html=True)
                    else:
                        st.error("Không thể xử lý file âm thanh này. Vui lòng kiểm tra định dạng.")
                
                except Exception as e:
                    st.error(f"Đã xảy ra lỗi trong quá trình tìm kiếm: {e}")
                finally:
                    # Dọn dẹp file tạm
                    if os.path.exists(tmp_path):
                        os.remove(tmp_path)
else:
    # Trạng thái chờ
    st.info("💡 Mẹo: Hãy tải lên một đoạn âm thanh ngắn (2-30s) của nhạc cụ hơi để có kết quả tốt nhất.")
    
    # Hiển thị demo ảnh/icon đẹp mắt
    # st.center = st.columns([1, 1, 1])[1]
    # with st.center:
    #     st.image("https://cdn-icons-png.flaticon.com/512/3659/3659774.png", use_container_width=True)

# Footer info
# st.markdown("---")
# with st.expander("📝 Chi tiết Kỹ thuật"):
#     st.write("**Công nghệ sử dụng:**")
#     st.write("- **Frontend:** Streamlit")
#     st.write("- **Audio Processing:** Librosa (MFCC, Spectral Contrast, pYIN F0)")
#     st.write("- **Vector DB:** Qdrant (Cosine Similarity)")
#     if qdrant_db:
#         try:
#             info = qdrant_db.get_collection_info()
#             st.json(info)
#         except:
#             st.write("Không thể lấy thông tin collection.")
