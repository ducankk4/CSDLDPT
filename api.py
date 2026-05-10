from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import shutil
import tempfile
import os
import pandas as pd
import numpy as np
from pathlib import Path
from audio_processor import AudioPreprocessor
from qdrant_manager import QdrantVectorDB
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Audio Search API", version="1.0.0")

# Khởi tạo global instances
audio_preprocessor = None
qdrant_vectordb = None
client = None
collection_name = None
wind_df = None
fname_to_label = None

@app.on_event("startup")
async def startup_event():
    """Khởi tạo các thành phần khi ứng dụng khởi động"""
    global audio_preprocessor, qdrant_vectordb, client, collection_name, wind_df, fname_to_label
    
    try:
        # Khởi tạo audio preprocessor
        audio_preprocessor = AudioPreprocessor()
        logger.info("✓ AudioPreprocessor khởi tạo thành công")
        
        # Khởi tạo Qdrant
        qdrant_vectordb = QdrantVectorDB()
        client = qdrant_vectordb.client
        collection_name = qdrant_vectordb.collection_name
        logger.info(f"✓ Qdrant khởi tạo thành công (Collection: {collection_name})")
        
        # Load mapping từ CSV
        csv_path = 'wind_instruments_1s.csv'
        if os.path.exists(csv_path):
            wind_df = pd.read_csv(csv_path)
            fname_to_label = dict(zip(wind_df['fname'], wind_df['label']))
            logger.info(f"✓ Đã load {len(fname_to_label)} file từ CSV")
        else:
            logger.warning(f"⚠ File {csv_path} không tìm thấy")
            fname_to_label = {}
    except Exception as e:
        logger.error(f"✗ Lỗi khởi tạo: {e}")
        raise

@app.get("/health")
async def health_check():
    """Kiểm tra trạng thái API"""
    return {
        "status": "healthy",
        "audio_preprocessor": audio_preprocessor is not None,
        "qdrant_client": client is not None,
        "collection": collection_name
    }

@app.post("/search")
async def search_audio(file: UploadFile = File(...), top_k: int = 5):
    """
    Tìm kiếm các file audio tương đồng
    
    Args:
        file: File audio được upload
        top_k: Số kết quả trả về (mặc định 5)
    
    Returns:
        Danh sách các file tương đồng với metadata
    """
    if audio_preprocessor is None or client is None:
        raise HTTPException(status_code=500, detail="Ứng dụng chưa sẵn sàng")
    
    temp_file = None
    try:
        # Lưu file tạm thời
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp:
            content = await file.read()
            temp.write(content)
            temp_file = temp.name
        
        # Trích xuất features từ file upload
        features = audio_preprocessor.preprocess(temp_file)
        if features is None:
            raise HTTPException(status_code=400, detail="Không thể xử lý file audio")
        
        # Convert numpy array to list for JSON serialization
        features_list = features.tolist()
        
        # Tìm kiếm trong Qdrant
        search_results = client.query_points(
            collection_name=collection_name,
            query=features_list,
            with_payload=True,
            limit=top_k
        ).points
        
        # Chuẩn bị kết quả
        results = []
        for i, point in enumerate(search_results, 1):
            payload = point.payload or {}
            file_name = payload.get('file_name', 'unknown')
            label = fname_to_label.get(file_name, 'unknown') if fname_to_label else 'unknown'
            
            results.append({
                'rank': i,
                'file_name': file_name,
                'file_path': payload.get('file_path', ''),
                'label': label,
                'distance': float(point.score),
                'similarity': float(1 - point.score)  # Chuyển distance thành similarity
            })
        
        return {
            "status": "success",
            "query_file": file.filename,
            "results_count": len(results),
            "results": results
        }
    
    except Exception as e:
        logger.error(f"Lỗi khi tìm kiếm: {e}")
        raise HTTPException(status_code=500, detail=f"Lỗi: {str(e)}")
    
    finally:
        # Xóa file tạm thời
        if temp_file and os.path.exists(temp_file):
            os.remove(temp_file)

@app.get("/stats")
async def get_stats():
    """Lấy thông tin về collection"""
    if client is None:
        raise HTTPException(status_code=500, detail="Qdrant client chưa sẵn sàng")
    
    try:
        collection = client.get_collection(collection_name)
        return {
            "collection_name": collection_name,
            "point_count": collection.points_count,
            "vector_size": collection.config.params.vectors.size
        }
    except Exception as e:
        logger.error(f"Lỗi khi lấy stats: {e}")
        raise HTTPException(status_code=500, detail=f"Lỗi: {str(e)}")

@app.get("/labels")
async def get_labels():
    """Lấy danh sách các nhãn (labels) có sẵn"""
    if fname_to_label is None:
        return {"labels": []}
    
    unique_labels = sorted(set(fname_to_label.values()))
    return {"labels": unique_labels}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
