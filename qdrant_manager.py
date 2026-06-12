
from qdrant_client import QdrantClient
from qdrant_client.http import models
from typing import List, Dict, Tuple
import numpy as np
import os
import logging

logging.basicConfig(level=logging.INFO)

class QdrantVectorDB:
    """Quản lý kết nối và thao тác với Qdrant"""
    
    def __init__(self, collection_name: str = "audio_features_v2", 
                 vector_size: int = 58,
                 host: str = "localhost", 
                 port: int = 6333,
                 path: str = None):
        """
        Khởi tạo Qdrant client
        
        Args:
            collection_name: Tên collection
            vector_size: Kích thước vector (số chiều)
            host: Host của Qdrant server
            port: Port của Qdrant server
            path: Đường dẫn lưu trữ local (nếu dùng in-memory)
        """
        self.collection_name = collection_name
        self.vector_size = vector_size
        
        # Kết nối Qdrant (sử dụng in-memory nếu path không có)
        if path:
            self.client = QdrantClient(path=path)
        else:
            try:
                # Thử kết nối đến server
                self.client = QdrantClient(url="http://localhost:6333")
                logging.info(f"✓ Kết nối đến Qdrant server tại {host}:{port} thành công")
            except Exception as e:
                logging.error(f"Không thể kết nối đến Qdrant server: {e}")
                logging.info("Sẽ sử dụng chế độ in-memory...")
                self.client = QdrantClient(":memory:")
        
        self.connect()
    
    def connect(self):
        """Kết nối và tạo collection nếu chưa tồn tại"""
        try:
            # Kiểm tra collection đã tồn tại
            collections = self.client.get_collections()
            collection_names = [col.name for col in collections.collections]
            
            if self.collection_name not in collection_names:
                self.create_collection()
            else:
                logging.info(f"Collection '{self.collection_name}' đã tồn tại")
        except Exception as e:
            logging.error(f"Lỗi khi kiểm tra collection: {e}")

    def create_collection(self):
        """Tạo collection mới"""
        try:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=self.vector_size,
                    distance=models.Distance.COSINE
                )
            )
            logging.info(f"✓ Tạo collection '{self.collection_name}' thành công")
        except Exception as e:
            logging.error(f"Lỗi khi tạo collection: {e}")
    
    def add_vector(self, vector: np.ndarray, 
                   metadata: Dict = None) -> bool:
        """
        Thêm một vector vào database
        
        Args:
            vector: Vector đặc trưng
            metadata: Metadata (tên file, nhãn, v.v.)
        """
        try:
            # Chuyển numpy array thành list
            vector_list = vector.tolist() if isinstance(vector, np.ndarray) else vector
            
            # Tạo payload
            payload = metadata or {}
            
            # Upsert vector
            self.client.upsert(
                collection_name=self.collection_name,
                points=[
                    models.PointStruct(
                        # id=point_id,
                        vector=vector_list,
                        payload=payload
                    )
                ]
            )
            return True
        except Exception as e:
            logging.error(f"Lỗi khi thêm vector: {e}")
            return False
    
    # def add_vectors_batch(self, vectors: List[np.ndarray], 
    #                      point_ids: List[int],
    #                      metadatas: List[Dict] = None) -> bool:
    #     """
    #     Thêm nhiều vectors cùng một lúc
    #     """
    #     try:
    #         points = []
    #         for i, (vector, point_id) in enumerate(zip(vectors, point_ids)):
    #             vector_list = vector.tolist() if isinstance(vector, np.ndarray) else vector
    #             metadata = metadatas[i] if metadatas else {}
                
    #             points.append(
    #                 models.PointStruct(
    #                     id=point_id,
    #                     vector=vector_list,
    #                     payload=metadata
    #                 )
    #             )
            
    #         self.client.upsert(
    #             collection_name=self.collection_name,
    #             points=points
    #         )
    #         logging.info(f"✓ Thêm {len(points)} vectors thành công")
    
    #         return True
    #     except Exception as e:
    #         logging.error(f"Lỗi khi thêm batch vectors: {e}")
    #         return False
    
    # def search(self, query_vector: np.ndarray, limit: int = 5) -> List[Dict]:
    #     """
    #     Tìm kiếm vectors tương tự
        
    #     Args:
    #         query_vector: Vector truy vấn
    #         limit: Số kết quả trả về
            
    #     Returns:
    #         Danh sách kết quả tìm kiếm với score tương tự
    #     """
    #     try:
    #         vector_list = query_vector.tolist() if isinstance(query_vector, np.ndarray) else query_vector
            
    #         search_results = self.client.search(
    #             collection_name=self.collection_name,
    #             query_vector=vector_list,
    #             limit=limit
    #         )
            
    #         results = []
    #         for result in search_results:
    #             results.append({
    #                 'id': result.id,
    #                 'score': result.score,
    #                 'metadata': result.payload
    #             })
            
    #         return results
    #     except Exception as e:
    #         print(f"Lỗi khi tìm kiếm: {e}")
    #         return []
    
    # def search_filter(self, query_vector: np.ndarray, 
    #                  filter_condition: Dict, limit: int = 5) -> List[Dict]:
    #     """
    #     Tìm kiếm với bộ lọc điều kiện
    #     """
    #     try:
    #         vector_list = query_vector.tolist() if isinstance(query_vector, np.ndarray) else query_vector
            
    #         search_results = self.client.search(
    #             collection_name=self.collection_name,
    #             query_vector=vector_list,
    #             limit=limit,
    #             query_filter=filter_condition
    #         )
            
    #         results = []
    #         for result in search_results:
    #             results.append({
    #                 'id': result.id,
    #                 'score': result.score,
    #                 'metadata': result.payload
    #             })
            
    #         return results
    #     except Exception as e:
    #         print(f"Lỗi khi tìm kiếm với bộ lọc: {e}")
    #         return []
    
    def get_vector(self, point_id: int) -> Dict:
        """Lấy thông tin vector theo ID"""
        try:
            result = self.client.retrieve(
                collection_name=self.collection_name,
                ids=[point_id]
            )
            
            if result:
                return {
                    'id': result[0].id,
                    'vector': result[0].vector,
                    'metadata': result[0].payload
                }
            return None
        except Exception as e:
            logging.error(f"Lỗi khi lấy vector: {e}")
            return None
    
    def delete_vector(self, point_id: int) -> bool:
        """Xóa một vector"""
        try:
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=models.PointIdsList(
                    ids=[point_id]
                )
            )
            return True
        except Exception as e:
            logging.error(f"Lỗi khi xóa vector: {e}")
            return False
    
    def get_collection_info(self) -> Dict:
        """Lấy thông tin collection"""
        try:
            info = self.client.get_collection(self.collection_name)
            return {
                # 'name': info.name,
                'vector_size': info.config.params.vectors.size,
                'vector_count': info.points_count,
                'distance_metric': str(info.config.params.vectors.distance)
            }
        except Exception as e:
            logging.error(f"Lỗi khi lấy thông tin collection: {e}")
            return {}
    
    def delete_collection(self) -> bool:
        """Xóa toàn bộ collection"""
        try:
            self.client.delete_collection(self.collection_name)
            logging.info(f"✓ Xóa collection '{self.collection_name}' thành công")
            return True
        except Exception as e:
            logging.error(f"Lỗi khi xóa collection: {e}")
            return False
