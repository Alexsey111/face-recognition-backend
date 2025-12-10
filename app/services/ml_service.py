"""
Сервис машинного обучения.
Интеграция с ML пайплайном для обработки изображений и распознавания лиц.
"""

import io
import base64
from typing import Optional, Dict, Any, List, Tuple
import numpy as np
import cv2
from PIL import Image
import httpx
import asyncio
from datetime import datetime

from ..config import settings
from ..utils.logger import get_logger
from ..utils.exceptions import ProcessingError, MLServiceError

logger = get_logger(__name__)


class MLService:
    """
    Сервис для работы с машинным обучением.
    """
    
    def __init__(self):
        self.ml_service_url = settings.ML_SERVICE_URL
        self.ml_service_timeout = settings.ML_SERVICE_TIMEOUT
        self.ml_service_api_key = settings.ML_SERVICE_API_KEY
        self.client = httpx.AsyncClient(timeout=self.ml_service_timeout)
    
    async def health_check(self) -> bool:
        """
        Проверка состояния ML сервиса.
        
        Returns:
            bool: True если ML сервис доступен
        """
        try:
            response = await self.client.get(
                f"{self.ml_service_url}/health",
                headers=self._get_headers()
            )
            return response.status_code == 200
        except Exception as e:
            logger.error(f"ML service health check failed: {str(e)}")
            return False
    
    async def generate_embedding(self, image_data: bytes) -> Dict[str, Any]:
        """
        Генерация эмбеддинга лица из изображения.
        
        Args:
            image_data: Двоичные данные изображения
            
        Returns:
            Dict[str, Any]: Результат генерации эмбеддинга
        """
        try:
            logger.info("Generating face embedding")
            
            # Подготавливаем данные для запроса
            payload = {
                "image": base64.b64encode(image_data).decode('utf-8'),
                "model_version": "v1.0",
                "return_metadata": True
            }
            
            # Отправляем запрос к ML сервису
            response = await self.client.post(
                f"{self.ml_service_url}/embeddings/generate",
                json=payload,
                headers=self._get_headers()
            )
            
            if response.status_code != 200:
                error_msg = f"ML service returned status {response.status_code}: {response.text}"
                logger.error(error_msg)
                raise MLServiceError(error_msg)
            
            result = response.json()
            
            # Проверяем результат
            if not result.get("success", False):
                raise MLServiceError(f"Embedding generation failed: {result.get('error', 'Unknown error')}")
            
            # Валидируем результат
            embedding = result.get("embedding")
            if not embedding:
                raise MLServiceError("No embedding in response")
            
            logger.info(f"Embedding generated successfully (dimension: {len(embedding)})")
            
            return {
                "success": True,
                "embedding": np.array(embedding, dtype=np.float32),
                "quality_score": result.get("quality_score", 0.0),
                "face_detected": result.get("face_detected", False),
                "multiple_faces": result.get("multiple_faces", False),
                "model_version": result.get("model_version", "unknown"),
                "processing_time": result.get("processing_time", 0.0)
            }
            
        except httpx.TimeoutException:
            error_msg = "ML service timeout"
            logger.error(error_msg)
            raise MLServiceError(error_msg)
        except Exception as e:
            logger.error(f"Embedding generation failed: {str(e)}")
            raise MLServiceError(f"Failed to generate embedding: {str(e)}")
    
    async def verify_face(
        self,
        image_data: bytes,
        reference_embedding: np.ndarray,
        threshold: float = 0.8
    ) -> Dict[str, Any]:
        """
        Верификация лица по эталонному эмбеддингу.
        
        Args:
            image_data: Двоичные данные изображения
            reference_embedding: Эталонный эмбеддинг
            threshold: Порог схожести
            
        Returns:
            Dict[str, Any]: Результат верификации
        """
        try:
            logger.info("Verifying face")
            
            # Генерируем эмбеддинг для изображения
            embedding_result = await self.generate_embedding(image_data)
            
            if not embedding_result.get("face_detected"):
                return {
                    "success": True,
                    "verified": False,
                    "confidence": 0.0,
                    "similarity_score": 0.0,
                    "face_detected": False,
                    "error": "No face detected in image"
                }
            
            # Вычисляем схожесть
            current_embedding = embedding_result["embedding"]
            similarity_score = self._compute_similarity(current_embedding, reference_embedding)
            
            # Определяем результат верификации
            verified = similarity_score >= threshold
            
            logger.info(f"Face verification: {verified} (similarity: {similarity_score:.3f}, threshold: {threshold})")
            
            return {
                "success": True,
                "verified": verified,
                "confidence": similarity_score,
                "similarity_score": similarity_score,
                "threshold": threshold,
                "face_detected": True,
                "face_quality": embedding_result.get("quality_score", 0.0),
                "distance": self._compute_distance(current_embedding, reference_embedding),
                "model_version": embedding_result.get("model_version", "unknown"),
                "processing_time": embedding_result.get("processing_time", 0.0)
            }
            
        except Exception as e:
            logger.error(f"Face verification failed: {str(e)}")
            raise MLServiceError(f"Failed to verify face: {str(e)}")
    
    async def compare_faces(
        self,
        image_data: bytes,
        reference_embedding: np.ndarray,
        threshold: float = 0.8
    ) -> Dict[str, Any]:
        """
        Сравнение лица с эталонным эмбеддингом.
        
        Args:
            image_data: Двоичные данные изображения
            reference_embedding: Эталонный эмбеддинг
            threshold: Порог схожести
            
        Returns:
            Dict[str, Any]: Результат сравнения
        """
        try:
            logger.info("Comparing faces")
            
            # Генерируем эмбеддинг для изображения
            embedding_result = await self.generate_embedding(image_data)
            
            if not embedding_result.get("face_detected"):
                return {
                    "success": True,
                    "similarity_score": 0.0,
                    "face_detected": False,
                    "error": "No face detected in image"
                }
            
            # Вычисляем схожесть
            current_embedding = embedding_result["embedding"]
            similarity_score = self._compute_similarity(current_embedding, reference_embedding)
            distance = self._compute_distance(current_embedding, reference_embedding)
            
            logger.info(f"Face comparison: similarity {similarity_score:.3f}, distance {distance:.3f}")
            
            return {
                "success": True,
                "similarity_score": similarity_score,
                "distance": distance,
                "face_detected": True,
                "multiple_faces": embedding_result.get("multiple_faces", False),
                "face_quality": embedding_result.get("quality_score", 0.0),
                "model_version": embedding_result.get("model_version", "unknown"),
                "processing_time": embedding_result.get("processing_time", 0.0)
            }
            
        except Exception as e:
            logger.error(f"Face comparison failed: {str(e)}")
            raise MLServiceError(f"Failed to compare faces: {str(e)}")
    
    async def check_liveness(
        self,
        image_data: bytes,
        challenge_type: str = "passive",
        challenge_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Проверка живости лица.
        
        Args:
            image_data: Двоичные данные изображения
            challenge_type: Тип проверки (passive, active, blink, smile, turn_head)
            challenge_data: Данные для активной проверки
            
        Returns:
            Dict[str, Any]: Результат проверки живости
        """
        try:
            logger.info(f"Checking liveness (type: {challenge_type})")
            
            # Подготавливаем данные для запроса
            payload = {
                "image": base64.b64encode(image_data).decode('utf-8'),
                "challenge_type": challenge_type,
                "challenge_data": challenge_data or {},
                "return_details": True
            }
            
            # Отправляем запрос к ML сервису
            response = await self.client.post(
                f"{self.ml_service_url}/liveness/check",
                json=payload,
                headers=self._get_headers()
            )
            
            if response.status_code != 200:
                error_msg = f"ML service returned status {response.status_code}: {response.text}"
                logger.error(error_msg)
                raise MLServiceError(error_msg)
            
            result = response.json()
            
            # Проверяем результат
            if not result.get("success", False):
                raise MLServiceError(f"Liveness check failed: {result.get('error', 'Unknown error')}")
            
            logger.info(f"Liveness check completed: {result.get('liveness_detected', False)} (confidence: {result.get('confidence', 0.0):.3f})")
            
            return {
                "success": True,
                "liveness_detected": result.get("liveness_detected", False),
                "confidence": result.get("confidence", 0.0),
                "anti_spoofing_score": result.get("anti_spoofing_score"),
                "face_detected": result.get("face_detected", False),
                "multiple_faces": result.get("multiple_faces", False),
                "image_quality": result.get("image_quality"),
                "recommendations": result.get("recommendations", []),
                "depth_analysis": result.get("depth_analysis"),
                "model_version": result.get("model_version", "unknown"),
                "processing_time": result.get("processing_time", 0.0)
            }
            
        except httpx.TimeoutException:
            error_msg = "ML service timeout"
            logger.error(error_msg)
            raise MLServiceError(error_msg)
        except Exception as e:
            logger.error(f"Liveness check failed: {str(e)}")
            raise MLServiceError(f"Failed to check liveness: {str(e)}")
    
    async def process_image(self, image_data: bytes, image_url: Optional[str] = None) -> Dict[str, Any]:
        """
        Общая обработка изображения.
        
        Args:
            image_data: Двоичные данные изображения
            image_url: URL изображения (опционально)
            
        Returns:
            Dict[str, Any]: Результат обработки
        """
        try:
            logger.info("Processing image")
            
            # Подготавливаем данные для запроса
            payload = {
                "image": base64.b64encode(image_data).decode('utf-8'),
                "image_url": image_url,
                "return_analysis": True,
                "return_metadata": True
            }
            
            # Отправляем запрос к ML сервису
            response = await self.client.post(
                f"{self.ml_service_url}/image/process",
                json=payload,
                headers=self._get_headers()
            )
            
            if response.status_code != 200:
                error_msg = f"ML service returned status {response.status_code}: {response.text}"
                logger.error(error_msg)
                raise MLServiceError(error_msg)
            
            result = response.json()
            
            # Проверяем результат
            if not result.get("success", False):
                raise MLServiceError(f"Image processing failed: {result.get('error', 'Unknown error')}")
            
            logger.info(f"Image processing completed successfully")
            
            return {
                "success": True,
                "face_detected": result.get("face_detected", False),
                "multiple_faces": result.get("multiple_faces", False),
                "quality_score": result.get("quality_score", 0.0),
                "image_analysis": result.get("image_analysis", {}),
                "model_version": result.get("model_version", "unknown"),
                "processing_time": result.get("processing_time", 0.0)
            }
            
        except httpx.TimeoutException:
            error_msg = "ML service timeout"
            logger.error(error_msg)
            raise MLServiceError(error_msg)
        except Exception as e:
            logger.error(f"Image processing failed: {str(e)}")
            raise MLServiceError(f"Failed to process image: {str(e)}")
    
    async def detect_faces(self, image_data: bytes) -> Dict[str, Any]:
        """
        Обнаружение лиц на изображении.
        
        Args:
            image_data: Двоичные данные изображения
            
        Returns:
            Dict[str, Any]: Результат обнаружения лиц
        """
        try:
            logger.info("Detecting faces")
            
            # Подготавливаем данные для запроса
            payload = {
                "image": base64.b64encode(image_data).decode('utf-8'),
                "return_coordinates": True,
                "return_confidence": True
            }
            
            # Отправляем запрос к ML сервису
            response = await self.client.post(
                f"{self.ml_service_url}/faces/detect",
                json=payload,
                headers=self._get_headers()
            )
            
            if response.status_code != 200:
                error_msg = f"ML service returned status {response.status_code}: {response.text}"
                logger.error(error_msg)
                raise MLServiceError(error_msg)
            
            result = response.json()
            
            # Проверяем результат
            if not result.get("success", False):
                raise MLServiceError(f"Face detection failed: {result.get('error', 'Unknown error')}")
            
            faces_count = len(result.get("faces", []))
            logger.info(f"Face detection completed: {faces_count} faces found")
            
            return {
                "success": True,
                "faces_count": faces_count,
                "faces": result.get("faces", []),
                "image_dimensions": result.get("image_dimensions", {}),
                "model_version": result.get("model_version", "unknown"),
                "processing_time": result.get("processing_time", 0.0)
            }
            
        except httpx.TimeoutException:
            error_msg = "ML service timeout"
            logger.error(error_msg)
            raise MLServiceError(error_msg)
        except Exception as e:
            logger.error(f"Face detection failed: {str(e)}")
            raise MLServiceError(f"Failed to detect faces: {str(e)}")
    
    async def batch_process_images(self, images_data: List[bytes]) -> List[Dict[str, Any]]:
        """
        Пакетная обработка изображений.
        
        Args:
            images_data: Список изображений для обработки
            
        Returns:
            List[Dict[str, Any]]: Результаты обработки
        """
        try:
            logger.info(f"Batch processing {len(images_data)} images")
            
            # Подготавливаем данные для запроса
            payload = {
                "images": [base64.b64encode(img).decode('utf-8') for img in images_data],
                "return_analysis": True
            }
            
            # Отправляем запрос к ML сервису
            response = await self.client.post(
                f"{self.ml_service_url}/images/batch-process",
                json=payload,
                headers=self._get_headers()
            )
            
            if response.status_code != 200:
                error_msg = f"ML service returned status {response.status_code}: {response.text}"
                logger.error(error_msg)
                raise MLServiceError(error_msg)
            
            result = response.json()
            
            # Проверяем результат
            if not result.get("success", False):
                raise MLServiceError(f"Batch processing failed: {result.get('error', 'Unknown error')}")
            
            results = result.get("results", [])
            logger.info(f"Batch processing completed: {len(results)} results")
            
            return results
            
        except httpx.TimeoutException:
            error_msg = "ML service timeout"
            logger.error(error_msg)
            raise MLServiceError(error_msg)
        except Exception as e:
            logger.error(f"Batch processing failed: {str(e)}")
            raise MLServiceError(f"Failed to batch process images: {str(e)}")
    
    async def get_model_info(self) -> Dict[str, Any]:
        """
        Получение информации о модели.
        
        Returns:
            Dict[str, Any]: Информация о модели
        """
        try:
            response = await self.client.get(
                f"{self.ml_service_url}/model/info",
                headers=self._get_headers()
            )
            
            if response.status_code != 200:
                error_msg = f"ML service returned status {response.status_code}: {response.text}"
                logger.error(error_msg)
                raise MLServiceError(error_msg)
            
            result = response.json()
            
            logger.info("Model info retrieved successfully")
            return result
            
        except Exception as e:
            logger.error(f"Failed to get model info: {str(e)}")
            raise MLServiceError(f"Failed to get model info: {str(e)}")
    
    def _get_headers(self) -> Dict[str, str]:
        """
        Получение заголовков для запросов к ML сервису.
        
        Returns:
            Dict[str, str]: Заголовки HTTP
        """
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "FaceRecognitionService/1.0"
        }
        
        if self.ml_service_api_key:
            headers["Authorization"] = f"Bearer {self.ml_service_api_key}"
        
        return headers
    
    def _compute_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Вычисление косинусной схожести между эмбеддингами.
        
        Args:
            embedding1: Первый эмбеддинг
            embedding2: Второй эмбеддинг
            
        Returns:
            float: Схожесть от 0 до 1
        """
        try:
            # Нормализуем векторы
            embedding1_norm = embedding1 / np.linalg.norm(embedding1)
            embedding2_norm = embedding2 / np.linalg.norm(embedding2)
            
            # Вычисляем косинусную схожесть
            similarity = np.dot(embedding1_norm, embedding2_norm)
            
            # Ограничиваем диапазон
            return float(np.clip(similarity, 0.0, 1.0))
            
        except Exception as e:
            logger.error(f"Error computing similarity: {str(e)}")
            return 0.0
    
    def _compute_distance(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Вычисление евклидового расстояния между эмбеддингами.
        
        Args:
            embedding1: Первый эмбеддинг
            embedding2: Второй эмбеддинг
            
        Returns:
            float: Евклидово расстояние
        """
        try:
            distance = np.linalg.norm(embedding1 - embedding2)
            return float(distance)
            
        except Exception as e:
            logger.error(f"Error computing distance: {str(e)}")
            return float('inf')
    
    async def close(self):
        """
        Закрытие клиента HTTP.
        """
        await self.client.aclose()