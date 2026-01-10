"""
E2E тесты для полного workflow Face Recognition API.
Тестирует complete flow: upload → reference → verify → liveness
"""

import pytest
import uuid
from unittest.mock import Mock, patch, AsyncMock
from fastapi.testclient import TestClient
from datetime import datetime, timezone
import base64
from PIL import Image
import io

# Создаем тестовое изображение
def create_test_image_data():
    """Создает тестовое изображение в формате base64"""
    img = Image.new('RGB', (224, 224), color='blue')
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='JPEG')
    return base64.b64encode(img_bytes.getvalue()).decode()


class TestCompleteWorkflow:
    """E2E тесты для полного workflow"""
    
    def setup_method(self):
        """Настройка для каждого теста"""
        from app.main import create_test_app
        self.app = create_test_app()
        self.client = TestClient(self.app)
        self.test_image_data = create_test_image_data()
        self.test_user_id = f"user_{uuid.uuid4().hex[:8]}"
        self.test_session_id = f"session_{uuid.uuid4().hex[:8]}"
    
    def test_complete_reference_verify_liveness_workflow(self):
        """Тест полного workflow: создание эталона → верификация → проверка живости"""
        
        with patch('app.routes.upload.settings'), \
             patch('app.routes.upload.ValidationService') as mock_upload_validation, \
             patch('app.routes.upload.StorageService') as mock_upload_storage, \
             patch('app.routes.upload.MLService') as mock_upload_ml, \
             \
             patch('app.routes.reference.settings'), \
             patch('app.routes.reference.ValidationService') as mock_ref_validation, \
             patch('app.routes.reference.StorageService') as mock_ref_storage, \
             patch('app.routes.reference.MLService') as mock_ref_ml, \
             patch('app.routes.reference.EncryptionService') as mock_ref_encryption, \
             patch('app.routes.reference.DatabaseService') as mock_ref_db, \
             \
             patch('app.routes.verify.settings'), \
             patch('app.routes.verify.ValidationService') as mock_verify_validation, \
             patch('app.routes.verify.MLService') as mock_verify_ml, \
             patch('app.routes.verify.EncryptionService') as mock_verify_encryption, \
             patch('app.routes.verify.DatabaseService') as mock_verify_db, \
             \
             patch('app.routes.liveness.settings'), \
             patch('app.routes.liveness.ValidationService') as mock_liveness_validation, \
             patch('app.routes.liveness.MLService') as mock_liveness_ml, \
             \
             patch('app.db.database.get_async_db_manager') as mock_db_manager:
            
            # Настройка моков для загрузки
            mock_upload_validation.return_value.validate_image.return_value = Mock(
                is_valid=True,
                image_data=self.test_image_data.encode(),
                image_format="JPEG",
                quality_score=0.8
            )
            mock_upload_storage.return_value.upload_image.return_value = {
                "file_url": "http://minio/test-upload.jpg",
                "file_size": 102400,
                "image_id": "upload_123"
            }
            mock_upload_ml.return_value.generate_embedding.return_value = {
                "success": True,
                "embedding": [0.1, 0.2, 0.3, 0.4],
                "quality_score": 0.8
            }
            
            # Настройка моков для reference
            mock_ref_validation.return_value.validate_image.return_value = Mock(
                is_valid=True,
                image_data=self.test_image_data.encode(),
                image_format="JPEG",
                quality_score=0.8
            )
            mock_ref_storage.return_value.upload_image.return_value = {
                "file_url": "http://minio/test-reference.jpg",
                "file_size": 102400,
                "image_id": "reference_123"
            }
            mock_ref_ml.return_value.generate_embedding.return_value = {
                "success": True,
                "embedding": [0.1, 0.2, 0.3, 0.4],
                "quality_score": 0.85
            }
            mock_ref_encryption.return_value.encrypt_embedding.return_value = b"encrypted_embedding"
            mock_ref_db.return_value.create_reference.return_value = {
                "id": "reference_123",
                "user_id": self.test_user_id,
                "label": "Test Reference",
                "quality_score": 0.85,
                "created_at": datetime.now(timezone.utc)
            }
            
            # Настройка моков для verify
            mock_verify_validation.return_value.validate_image.return_value = Mock(
                is_valid=True,
                image_data=self.test_image_data.encode(),
                quality_score=0.8,
                image_format="JPEG"
            )
            
            reference_mock = Mock()
            reference_mock.id = "reference_123"
            reference_mock.embedding = b"encrypted_embedding"
            reference_mock.is_active = True
            mock_verify_db.return_value.get_reference_by_id.return_value = reference_mock
            mock_verify_encryption.return_value.decrypt_embedding.return_value = [0.1, 0.2, 0.3, 0.4]
            mock_verify_ml.return_value.verify_face.return_value = {
                "success": True,
                "verified": True,
                "confidence": 0.9,
                "similarity_score": 0.85,
                "face_detected": True,
                "face_quality": 0.8
            }
            
            # Настройка моков для liveness
            mock_liveness_validation.return_value.validate_image.return_value = Mock(
                is_valid=True,
                image_data=self.test_image_data.encode(),
                quality_score=0.8,
                image_format="JPEG"
            )
            mock_liveness_validation.return_value.analyze_spoof_signs.return_value = {
                "score": 0.8,
                "flags": []
            }
            mock_liveness_ml.return_value.check_liveness.return_value = {
                "success": True,
                "liveness_detected": True,
                "confidence": 0.9,
                "anti_spoofing_score": 0.85,
                "face_detected": True,
                "multiple_faces": False,
                "image_quality": 0.8,
                "recommendations": []
            }
            
            # Настройка моков БД
            mock_db_instance = Mock()
            mock_db_instance.get_session.return_value.__aenter__ = AsyncMock(return_value=mock_db_instance)
            mock_db_instance.get_session.return_value.__aexit__ = AsyncMock()
            mock_db_manager.return_value = mock_db_instance
            
            # === ЭТАП 1: Загрузка изображения ===
            upload_response = self.client.post(
                "/api/v1/upload",
                json={
                    "image_data": f"data:image/jpeg;base64,{self.test_image_data}",
                    "user_id": self.test_user_id,
                    "metadata": {"source": "e2e_test"}
                }
            )
            
            assert upload_response.status_code == 200
            upload_data = upload_response.json()
            assert upload_data["success"] is True
            assert "image_id" in upload_data
            print("✅ Upload этап пройден")
            
            # === ЭТАП 2: Создание эталона ===
            reference_response = self.client.post(
                "/api/v1/reference",
                json={
                    "user_id": self.test_user_id,
                    "image_data": f"data:image/jpeg;base64,{self.test_image_data}",
                    "label": "E2E Test Reference",
                    "quality_threshold": 0.5
                }
            )
            
            assert reference_response.status_code == 200
            reference_data = reference_response.json()
            assert reference_data["success"] is True
            assert reference_data["label"] == "E2E Test Reference"
            reference_id = reference_data["reference_id"]
            print(f"✅ Reference этап пройден (ID: {reference_id})")
            
            # === ЭТАП 3: Верификация лица ===
            verify_response = self.client.post(
                "/api/v1/verify",
                json={
                    "session_id": self.test_session_id,
                    "image_data": f"data:image/jpeg;base64,{self.test_image_data}",
                    "reference_id": reference_id,
                    "threshold": 0.8
                }
            )
            
            assert verify_response.status_code == 200
            verify_data = verify_response.json()
            assert verify_data["success"] is True
            assert verify_data["verified"] is True
            assert verify_data["confidence"] > 0.8
            print(f"✅ Verify этап пройден (verified: {verify_data['verified']})")
            
            # === ЭТАП 4: Проверка живости ===
            liveness_response = self.client.post(
                "/api/v1/liveness",
                json={
                    "session_id": self.test_session_id,
                    "image_data": f"data:image/jpeg;base64,{self.test_image_data}",
                    "challenge_type": "passive"
                }
            )
            
            assert liveness_response.status_code == 200
            liveness_data = liveness_response.json()
            assert liveness_data["success"] is True
            assert liveness_data["liveness_detected"] is True
            assert liveness_data["confidence"] > 0.8
            print(f"✅ Liveness этап пройден (live: {liveness_data['liveness_detected']})")
            
            print("🎉 Полный E2E workflow завершен успешно!")
    
    def test_workflow_with_sessions(self):
        """Тест workflow с использованием сессий"""
        
        with patch('app.routes.verify.settings'), \
             patch('app.routes.verify.ValidationService') as mock_verify_validation, \
             patch('app.routes.verify.MLService') as mock_verify_ml, \
             patch('app.routes.verify.EncryptionService') as mock_verify_encryption, \
             patch('app.routes.verify.DatabaseService') as mock_verify_db, \
             patch('app.routes.verify.CacheService') as mock_verify_cache, \
             \
             patch('app.routes.liveness.settings'), \
             patch('app.routes.liveness.ValidationService') as mock_liveness_validation, \
             patch('app.routes.liveness.MLService') as mock_liveness_ml, \
             patch('app.routes.liveness.DatabaseService') as mock_liveness_db, \
             patch('app.routes.liveness.CacheService') as mock_liveness_cache:
            
            # Настройка моков для verify сессии
            mock_verify_cache.return_value.set_verification_session.return_value = True
            
            mock_verify_validation.return_value.validate_image.return_value = Mock(
                is_valid=True,
                image_data=self.test_image_data.encode(),
                quality_score=0.8,
                image_format="JPEG"
            )
            
            reference_mock = Mock()
            reference_mock.id = "reference_123"
            reference_mock.embedding = b"encrypted_embedding"
            reference_mock.is_active = True
            mock_verify_db.return_value.get_reference_by_id.return_value = reference_mock
            mock_verify_encryption.return_value.decrypt_embedding.return_value = [0.1, 0.2, 0.3, 0.4]
            mock_verify_ml.return_value.verify_face.return_value = {
                "success": True,
                "verified": True,
                "confidence": 0.9,
                "similarity_score": 0.85,
                "face_detected": True,
                "face_quality": 0.8
            }
            
            # Настройка моков для liveness сессии
            mock_liveness_cache.return_value.set_verification_session.return_value = True
            mock_liveness_cache.return_value.get_verification_session.return_value = {
                "session_id": self.test_session_id,
                "session_type": "liveness",
                "status": "pending"
            }
            
            mock_liveness_validation.return_value.validate_image.return_value = Mock(
                is_valid=True,
                image_data=self.test_image_data.encode(),
                quality_score=0.8,
                image_format="JPEG"
            )
            mock_liveness_validation.return_value.analyze_spoof_signs.return_value = {
                "score": 0.8,
                "flags": []
            }
            mock_liveness_ml.return_value.check_liveness.return_value = {
                "success": True,
                "liveness_detected": True,
                "confidence": 0.9,
                "anti_spoofing_score": 0.85,
                "face_detected": True,
                "multiple_faces": False,
                "image_quality": 0.8,
                "recommendations": []
            }
            
            # === ЭТАП 1: Создание verify сессии ===
            verify_session_response = self.client.post(
                "/api/v1/verify/session",
                json={
                    "session_type": "verification",
                    "user_id": self.test_user_id,
                    "expires_in_minutes": 30
                }
            )
            
            assert verify_session_response.status_code == 200
            verify_session_data = verify_session_response.json()
            assert verify_session_data["success"] is True
            verify_session_id = verify_session_data["session_id"]
            print(f"✅ Verify session created: {verify_session_id}")
            
            # === ЭТАП 2: Верификация в сессии ===
            verify_in_session_response = self.client.post(
                f"/api/v1/verify/session/{verify_session_id}",
                json={
                    "image_data": f"data:image/jpeg;base64,{self.test_image_data}",
                    "reference_id": "reference_123",
                    "threshold": 0.8
                }
            )
            
            assert verify_in_session_response.status_code == 200
            verify_in_session_data = verify_in_session_response.json()
            assert verify_in_session_data["success"] is True
            assert verify_in_session_data["verified"] is True
            print(f"✅ Verify in session completed: {verify_in_session_data['verified']}")
            
            # === ЭТАП 3: Создание liveness сессии ===
            liveness_session_response = self.client.post(
                "/api/v1/liveness/session",
                json={
                    "session_type": "liveness",
                    "user_id": self.test_user_id,
                    "expires_in_minutes": 30
                }
            )
            
            assert liveness_session_response.status_code == 200
            liveness_session_data = liveness_session_response.json()
            assert liveness_session_data["success"] is True
            liveness_session_id = liveness_session_data["session_id"]
            print(f"✅ Liveness session created: {liveness_session_id}")
            
            # === ЭТАП 4: Проверка живости в сессии ===
            liveness_in_session_response = self.client.post(
                f"/api/v1/liveness/session/{liveness_session_id}",
                json={
                    "image_data": f"data:image/jpeg;base64,{self.test_image_data}",
                    "challenge_type": "passive"
                }
            )
            
            assert liveness_in_session_response.status_code == 200
            liveness_in_session_data = liveness_in_session_response.json()
            assert liveness_in_session_data["success"] is True
            assert liveness_in_session_data["liveness_detected"] is True
            print(f"✅ Liveness in session completed: {liveness_in_session_data['liveness_detected']}")
            
            print("🎉 E2E workflow с сессиями завершен успешно!")
    
    def test_error_handling_workflow(self):
        """Тест обработки ошибок в workflow"""
        
        with patch('app.routes.reference.settings'), \
             patch('app.routes.reference.ValidationService') as mock_validation:
            
            # Мок для невалидного изображения
            mock_validation.return_value.validate_image.return_value = Mock(
                is_valid=False,
                error_message="Invalid image format"
            )
            
            # === ЭТАП 1: Попытка создать reference с невалидным изображением ===
            reference_response = self.client.post(
                "/api/v1/reference",
                json={
                    "user_id": self.test_user_id,
                    "image_data": "invalid_image_data",
                    "label": "Invalid Reference"
                }
            )
            
            # Ожидаем ошибку валидации
            assert reference_response.status_code == 400
            reference_error_data = reference_response.json()
            assert reference_error_data["detail"]["error_code"] == "VALIDATION_ERROR"
            print("✅ Ошибка валидации обработана корректно")
            
            # === ЭТАП 2: Попытка верификации без reference_id и user_id ===
            verify_response = self.client.post(
                "/api/v1/verify",
                json={
                    "session_id": self.test_session_id,
                    "image_data": f"data:image/jpeg;base64,{self.test_image_data}"
                    # Отсутствует reference_id и user_id
                }
            )
            
            # Ожидаем ошибку валидации
            assert verify_response.status_code == 400
            verify_error_data = verify_response.json()
            assert verify_error_data["detail"]["error_code"] == "VALIDATION_ERROR"
            print("✅ Ошибка отсутствующих параметров обработана корректно")
            
            # === ЭТАП 3: Попытка liveness с активным челленджем без данных ===
            liveness_response = self.client.post(
                "/api/v1/liveness",
                json={
                    "session_id": self.test_session_id,
                    "image_data": f"data:image/jpeg;base64,{self.test_image_data}",
                    "challenge_type": "active"
                    # Отсутствует challenge_data
                }
            )
            
            # Ожидаем ошибку валидации
            assert liveness_response.status_code == 400
            liveness_error_data = liveness_response.json()
            assert liveness_error_data["detail"]["error_code"] == "VALIDATION_ERROR"
            print("✅ Ошибка активного челленджа обработана корректно")
            
            print("🎉 Тест обработки ошибок завершен успешно!")
    
    def test_performance_workflow(self):
        """Тест производительности workflow"""
        import time
        
        with patch('app.routes.reference.settings'), \
             patch('app.routes.verify.settings'), \
             patch('app.routes.liveness.settings'):
            
            # Измеряем время выполнения для каждого этапа
            start_time = time.time()
            
            # === ЭТАП 1: Reference ===
            ref_start = time.time()
            # Симулируем быстрый ответ для reference
            with patch('app.routes.reference.ValidationService') as mock_ref_validation, \
                 patch('app.routes.reference.StorageService') as mock_ref_storage, \
                 patch('app.routes.reference.MLService') as mock_ref_ml, \
                 patch('app.routes.reference.EncryptionService') as mock_ref_encryption, \
                 patch('app.routes.reference.DatabaseService') as mock_ref_db:
                
                mock_ref_validation.return_value.validate_image.return_value = Mock(
                    is_valid=True,
                    image_data=self.test_image_data.encode(),
                    image_format="JPEG",
                    quality_score=0.8
                )
                mock_ref_storage.return_value.upload_image.return_value = {"file_url": "http://test.jpg"}
                mock_ref_ml.return_value.generate_embedding.return_value = {
                    "success": True,
                    "embedding": [0.1, 0.2, 0.3, 0.4],
                    "quality_score": 0.8
                }
                mock_ref_encryption.return_value.encrypt_embedding.return_value = b"encrypted"
                mock_ref_db.return_value.create_reference.return_value = {
                    "id": "ref_123",
                    "user_id": self.test_user_id,
                    "label": "Performance Test",
                    "quality_score": 0.8,
                    "created_at": datetime.now(timezone.utc)
                }
                
                ref_response = self.client.post(
                    "/api/v1/reference",
                    json={
                        "user_id": self.test_user_id,
                        "image_data": f"data:image/jpeg;base64,{self.test_image_data}",
                        "label": "Performance Test"
                    }
                )
                
                ref_time = time.time() - ref_start
                assert ref_response.status_code == 200
                print(f"⏱️ Reference этап: {ref_time:.3f}s")
            
            # === ЭТАП 2: Verify ===
            verify_start = time.time()
            with patch('app.routes.verify.ValidationService') as mock_verify_validation, \
                 patch('app.routes.verify.MLService') as mock_verify_ml, \
                 patch('app.routes.verify.EncryptionService') as mock_verify_encryption, \
                 patch('app.routes.verify.DatabaseService') as mock_verify_db:
                
                mock_verify_validation.return_value.validate_image.return_value = Mock(
                    is_valid=True,
                    image_data=self.test_image_data.encode(),
                    quality_score=0.8,
                    image_format="JPEG"
                )
                
                reference_mock = Mock()
                reference_mock.id = "ref_123"
                reference_mock.embedding = b"encrypted"
                reference_mock.is_active = True
                mock_verify_db.return_value.get_reference_by_id.return_value = reference_mock
                mock_verify_encryption.return_value.decrypt_embedding.return_value = [0.1, 0.2, 0.3, 0.4]
                mock_verify_ml.return_value.verify_face.return_value = {
                    "success": True,
                    "verified": True,
                    "confidence": 0.9,
                    "similarity_score": 0.85,
                    "face_detected": True,
                    "face_quality": 0.8
                }
                
                verify_response = self.client.post(
                    "/api/v1/verify",
                    json={
                        "session_id": self.test_session_id,
                        "image_data": f"data:image/jpeg;base64,{self.test_image_data}",
                        "reference_id": "ref_123"
                    }
                )
                
                verify_time = time.time() - verify_start
                assert verify_response.status_code == 200
                print(f"⏱️ Verify этап: {verify_time:.3f}s")
            
            # === ЭТАП 3: Liveness ===
            liveness_start = time.time()
            with patch('app.routes.liveness.ValidationService') as mock_liveness_validation, \
                 patch('app.routes.liveness.MLService') as mock_liveness_ml:
                
                mock_liveness_validation.return_value.validate_image.return_value = Mock(
                    is_valid=True,
                    image_data=self.test_image_data.encode(),
                    quality_score=0.8,
                    image_format="JPEG"
                )
                mock_liveness_validation.return_value.analyze_spoof_signs.return_value = {
                    "score": 0.8,
                    "flags": []
                }
                mock_liveness_ml.return_value.check_liveness.return_value = {
                    "success": True,
                    "liveness_detected": True,
                    "confidence": 0.9,
                    "anti_spoofing_score": 0.85,
                    "face_detected": True,
                    "multiple_faces": False,
                    "image_quality": 0.8,
                    "recommendations": []
                }
                
                liveness_response = self.client.post(
                    "/api/v1/liveness",
                    json={
                        "session_id": self.test_session_id,
                        "image_data": f"data:image/jpeg;base64,{self.test_image_data}",
                        "challenge_type": "passive"
                    }
                )
                
                liveness_time = time.time() - liveness_start
                assert liveness_response.status_code == 200
                print(f"⏱️ Liveness этап: {liveness_time:.3f}s")
            
            total_time = time.time() - start_time
            print(f"⏱️ Общее время workflow: {total_time:.3f}s")
            
            # Проверяем что каждый этап выполняется быстро (< 1 секунды с моками)
            assert ref_time < 1.0, f"Reference этап слишком медленный: {ref_time:.3f}s"
            assert verify_time < 1.0, f"Verify этап слишком медленный: {verify_time:.3f}s"
            assert liveness_time < 1.0, f"Liveness этап слишком медленный: {liveness_time:.3f}s"
            
            print("✅ Тест производительности пройден!")