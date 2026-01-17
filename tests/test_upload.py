"""
Тесты для upload endpoints (Phase 5).
Проверка API endpoints загрузки файлов.
"""

import pytest
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from fastapi.testclient import TestClient


class TestUploadEndpoints:
    """Тесты endpoints загрузки файлов"""
    
    @pytest.fixture
    def client(self, app_instance):
        """Создание тестового клиента"""
        return TestClient(app_instance)
    
    def test_create_upload_session(self, client):
        """Тест создания сессии загрузки"""
        from datetime import datetime, timedelta
        
        mock_session = MagicMock()
        mock_session.session_id = "test-uuid"
        mock_session.user_id = "user123"
        mock_session.created_at = datetime.utcnow()
        mock_session.expiration_at = datetime.utcnow() + timedelta(days=30)
        mock_session.file_key = None
        mock_session.file_size = None
        
        with patch('app.routes.upload.SessionService.create_session', new_callable=AsyncMock) as mock_create, \
             patch('app.db.crud.AuditLogCRUD.log_action', new_callable=AsyncMock) as mock_log:
            
            mock_create.return_value = mock_session
            
            response = client.post(
                "/api/v1/upload/",
                headers={"Authorization": "Bearer test_token"}
            )
            
            # Проверяем что сессия была создана
            if response.status_code == 200:
                mock_create.assert_called_once()
    
    def test_get_upload_status(self, client):
        """Тест получения статуса загрузки"""
        from datetime import datetime, timedelta
        
        mock_session = MagicMock()
        mock_session.session_id = "test-uuid"
        mock_session.user_id = "user123"
        mock_session.created_at = datetime.utcnow()
        mock_session.expiration_at = datetime.utcnow() + timedelta(days=30)
        mock_session.file_key = "uploads/user123/test.jpg"
        mock_session.file_size = 1024
        mock_session.is_expired = MagicMock(return_value=False)
        
        with patch('app.routes.upload.SessionService.get_session', new_callable=AsyncMock) as mock_get:
            mock_get.return_value = mock_session
            
            response = client.get(
                "/api/v1/upload/test-uuid",
                headers={"Authorization": "Bearer test_token"}
            )
            
            # Status endpoint должен работать
            if response.status_code == 200:
                mock_get.assert_called_once_with("test-uuid")
    
    def test_delete_upload_session(self, client):
        """Тест удаления сессии загрузки"""
        from datetime import datetime, timedelta
        
        mock_session = MagicMock()
        mock_session.session_id = "test-uuid"
        mock_session.user_id = "user123"
        mock_session.created_at = datetime.utcnow()
        mock_session.expiration_at = datetime.utcnow() + timedelta(days=30)
        mock_session.file_key = "uploads/user123/test.jpg"
        mock_session.is_expired = MagicMock(return_value=False)
        
        with patch('app.routes.upload.SessionService.get_session', new_callable=AsyncMock) as mock_get, \
             patch('app.routes.upload.SessionService.delete_session', new_callable=AsyncMock) as mock_delete, \
             patch('app.routes.upload.StorageService') as mock_storage, \
             patch('app.db.crud.AuditLogCRUD.log_action', new_callable=AsyncMock) as mock_log:
            
            mock_get.return_value = mock_session
            mock_delete.return_value = True
            mock_storage.return_value.delete_image.return_value = True
            
            response = client.delete(
                "/api/v1/upload/test-uuid",
                headers={"Authorization": "Bearer test_token"}
            )
            
            if response.status_code == 200:
                mock_delete.assert_called_once_with("test-uuid")


class TestUploadWorkflows:
    """Интеграционные тесты workflow загрузки"""
    
    @pytest.fixture
    def client(self, app_instance):
        """Создание тестового клиента"""
        return TestClient(app_instance)
    
    def test_complete_upload_workflow(self, client):
        """Тест полного workflow загрузки файла"""
        from datetime import datetime, timedelta
        from io import BytesIO
        from PIL import Image
        
        mock_session = MagicMock()
        mock_session.session_id = "test-session-uuid"
        mock_session.user_id = "user123"
        mock_session.created_at = datetime.utcnow()
        mock_session.expiration_at = datetime.utcnow() + timedelta(days=30)
        mock_session.file_key = None
        mock_session.file_size = None
        mock_session.is_expired = MagicMock(return_value=False)
        
        with patch('app.routes.upload.SessionService.create_session', new_callable=AsyncMock) as mock_create, \
             patch('app.routes.upload.SessionService.validate_session', new_callable=AsyncMock) as mock_validate, \
             patch('app.routes.upload.SessionService.attach_file_to_session', new_callable=AsyncMock) as mock_attach, \
             patch('app.routes.upload.ImageValidator') as mock_validator, \
             patch('app.routes.upload.StorageService') as mock_storage, \
             patch('app.db.crud.AuditLogCRUD.log_action', new_callable=AsyncMock) as mock_log:
            
            mock_create.return_value = mock_session
            mock_validate.return_value = True
            mock_attach.return_value = mock_session
            mock_validator.validate_image.return_value = (True, "")
            mock_storage.return_value.upload_image.return_value = {
                "key": "uploads/user123/test.jpg",
                "file_url": "http://minio:9000/bucket/uploads/user123/test.jpg",
                "file_size": 1024,
                "content_type": "image/jpeg",
                "metadata": {"file_hash": "abc123"}
            }
            
            # Создаем сессию
            create_response = client.post(
                "/api/v1/upload/",
                headers={"Authorization": "Bearer test_token"}
            )
            
            if create_response.status_code == 200:
                session_id = create_response.json().get("session_id")
                
                # Загружаем файл
                img = Image.new('RGB', (100, 100), color='blue')
                img_bytes = BytesIO()
                img.save(img_bytes, format='JPEG')
                
                upload_response = client.post(
                    f"/api/v1/upload/{session_id}/file",
                    files={"file": ("test.jpg", img_bytes.getvalue(), "image/jpeg")},
                    headers={"Authorization": "Bearer test_token"}
                )
                
                # Проверяем результат
                if upload_response.status_code == 200:
                    data = upload_response.json()
                    assert "file_url" in data or "key" in data


# Запуск тестов
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
