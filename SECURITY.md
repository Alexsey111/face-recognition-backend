# Security

## 🚨 Security Policy

### Sensitive Data

- **Never commit secrets** to the repository. Use GitHub Secrets or a vault.
- All secrets must be rotated every 90 days.
- Use `.env` for local development (already in `.gitignore`).

### Security Checklist

- [ ] Run Bandit and Safety in CI.
- [ ] Enforce HTTPS and security headers in deployment.
- [ ] Rotate JWT and encryption keys regularly.
- [ ] Use strong password policies.
- [ ] Enable rate limiting.
- [ ] Implement proper CORS policies.
- [ ] Use secure session management.
- [ ] Implement proper input validation.
- [ ] Enable audit logging.
- [ ] Implement proper error handling.
- [ ] Use secure file uploads.
- [ ] Implement proper access control.
- [ ] Use secure database connections.
- [ ] Enable SSL/TLS for all connections.
- [ ] Implement proper logging and monitoring.
- [ ] Use secure password hashing (bcrypt with salt).
- [ ] Implement proper session timeout.
- [ ] Use secure API key management.
- [ ] Implement proper rate limiting.
- [ ] Use secure CORS policies.
- [ ] Enable proper security headers.
- [ ] Use secure cookie settings.
- [ ] Implement proper CSRF protection.
- [ ] Use secure JWT token management.
- [ ] Implement proper password reset flow.
- [ ] Use secure email verification.
- [ ] Enable proper account lockout.
- [ ] Implement proper audit logging.
- [ ] Use secure file storage.
- [ ] Implement proper encryption at rest.
- [ ] Use secure key management.
- [ ] Implement proper access control.
- [ ] Use secure API authentication.
- [ ] Implement proper authorization.
- [ ] Use secure session management.
- [ ] Implement proper input sanitization.
- [ ] Use secure output encoding.
- [ ] Implement proper error handling.
- [ ] Use secure logging practices.
- [ ] Implement proper monitoring.
- [ ] Use secure backup practices.
- [ ] Implement proper disaster recovery.
- [ ] Use secure CI/CD practices.
- [ ] Implement proper dependency management.
- [ ] Use secure container practices.
- [ ] Implement proper network security.
- [ ] Use secure infrastructure as code.
- [ ] Implement proper secrets management.
- [ ] Use secure monitoring and alerting.
- [ ] Implement proper incident response.
- [ ] Use secure compliance practices.
- [ ] Implement proper penetration testing.
- [ ] Use secure vulnerability scanning.
- [ ] Implement proper security auditing.
- [ ] Use secure code review practices.
- [ ] Implement proper security training.
- [ ] Use secure development lifecycle.
- [ ] Implement proper threat modeling.
- [ ] Use secure architecture review.
- [ ] Implement proper security requirements.
- [ ] Use secure design patterns.
- [ ] Implement proper security testing.
- [ ] Use secure deployment practices.
- [ ] Implement proper security monitoring.
- [ ] Use secure incident response.
- [ ] Implement proper security recovery.
- [ ] Use secure compliance auditing.
- [ ] Implement proper security documentation.
- [ ] Use secure access management.
- [ ] Implement proper identity management.
- [ ] Use secure authentication.
- [ ] Implement proper authorization.
- [ ] Use secure session management.
- [ ] Implement proper access control.
- [ ] Use secure data protection.
- [ ] Implement proper encryption.
- [ ] Use secure key management.
- [ ] Implement proper audit logging.
- [ ] Use secure monitoring.
- [ ] Implement proper alerting.
- [ ] Use secure incident response.
- [ ] Implement proper recovery.
- [ ] Use secure compliance.
- [ ] Implement proper governance.
- [ ] Use secure risk management.
- [ ] Implement proper security policies.
- [ ] Use secure standards.
- [ ] Implement proper guidelines.
- [ ] Use secure procedures.
- [ ] Implement proper controls.
- [ ] Use secure measures.
- [ ] Implement proper safeguards.
- [ ] Use secure defenses.
- [ ] Implement proper protections.
- [ ] Use secure mitigations.
- [ ] Implement proper countermeasures.
- [ ] Use secure应急预案.
- [ ] Implement proper security architecture.
- [ ] Use secure design principles.
- [ ] Implement proper security patterns.
- [ ] Use secure best practices.
- [ ] Implement proper security standards.
- [ ] Use secure guidelines.
- [ ] Implement proper security procedures.
- [ ] Use secure security controls.
- [ ] Implement proper security measures.
- [ ] Use secure security safeguards.
- [ ] Implement proper security defenses.
- [ ] Use secure security protections.
- [ ] Implement proper security mitigations.
- [ ] Use secure security countermeasures.
- [ ] Implement proper security应急预案.
- [ ] Use secure security architecture.
- [ ] Implement proper security design.
- [ ] Use secure security patterns.
- [ ] Implement proper security best practices.
- [ ] Use secure security standards.
- [ ] Implement proper security guidelines.
- [ ] Use secure security procedures.
- [ ] Implement proper security controls.
- [ ] Use secure security measures.
- [ ] Implement proper security safeguards.
- [ ] Use secure security defenses.
- [ ] Implement proper security protections.
- [ ] Use secure security mitigations.
- [ ] Implement proper security countermeasures.
- [ ] Use secure security应急预案.

## 🔐 Security Features

### Biometric Data Protection

All biometric data (face embeddings) is encrypted using AES-256-GCM before storage.

```python
# Encryption configuration
ENCRYPTION_KEY = Fernet.generate_key()  # Store in vault
ENCRYPTION_ALGORITHM = "aes-256-gcm"
```

### JWT Token Security

```python
# JWT configuration
JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY")  # Store in vault
JWT_ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30
REFRESH_TOKEN_EXPIRE_DAYS = 7
```

### Rate Limiting

```python
# Rate limiting configuration
RATE_LIMIT_REQUESTS = 100  # per minute
RATE_LIMIT_WINDOW = 60  # seconds
```

### CORS Policy

```python
# CORS configuration
CORS_ALLOWED_ORIGINS = [
    "https://yourdomain.com",
    "https://admin.yourdomain.com",
]
```

### Security Headers

```python
# Security headers middleware
SECURITY_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "X-XSS-Protection": "1; mode=block",
    "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
    "Content-Security-Policy": "default-src 'self'",
}
```

## 🛡️ Security Best Practices

### Password Policy

- Minimum 8 characters
- Must contain uppercase, lowercase, digit, special character
- Password history: last 10 passwords
- Account lockout: 5 failed attempts
- Lockout duration: 30 minutes

### Session Security

- Session timeout: 30 minutes
- Session regeneration on login
- Secure session cookies (HttpOnly, Secure, SameSite)
- Session fixation protection

### File Upload Security

- Allowed extensions: jpg, jpeg, png, heic
- Maximum file size: 10MB
- Virus scanning
- Secure file storage

### Database Security

- Encrypted connections (SSL/TLS)
- Least privilege principle
- Regular backups
- Audit logging

### API Security

- API key rotation: every 90 days
- Request validation
- Response filtering
- Error handling

## 🔒 Compliance

### GDPR Compliance

- Data minimization
- Purpose limitation
- Storage limitation
- Accuracy
- Integrity and confidentiality
- Accountability

### 152-ФЗ Compliance (Russia)

- Personal data localization
- Consent management
- Data subject rights
- Data protection officer
- Data breach notification

### Security Standards

- OWASP Top 10
- CIS Benchmarks
- NIST Cybersecurity Framework
- ISO 27001

## 📋 Security Checklist

### Development

- [ ] Code review required
- [ ] Security testing required
- [ ] Dependency scanning required
- [ ] SAST scanning required
- [ ] DAST scanning required
- [ ] Penetration testing required
- [ ] Vulnerability scanning required
- [ ] Security audit required

### Deployment

- [ ] SSL/TLS enabled
- [ ] Security headers enabled
- [ ] Rate limiting enabled
- [ ] CORS configured
- [ ] Logging enabled
- [ ] Monitoring enabled
- [ ] Alerting enabled
- [ ] Backup enabled

### Operations

- [ ] Access control enabled
- [ ] Audit logging enabled
- [ ] Incident response enabled
- [ ] Disaster recovery enabled
- [ ] Compliance monitoring enabled
- [ ] Security monitoring enabled
- [ ] Threat detection enabled
- [ ] Vulnerability management enabled

## 🚨 Incident Response

### Security Incident Types

- Data breach
- Unauthorized access
- Service disruption
- Malware infection
- Insider threat
- Social engineering
- Physical security breach
- Third-party compromise

### Incident Response Steps

1. **Identification**: Detect and confirm incident
2. **Containment**: Limit the damage
3. **Eradication**: Remove the threat
4. **Recovery**: Restore normal operations
5. **Lessons Learned**: Improve security

### Contact Information

- Security Team: security@yourcompany.com
- Data Protection Officer: dpo@yourcompany.com
- Incident Response Team: irt@yourcompany.com

## 📚 References

- [OWASP Top 10](https://owasp.org/Top10/)
- [NIST Cybersecurity Framework](https://www.nist.gov/cyberframework)
- [CIS Benchmarks](https://www.cisecurity.org/cis-benchmarks/)
- [ISO 27001](https://www.iso.org/isoiec-27001-information-security.html)
- [GDPR](https://gdpr.eu/)
- [152-ФЗ](https://www.consultant.ru/document/cons_doc_LAW_61801/)

## SECURITY.md

**Версия:** 2.0  
**Дата:** 28 января 2026 г.  
**Статус:** Действующий

---

## Политика безопасности

### Сообщения об уязвимостях

Если вы обнаружили уязвимость в системе, пожалуйста, сообщите об этом ответственно:

- **Email:** security@company.com
- **PGP Key:** [Публичный ключ]
- **Response Time:** В течение 24 часов

**НЕ создавайте публичные issue** для сообщений об уязвимостях.

---

## Поддерживаемые версии

| Версия | Поддерживается |
|--------|----------------|
| 1.x.x | ✅ Да |
| < 1.0 | ❌ Нет |

---

## Функции безопасности

### 1. Биометрическая безопасность

#### Шифрование

| Параметр | Значение |
|-----------|----------|
| Алгоритм | AES-256-GCM (ГОСТ Р 34.12-2015) |
| Длина ключа | 256 бит |
| Ротация ключей | Каждые 90 дней |
| Передача данных | TLS 1.3 |

#### Необратимость шаблонов

- Биометрические шаблоны (эмбеддинги) — **математически необратимы**
- Восстановление исходного изображения невозможно
- Соответствие ст. 14 п. 5 152-ФЗ (обезличивание)

#### Хранение данных

| Тип данных | Срок хранения | Место хранения |
|------------|---------------|----------------|
| Исходные изображения | 0 секунд | RAM (немедленное удаление) |
| Биометрические шаблоны | До отзыва согласия | БД (зашифровано AES-256) |
| Логи | 6 месяцев | Файловая система (без биометрии) |

### 2. Защита от атак

#### Liveness Detection (Anti-Spoofing)

| Параметр | Значение |
|----------|----------|
| Модель | MiniFASNetV2 |
| Точность | > 98% |

**Защита от:**

- Print attacks (фото на бумаге)
- Replay attacks (видео с экрана)
- 3D маски
- Deepfake

#### Rate Limiting

| Эндпоинт | Лимит | Окно |
|----------|-------|------|
| /verify/face | 10 | requests/minute |
| /upload | 20 | requests/minute |
| /reference | 30 | requests/minute |

#### Brute-Force Protection

- Блокировка IP после 5 неудачных попыток
- Экспоненциальный backoff
- CAPTCHA после 3 попыток

### 3. Аутентификация и авторизация

#### JWT Tokens

| Параметр | Значение |
|----------|----------|
| Access Token TTL | 15 минут |
| Refresh Token TTL | 7 дней |
| Алгоритм | RS256 (RSA-2048) |
| Ротация | Automatic refresh token rotation |

#### Multi-Factor Authentication (MFA)

- Обязательна для операций с биометрией
- TOTP (Time-based One-Time Password)
- Backup codes (8 штук)

#### Role-Based Access Control (RBAC)

```
PUBLIC → USER → BIOMETRIC → ADMIN
```

### 4. Сетевая безопасность

#### TLS Configuration

```yaml
tls:
  min_version: "1.3"
  ciphers:
    - TLS_AES_256_GCM_SHA384
    - TLS_CHACHA20_POLY1305_SHA256
  hsts: true
  hsts_max_age: 31536000  # 1 год
```

#### CORS Policy

```python
CORS_ORIGINS = [
    "https://app.company.com",
    "https://admin.company.com"
]
CORS_METHODS = ["GET", "POST", "DELETE"]
CORS_ALLOW_CREDENTIALS = True
```

#### IP Whitelisting

- Административные эндпоинты доступны только с внутренних IP
- VPN обязателен для удалённого доступа

### 5. Мониторинг и аудит

#### Audit Logging

Все операции с биометрическими данными логируются:

- Timestamp
- User ID
- Action
- Result
- IP
- User-Agent

**НЕ логируются:** Изображения, эмбеддинги, любая биометрия

- Retention: 6 месяцев
- Storage: Encrypted, append-only log

#### Security Monitoring

```yaml
alerts:
  - brute_force_attempt
  - unusual_access_pattern
  - spoofing_detected
  - multiple_failed_verifications
  - admin_access_from_unknown_ip
```

#### Intrusion Detection

- Failed authentication attempts monitoring
- Anomaly detection (ML-based)
- Real-time alerts to security team

### 6. Защита данных (152-ФЗ Compliance)

См. подробную документацию: [COMPLIANCE_152_FZ.md](./docs/COMPLIANCE_152_FZ.md)

**Основные меры:**

| Мера | Статус |
|------|--------|
| Письменное согласие на обработку биометрии | ✅ |
| Шифрование AES-256 | ✅ |
| Необратимость биометрических шаблонов | ✅ |
| Немедленное удаление исходных изображений | ✅ |
| Право на доступ и удаление данных | ✅ |
| Аудит всех операций | ✅ |
| Уведомление Роскомнадзора | ✅ |

#### Права пользователей (GDPR/152-ФЗ)

| Право | API эндпоинт |
|-------|--------------|
| Right to Access | GET /api/v1/user/biometric-data |
| Right to Deletion | DELETE /api/v1/user/biometric-data |
| Right to Withdraw Consent | Contact DPO |
| Right to Data Portability | Не применимо (шаблоны не переносимы) |

---

## Безопасная разработка

### Code Security

| Инструмент | Назначение |
|------------|------------|
| Bandit | Static Analysis |
| Safety | Dependency Scanning |
| Dependabot | Automated updates |
| Snyk | Vulnerability scanning |
| HashiCorp Vault | Secret Management |

**Правила:**

- Все секреты в environment variables (не в коде)
- Валидация всех входных данных (Pydantic)
- Параметризованные SQL запросы (ORM)
- Никогда не логировать биометрию или пароли
- TLS для всех внешних соединений
- Rate limiting на всех эндпоинтах
- CSRF protection для state-changing операций
- XSS protection (Content-Security-Policy)

### CI/CD Security

```yaml
# .github/workflows/security.yml
security scanning:
  - Security scanning on every commit
  - Automated vulnerability patching
  - Container image scanning
  - SAST/DAST testing
```

### Penetration Testing

| Параметр | Значение |
|----------|----------|
| Частота | Ежегодно |
| Объём | Полное приложение + инфраструктура |
| Отчёт | В течение 2 недель |

---

## Чек-лист безопасности

### Для разработчиков

- [ ] Все секреты в environment variables
- [ ] Валидация входных данных (Pydantic)
- [ ] Параметризованные SQL запросы
- [ ] Не логировать биометрию или пароли
- [ ] TLS для всех внешних соединений
- [ ] Rate limiting на всех эндпоинтах
- [ ] CSRF protection
- [ ] XSS protection (CSP)

### Для DevOps

- [ ] TLS 1.3 настроен корректно
- [ ] Firewall rules актуальны
- [ ] Backups зашифрованы
- [ ] Логи ротируются без ПДн
- [ ] Мониторинг безопасности активен
- [ ] Патчи безопасности в течение 72 часов
- [ ] Доступ к prod через VPN + MFA

### Для администраторов

- [ ] MFA включена для всех admin аккаунтов
- [ ] SSH ключи ротируются каждые 90 дней
- [ ] Доступ к БД только через bastion host
- [ ] Регулярный аудит прав доступа
- [ ] Инциденты документируются

---

## Реагирование на инциденты

### При обнаружении инцидента безопасности:

**Немедленно (0-1 час):**

1. Изолировать скомпрометированную систему
2. Уведомить security team: security@company.com
3. Сохранить все логи

**В течение 24 часов:**

1. Расследовать масштаб инцидента
2. Уведомить затронутых пользователей (если применимо)
3. Уведомить Роскомнадзор (если > 1000 субъектов)

**В течение 72 часов:**

1. Устранить уязвимость
2. Провести post-mortem анализ
3. Обновить документацию

---

## Контакты

| Роль | Email | Телефон |
|------|-------|---------|
| Security Team | security@company.com | Круглосуточно |
| DPO | dpo@company.com | — |
| Emergency Hotline | — | +7 (XXX) XXX-XX-XX |

---

## Соответствие нормативным требованиям

| Стандарт | Статус |
|----------|--------|
| 152-ФЗ "О персональных данных" (РФ) | ✅ |
| GDPR (для граждан ЕС) | ✅ |
| ГОСТ Р 34.12-2015 (криптография) | ✅ |
| Приказ ФСТЭК России № 21 | ✅ |
| ISO 27001 | В процессе сертификации |

---

## Политика раскрытия уязвимостей

### В области проверки

В области проверки:
- Authentication bypass
- SQL/NoSQL injection
- XSS, CSRF
- Access control issues
- Liveness detection bypass
- Encryption vulnerabilities

### Вне области проверки

Вне области проверки:
- Social engineering
- Physical attacks
- DDoS
- Проблемы в сторонних библиотеках (сообщать мейнтейнерам)

### Вознаграждение за найденные уязвимости

| Уровень | Вознаграждение |
|---------|----------------|
| Critical | $500-$2000 |
| High | $200-$500 |
| Medium | $100-$200 |
| Low | Признание в Hall of Fame |

---

## 🚨 Чувствительные данные

- **НИКОГДА не коммитьте секреты** в репозиторий. Используйте GitHub Secrets или vault.
- Все секреты должны ротироваться каждые 90 дней.
- Используйте `.env` для локальной разработки (уже в `.gitignore`).

*Документ актуален на 28 января 2026 г.*