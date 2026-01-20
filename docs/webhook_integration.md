# Webhook Integration Guide

## 📋 Overview

Webhook система Face Recognition Service позволяет получать real-time уведомления о событиях верификации и проверки живости. Система гарантирует доставку через механизм retry с exponential backoff и обеспечивает безопасность через HMAC-SHA256 подпись.

---

## 🚀 Quick Start

### 1. Configuration

Добавьте webhook переменные в `.env`:

```bash
# Webhook Configuration
WEBHOOK_URL=https://your-crm.com/api/webhook
WEBHOOK_SECRET=your-secret-key-min-32-chars
WEBHOOK_TIMEOUT=10
WEBHOOK_MAX_RETRIES=3
WEBHOOK_RETRY_DELAY=1
