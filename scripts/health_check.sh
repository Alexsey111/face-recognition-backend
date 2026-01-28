#!/bin/bash
# =============================================================================
# Health Check Script for Deployment Validation
# =============================================================================
# Проверяет доступность всех ключевых endpoints сервиса
#
# Usage: ./scripts/health_check.sh <base_url> [port]
# Example: ./scripts/health_check.sh http://localhost 8000
# =============================================================================

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
BASE_URL=$1
PORT=${2:-8000}
MAX_RETRIES=10
RETRY_DELAY=5
TIMEOUT=10

# Check dependencies
check_dependencies() {
    echo "🔍 Checking dependencies..."
    local missing=()
    
    if ! command -v curl &> /dev/null; then
        missing+=("curl")
    fi
    
    if ! command -v bc &> /dev/null; then
        missing+=("bc")
    fi
    
    if [ ${#missing[@]} -ne 0 ]; then
        echo -e "${RED}❌ Missing dependencies: ${missing[*]}${NC}"
        exit 1
    fi
    
    echo -e "${GREEN}✅ All dependencies available${NC}"
}

# Parse URL and set endpoint
build_url() {
    local endpoint=$1
    # Remove trailing slash from BASE_URL if present
    local clean_url="${BASE_URL%/}"
    echo "${clean_url}:${PORT}${endpoint}"
}

# Health endpoint check
check_health() {
    echo ""
    echo "1️⃣  Checking /health endpoint..."
    local url=$(build_url "/health")
    local passed=false
    
    for i in $(seq 1 $MAX_RETRIES); do
        if curl -f -s --connect-timeout $TIMEOUT "$url" > /dev/null 2>&1; then
            echo -e "   ${GREEN}✅ Health check passed${NC}"
            passed=true
            break
        else
            echo -e "   ${YELLOW}⏳ Retry $i/$MAX_RETRIES...${NC}"
            sleep $RETRY_DELAY
        fi
    done
    
    if [ "$passed" = false ]; then
        echo -e "   ${RED}❌ Health check failed${NC}"
        return 1
    fi
}

# Metrics endpoint check
check_metrics() {
    echo ""
    echo "2️⃣  Checking /metrics/prometheus endpoint..."
    local url=$(build_url "/metrics/prometheus")
    
    local response
    response=$(curl -s --connect-timeout $TIMEOUT "$url" 2>&1) || {
        echo -e "   ${RED}❌ Failed to fetch metrics${NC}"
        return 1
    }
    
    if echo "$response" | grep -q "verification_requests_total"; then
        echo -e "   ${GREEN}✅ Metrics endpoint OK${NC}"
    else
        echo -e "   ${RED}❌ Metrics endpoint failed - verification_requests_total not found${NC}"
        return 1
    fi
}

# Supported formats check
check_formats() {
    echo ""
    echo "3️⃣  Checking /upload/supported-formats endpoint..."
    local url=$(build_url "/upload/supported-formats")
    
    local response
    response=$(curl -s --connect-timeout $TIMEOUT "$url" 2>&1) || {
        echo -e "   ${RED}❌ Failed to fetch formats${NC}"
        return 1
    }
    
    if echo "$response" | grep -q "HEIC"; then
        echo -e "   ${GREEN}✅ Supported formats endpoint OK${NC}"
    else
        echo -e "   ${RED}❌ Supported formats failed - HEIC not found${NC}"
        return 1
    fi
}

# Response time check
check_response_time() {
    echo ""
    echo "4️⃣  Checking response time..."
    local url=$(build_url "/health")
    
    local response_time
    response_time=$(curl -o /dev/null -s --connect-timeout $TIMEOUT -w '%{time_total}' "$url" 2>/dev/null) || {
        echo -e "   ${RED}❌ Failed to measure response time${NC}"
        return 1
    }
    
    local response_time_ms
    response_time_ms=$(echo "$response_time * 1000" | bc 2>/dev/null || echo "0")
    
    if (( $(echo "$response_time_ms < 500" | bc -l) )); then
        echo -e "   ${GREEN}✅ Response time OK: ${response_time_ms}ms${NC}"
    else
        echo -e "   ${YELLOW}⚠️  Response time high: ${response_time_ms}ms${NC}"
    fi
}

# Ready endpoint check (optional)
check_ready() {
    echo ""
    echo "5️⃣  Checking /ready endpoint..."
    local url=$(build_url "/ready")
    
    if curl -f -s --connect-timeout $TIMEOUT "$url" > /dev/null 2>&1; then
        echo -e "   ${GREEN}✅ Ready endpoint OK${NC}"
    else
        echo -e "   ${YELLOW}⚠️  Ready endpoint not available (optional)${NC}"
    fi
}

# Main execution
main() {
    echo "========================================"
    echo "🚀 Health Check Script"
    echo "========================================"
    echo ""
    echo "   Base URL: $BASE_URL"
    echo "   Port:     $PORT"
    echo ""
    
    # Validate input
    if [ -z "$BASE_URL" ]; then
        echo -e "${RED}❌ Usage: $0 <base_url> [port]${NC}"
        echo ""
        echo "Examples:"
        echo "   $0 http://localhost 8000"
        echo "   $0 http://127.0.0.1"
        exit 1
    fi
    
    check_dependencies
    
    local start_time=$(date +%s)
    
    # Run all checks
    check_health || exit 1
    check_metrics || exit 1
    check_formats || exit 1
    check_response_time || exit 1
    check_ready || true
    
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    
    echo ""
    echo "========================================"
    echo -e "${GREEN}✅ All health checks passed!${NC}"
    echo "========================================"
    echo ""
    echo "   Duration: ${duration}s"
    echo "   URL:      ${BASE_URL}:${PORT}"
    echo ""
}

main "$@"