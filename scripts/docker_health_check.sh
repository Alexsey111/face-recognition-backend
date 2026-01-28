#!/bin/bash
# =============================================================================
# Docker Health Check Script
# =============================================================================
# Запускает health check для всех сервисов в Docker Compose
#
# Usage: ./scripts/docker_health_check.sh [service_name]
# Examples:
#   ./scripts/docker_health_check.sh           # все сервисы
#   ./scripts/docker_health_check.sh api       # только API
# =============================================================================

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

SERVICE=${1:-all}
COMPOSE_FILE="docker-compose.prod.yml"

echo "========================================"
echo "🐳 Docker Health Check"
echo "========================================"
echo ""
echo "   Service: $SERVICE"
echo "   Compose: $COMPOSE_FILE"
echo ""

# Check if containers are running
check_container_status() {
    local container=$1
    local status=$(docker inspect -f '{{.State.Status}}' "$container" 2>/dev/null || echo "not_found")
    echo "$status"
}

# Get container health status
get_health_status() {
    local container=$1
    docker inspect -f '{{if .State.Health}}{{.State.Health.Status}}{{else}}no_healthcheck{{end}}' "$container" 2>/dev/null || echo "unknown"
}

# Check API service
check_api() {
    echo "1️⃣  API Service"
    local container="face_recognition_api"
    local status=$(check_container_status "$container")
    
    if [ "$status" = "running" ]; then
        local health=$(get_health_status "$container")
        if [ "$health" = "healthy" ]; then
            echo -e "   ${GREEN}✅ Container running, healthy${NC}"
            return 0
        else
            echo -e "   ${YELLOW}⚠️  Container running, health: $health${NC}"
            return 1
        fi
    else
        echo -e "   ${RED}❌ Container status: $status${NC}"
        return 1
    fi
}

# Check PostgreSQL
check_postgres() {
    echo "2️⃣  PostgreSQL"
    local container="face_recognition_postgres"
    local status=$(check_container_status "$container")
    
    if [ "$status" = "running" ]; then
        local health=$(get_health_status "$container")
        if [ "$health" = "healthy" ]; then
            echo -e "   ${GREEN}✅ Healthy${NC}"
            return 0
        else
            echo -e "   ${YELLOW}⚠️  Health: $health${NC}"
            return 1
        fi
    else
        echo -e "   ${RED}❌ Status: $status${NC}"
        return 1
    fi
}

# Check Redis
check_redis() {
    echo "3️⃣  Redis"
    local container="face_recognition_redis"
    local status=$(check_container_status "$container")
    
    if [ "$status" = "running" ]; then
        local health=$(get_health_status "$container")
        if [ "$health" = "healthy" ]; then
            echo -e "   ${GREEN}✅ Healthy${NC}"
            return 0
        else
            echo -e "   ${YELLOW}⚠️  Health: $health${NC}"
            return 1
        fi
    else
        echo -e "   ${RED}❌ Status: $status${NC}"
        return 1
    fi
}

# Check MinIO
check_minio() {
    echo "4️⃣  MinIO"
    local container="face_recognition_minio"
    local status=$(check_container_status "$container")
    
    if [ "$status" = "running" ]; then
        local health=$(get_health_status "$container")
        if [ "$health" = "healthy" ]; then
            echo -e "   ${GREEN}✅ Healthy${NC}"
            return 0
        else
            echo -e "   ${YELLOW}⚠️  Health: $health${NC}"
            return 1
        fi
    else
        echo -e "   ${RED}❌ Status: $status${NC}"
        return 1
    fi
}

# Run HTTP health check
check_http_health() {
    echo ""
    echo "5️⃣  HTTP Health Endpoints"
    
    local endpoints=(
        "/health"
        "/metrics/prometheus"
        "/upload/supported-formats"
    )
    
    local all_passed=true
    for endpoint in "${endpoints[@]}"; do
        local url="http://localhost:8000$endpoint"
        if curl -sf --connect-timeout 5 "$url" > /dev/null 2>&1; then
            echo -e "   ${GREEN}✅ $endpoint${NC}"
        else
            echo -e "   ${RED}❌ $endpoint${NC}"
            all_passed=false
        fi
    done
    
    if [ "$all_passed" = false ]; then
        return 1
    fi
}

# Main
main() {
    local failed=0
    
    case "$SERVICE" in
        api)
            check_api || failed=1
            check_http_health || failed=1
            ;;
        postgres|pd)
            check_postgres || failed=1
            ;;
        redis)
            check_redis || failed=1
            ;;
        minio|s3)
            check_minio || failed=1
            ;;
        all|"")
            check_api || failed=1
            check_postgres || failed=1
            check_redis || failed=1
            check_minio || failed=1
            check_http_health || failed=1
            ;;
        *)
            echo -e "${RED}❌ Unknown service: $SERVICE${NC}"
            echo "Available: all, api, postgres, redis, minio"
            exit 1
            ;;
    esac
    
    echo ""
    echo "========================================"
    if [ $failed -eq 0 ]; then
        echo -e "${GREEN}✅ All checks passed!${NC}"
    else
        echo -e "${RED}❌ Some checks failed${NC}"
    fi
    echo "========================================"
    
    exit $failed
}

main "$@"