#!/bin/bash

# Function to display usage information
show_usage() {
    echo "Usage: $0 [command]"
    echo "Commands:"
    echo "  start     - Start all services"
    echo "  stop      - Stop all services"
    echo "  restart   - Restart all services"
    echo "  build     - Rebuild all services"
    echo "  logs      - View logs from all services"
    echo "  migrate   - Run database migrations"
    echo "  shell     - Open a shell in the API container"
    echo "  test      - Run tests"
    echo "  clean     - Remove all containers and volumes"
}

# Check if Docker Compose is installed
if ! command -v docker-compose &> /dev/null; then
    echo "Error: docker-compose is not installed"
    exit 1
fi

# Execute command based on argument
case "$1" in
    "start")
        docker-compose up -d
        ;;
    "stop")
        docker-compose down
        ;;
    "restart")
        docker-compose down
        docker-compose up -d
        ;;
    "build")
        docker-compose build
        ;;
    "logs")
        docker-compose logs -f
        ;;
    "migrate")
        docker-compose run --rm api alembic upgrade head
        ;;
    "shell")
        docker-compose run --rm api /bin/bash
        ;;
    "test")
        docker-compose run --rm api pytest
        ;;
    "clean")
        docker-compose down -v
        ;;
    *)
        show_usage
        exit 1
        ;;
esac 