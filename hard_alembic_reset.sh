#!/bin/bash

set -e

echo "1. Остановка backend и PostgreSQL (если нужно)..."
sudo systemctl stop gpt-backend.service || true
sudo systemctl stop postgresql || true

echo "2. Очищаем alembic миграции и кэш..."
rm -rf alembic/versions/*
rm -rf alembic/__pycache__/
rm -rf backend/alembic/versions/* || true
rm -rf backend/alembic/__pycache__/ || true
find . -name '*.pyc' -delete

echo "3. Запускаем PostgreSQL снова..."
sudo systemctl start postgresql

echo "4. Удаляем и создаём базу заново..."
sudo -u postgres dropdb gptdb || true
sudo -u postgres createdb gptdb

echo "5. Инициализируем alembic заново..."
# alembic init alembic  # только если alembic/ был удалён! Обычно не нужно.

echo "6. Генерируем новую ревизию..."
alembic revision --autogenerate -m "init schema"

echo "7. Применяем миграцию..."
alembic upgrade head

echo "Готово! Alembic и база полностью сброшены и чисты."
