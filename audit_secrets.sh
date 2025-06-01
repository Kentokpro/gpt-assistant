#!/bin/bash

echo "=================="
echo "  АУДИТ СЕКРЕТОВ"
echo "=================="
PROJECT_ROOT="${1:-$(pwd)}"

echo
echo "1. Поиск файлов с потенциальными секретами (.env, .bak, .log, .sql, .gz, .dump, .tar):"
find "$PROJECT_ROOT" -type f \( \
    -iname "*.env" -o -iname "*.bak" -o -iname "*.log" -o -iname "*.sql" \
    -o -iname "*.gz" -o -iname "*.dump" -o -iname "*.tar" -o -iname "*.zip" \
    -o -iname "*.pkl" -o -iname "*.json" \
    \)

echo
echo "2. Поиск чувствительных слов (password, token, secret, key, dsn) во всех файлах:"
grep -r -i --color=auto -E 'password|token|secret|apikey|dsn|key' "$PROJECT_ROOT" 2>/dev/null | grep -v __pycache__ | grep -v .pyc

echo
echo "3. Проверка наличия служебных и IDE-файлов (pycache, vscode, idea):"
find "$PROJECT_ROOT" -type d \( -name "__pycache__" -o -name ".vscode" -o -name ".idea" \)

echo
echo "4. Быстрый чек docker-compose.yml на явные пароли/токены:"
if [ -f "$PROJECT_ROOT/docker-compose.yml" ]; then
    grep -iE 'password|token|secret|key' "$PROJECT_ROOT/docker-compose.yml"
else
    echo "Файл docker-compose.yml не найден."
fi

echo
echo "5. Поиск alembic.ini и вывод строки с паролем (если есть):"
if [ -f "$PROJECT_ROOT/backend/alembic.ini" ]; then
    grep -E 'url *= *' "$PROJECT_ROOT/backend/alembic.ini"
else
    find "$PROJECT_ROOT" -name "alembic.ini" -exec grep -H -E 'url *= *' {} \;
fi

echo
echo "6. Совет по .gitignore (добавить минимум):"
cat <<EOF
# Секреты
*.env
.env
.env.*
backend/.env

# Логи и дампы
*.log
*.bak
*.sql
*.gz
*.dump
*.tar
*.zip
*.pkl
*.json

# IDE, pycache
__pycache__/
.idea/
.vscode/
*.pyc
*.pyo
EOF

echo
echo "===== АУДИТ ЗАВЕРШЁН! ====="
