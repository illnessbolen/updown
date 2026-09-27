#!/usr/bin/env bash
# Запуск одной командой: ./start.sh            -> меню
#                        ./start.sh paper      -> сразу режим (любые аргументы bot.py)
#                        ./start.sh test       -> прогнать тесты
# При первом запуске создаёт .venv, ставит зависимости и копирует .env.example в .env.
set -euo pipefail
cd "$(dirname "$0")"

PY="${PYTHON:-}"
if [ -z "$PY" ]; then
  for c in python3.13 python3.12 python3.11 python3.10 python3 python; do
    if command -v "$c" >/dev/null 2>&1 && \
       "$c" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' 2>/dev/null; then
      PY="$c"; break
    fi
  done
fi
if [ -z "$PY" ]; then
  echo "Нужен Python 3.10 или новее (python3 --version). Установите его и запустите снова." >&2
  exit 1
fi

if [ ! -x .venv/bin/python ]; then
  echo "[setup] создаю виртуальное окружение .venv ($("$PY" --version))"
  if ! "$PY" -m venv .venv; then
    echo "Не удалось создать venv. На Debian/Ubuntu: sudo apt install python3-venv" >&2
    rm -rf .venv
    exit 1
  fi
fi
VPY=.venv/bin/python

if [ ! -f .venv/.installed ] || [ requirements.txt -nt .venv/.installed ]; then
  echo "[setup] устанавливаю зависимости (один раз)"
  "$VPY" -m pip install --upgrade pip >/dev/null
  "$VPY" -m pip install -r requirements.txt
  touch .venv/.installed
fi

if [ ! -f .env ]; then
  cp .env.example .env
  echo "[setup] создан .env из .env.example — настройки можно поменять там"
fi

if [ $# -eq 0 ]; then
  cat <<'MENU'

  Polymarket Up/Down — что запустить?
    1) discover    какие Up/Down рынки сейчас активны
    2) shadow      живые данные + модель + лог сигналов (без ордеров), с записью тиков
    3) paper       то же + симуляция исполнения на виртуальном балансе, с записью тиков
    4) analyze     калибровка модели и итоги сигналов/paper
    5) stats       статистика закрытых сделок
    6) hypothesis  проверка: отличим ли edge от нуля
    7) report      еженедельный отчёт сейчас
    8) test        прогнать тесты
    0) выход
  Остановка shadow/paper: Ctrl+C. Экстренно остановить ордера: создать файл STOP в этой папке.

MENU
  read -r -p "Выбор: " choice
  case "$choice" in
    1) set -- discover ;;
    2) set -- shadow --record ;;
    3) set -- paper --record ;;
    4) set -- analyze ;;
    5) set -- stats ;;
    6) set -- hypothesis ;;
    7) set -- report ;;
    8) set -- test ;;
    *) exit 0 ;;
  esac
fi

if [ "$1" = "test" ]; then
  "$VPY" -m pip install -q -r requirements-dev.txt
  exec "$VPY" -m pytest -q
fi
exec "$VPY" bot.py "$@"
