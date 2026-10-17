#!/bin/bash

if [ -f "$PWD/.env" ]; then
    set -o allexport
    . "$PWD/.env"
    set +o allexport
fi

CURRENT_ENVIRONMENT=$ENVIRONMENT
export ENVIRONMENT=pytest
export PYTHONPATH=$(pwd)

echo "Start tests with coverage..."
OMITTED="test*,tests/*,**/__init__.py,main.py,config/*"

python -m coverage run -m pytest --junitxml report.xml .
pytest_exit_code=$?

python -m coverage report -m --fail-under=80 --skip-covered --omit="$OMITTED" > coverage.log
coverage_exit_code=$?
cat coverage.log

python -m coverage html --skip-covered --omit="$OMITTED"
python -m coverage json --omit="$OMITTED"
python -m coverage xml --skip-empty --omit="$OMITTED"

echo "HTML report created at: htmlcov/index.html"
export ENVIRONMENT=$CURRENT_ENVIRONMENT

if [ "$pytest_exit_code" -ne 0 ]; then
    exit "$pytest_exit_code"
fi
exit "$coverage_exit_code"
