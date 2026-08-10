#!/bin/bash
pip install -r api/requirements.txt
uvicorn api.app:app --host 0.0.0.0 --port 8000
