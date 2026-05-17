#!/bin/bash
# Start the FastAPI backend in the background
uvicorn app.main:app --host 0.0.0.0 --port 8000 &

# Start the Streamlit frontend
export API_URL="http://127.0.0.1:8000"
streamlit run app/ui.py --server.port "${PORT:-7860}" --server.address 0.0.0.0
