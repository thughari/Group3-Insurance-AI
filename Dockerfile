FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Expose Streamlit's port
EXPOSE 7860 

ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Make the script executable and run it
# Default command for Hugging Face Spaces
CMD ["bash", "start.sh"]
