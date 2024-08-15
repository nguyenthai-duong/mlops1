FROM python:3.9-slim


WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .

# Expose cổng 5000
EXPOSE 5000

# Chạy ứng dụng
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "5000"]
