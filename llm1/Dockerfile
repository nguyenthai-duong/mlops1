FROM nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu22.04

# Thiết lập môi trường để tránh các yêu cầu tương tác
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Ha_Noi

# Cài đặt các công cụ cần thiết
RUN apt-get update && apt-get install -y \
    software-properties-common \
    build-essential \
    wget \
    curl \
    tzdata \
    && ln -fs /usr/share/zoneinfo/$TZ /etc/localtime \
    && dpkg-reconfigure --frontend noninteractive tzdata \
    && rm -rf /var/lib/apt/lists/*

# Cài đặt Python 3.9
RUN add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y python3.9 python3.9-dev python3.9-distutils && \
    rm -rf /var/lib/apt/lists/*

# Thiết lập Python 3.9 làm mặc định
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.9 1

# Cài đặt pip cho Python 3.9
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.9

WORKDIR /app

COPY . /app

# Cài đặt các dependency từ requirements.txt
RUN pip3 install --no-cache-dir -r requirements.txt

# Command để chạy FastAPI với Uvicorn
CMD ["uvicorn", "llm:app", "--host", "0.0.0.0", "--port", "8000"]
