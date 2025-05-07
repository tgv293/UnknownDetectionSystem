FROM continuumio/miniconda3:latest

WORKDIR /app

# Cài đặt các gói hệ thống cần thiết cho OpenCV
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Cài đặt Mamba để giải quyết dependencies nhanh hơn
RUN conda install -n base -c conda-forge mamba -y

# Thêm sau dòng tạo thư mục temp_captures:
RUN mkdir -p /app/temp_captures
RUN mkdir -p /app/dataset
RUN mkdir -p /app/config

# Thêm quyền ghi vào thư mục config
RUN chmod -R 777 /app/temp_captures
RUN chmod -R 777 /app/dataset
RUN chmod -R 777 /app/config

# Sao chép các tệp cấu hình
COPY environment.yml .

# Tạo môi trường conda và cài đặt các gói bằng mamba
RUN mamba env create -f environment.yml

# Kích hoạt môi trường conda
SHELL ["conda", "run", "-n", "security-system", "/bin/bash", "-c"]

# Cài đặt các gói pip bổ sung
COPY requirements-pip.txt .
RUN pip install -r requirements-pip.txt

# Sao chép mã nguồn ứng dụng
COPY . .

# Đặt entry point để kích hoạt môi trường conda và chạy ứng dụng
CMD ["conda", "run", "--no-capture-output", "-n", "security-system", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]