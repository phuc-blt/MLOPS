FROM python:3.9

# Set working directory
WORKDIR /app

# Cài đặt thư viện OpenGL (libGL.so.1) để tránh lỗi với OpenCV
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Copy files
COPY . /app

# Install dependencies
RUN pip install -r requirements.txt

# Expose port
EXPOSE 8000

# Run the app
CMD ["python", "app.py"]
