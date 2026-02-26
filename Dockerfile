FROM python:3.10

WORKDIR /app

RUN apt-get update && apt-get install -y \
    git \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1 \
    build-essential \
    cmake

COPY . /app

RUN pip install --upgrade pip
RUN pip install torch==2.1.2
RUN pip install opencv-python numpy fastapi uvicorn pillow
RUN pip install 'git+https://github.com/facebookresearch/detectron2.git'

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
