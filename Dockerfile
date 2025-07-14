# 使用官方 Ubuntu 镜像
FROM ubuntu:22.04


# 设置默认 RTSP 地址
ENV RTSP_STREAM_URL="rtsp://localhost:8554/stream"


# 设置APT镜像源为清华，并安装基础依赖
RUN sed -i 's@http://.*archive.ubuntu.com@http://mirrors.tuna.tsinghua.edu.cn@g' /etc/apt/sources.list && \
    cat /etc/apt/sources.list && \
    apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    git ffmpeg libgl1-mesa-glx libglib2.0-0 python3-dev python3-pip build-essential \
    && rm -rf /var/lib/apt/lists/*

# 创建 python 的软链接（root 权限）
RUN ln -s /usr/bin/python3 /usr/bin/python

# 创建非 root 用户并设定工作目录
RUN useradd -m appuser && mkdir /app && chown appuser:appuser /app
USER appuser
WORKDIR /app

# 安装 MindSpore（优先 GPU 版本）
COPY requirements.txt ./ 
RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple && \
    pip install --upgrade pip && \
    ( \
        echo "尝试安装 GPU 版本..." && \
        pip install mindspore-gpu==2.5.0 && \
        python3 -c "import mindspore; mindspore.run_check()" \
    ) || ( \
        echo "GPU 环境不可用，切换 CPU 版本" && \
        pip uninstall -y mindspore-gpu mindspore && \
        pip install mindspore==2.5.0 \
    ) && \
    pip install --no-cache-dir -r requirements.txt pycocotools

# 拷贝项目并安装本地依赖
COPY --chown=appuser:appuser . . 
RUN pip install --no-cache-dir -e .

# 启动参数
CMD ["python", "demo/realtime_predict.py", \
    "--task", "detect", \
    "--weight", "weights/yolov5/yolov5s.ckpt", \
    "--config", "configs/yolov5/yolov5s.yaml", \
    "--device_target", "GPU", \
    "--camera_index", "${RTSP_STREAM_URL}", \
    "--frame_size", "1280", "720", \
    "--show_fps", "True", \
    "--save_result", "False"]
