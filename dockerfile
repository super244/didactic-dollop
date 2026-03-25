FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

RUN apt-get update \
    && apt-get install -y --no-install-recommends git python3-pip \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --no-cache-dir accelerate transformers peft datasets gradio

WORKDIR /app
COPY . /app

EXPOSE 7860

CMD ["python", "app.py"]
