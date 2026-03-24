FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

# Install dependencies
RUN apt-get update && apt-get install -y git python3-pip
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
RUN pip install transformers peft datasets gradio

# Copy code
COPY . /app
WORKDIR /app

# Expose port
EXPOSE 7860

# Start app
CMD ["python", "app.py"]
