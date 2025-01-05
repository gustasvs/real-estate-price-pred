FROM python:3.10.7-slim
WORKDIR /app

ENV PIP_CACHE_DIR=/pip_cache

# Copy only the requirements file to utilize Docker cache
COPY requirements.txt /app/

# Install dependencies and torch from cpu
RUN pip install torch==2.2.2 torchvision==0.17.2 --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application
COPY . /app

CMD ["python", "handle_queue.py"]
