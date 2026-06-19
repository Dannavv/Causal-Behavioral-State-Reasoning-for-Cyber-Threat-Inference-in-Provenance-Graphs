FROM python:3.9-slim

WORKDIR /app

# Install system deps for tigramite and numpy
RUN apt-get update && apt-get install -y \
    gcc g++ make libopenblas-dev && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire codebase
COPY . .

# Default command: run all seeds (use a small dataset for demo)
CMD ["python", "run_all_seeds.py"]
