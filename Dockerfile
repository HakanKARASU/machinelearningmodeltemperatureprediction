FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    gfortran \
    libopenblas-dev \
    liblapack-dev \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --upgrade pip setuptools wheel

# Install Python packages
RUN pip install --no-cache-dir \
    pandas \
    numpy \
    matplotlib \
    seaborn \
    scipy \
    scikit-learn \
    joblib \
    openpyxl

# Copy all application files
COPY config.py .
COPY stage1_data_loader.py .
COPY stage2_feature_engineer.py .
COPY stage3_data_splitter.py .
COPY stage4_model_trainer.py .
COPY stage5_best_selector.py .
COPY stage6_visualizer.py .
COPY stage7_subset_analyzer.py .
COPY stage8_comparator.py .
COPY main.py .

# Run main pipeline
CMD ["python3", "main.py"]
