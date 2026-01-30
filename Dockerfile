# AlphaFactory v3.0 Dockerfile
# Optimized for Multi-Crypto DRL Trading

FROM python:3.11-slim

# Install system dependencies for C++ compilation and technical analysis libraries
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    libssl-dev \
    libffi-dev \
    python3-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project
COPY . .

# Compile the C++ Execution Engine
RUN cd execution_engine && \
    g++ -O3 -Wall -std=c++17 main.cpp matching_engine.cpp simulator.cpp -o alpha_simulator

# Expose port for Streamlit Dashboard
EXPOSE 8501

# Set environment variables
ENV PYTHONPATH="${PYTHONPATH}:/app"

# Default command (just keep it running or start training)
# For deployment, the user might want to run the dashboard
CMD ["streamlit", "run", "dashboard_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
