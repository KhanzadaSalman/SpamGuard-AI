# Use a stable Python version
FROM python:3.10

# Hugging Face Spaces requirements for users
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

WORKDIR /app

# Install the exact versions needed for Keras 3 models
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir tensorflow==2.16.1 tf-keras fastapi uvicorn jinja2 python-multipart numpy h5py

# Copy everything into the container
COPY --chown=user . .

# Start the app on the port Hugging Face expects (7860)
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]