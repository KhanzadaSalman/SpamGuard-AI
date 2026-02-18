# We upgrade to 3.11 to support Keras 3.13+
FROM python:3.11

RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

WORKDIR /app

# Upgrade pip and install the latest Keras 3 compatible versions
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir tensorflow keras fastapi uvicorn jinja2 python-multipart numpy h5py

COPY --chown=user . .

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]