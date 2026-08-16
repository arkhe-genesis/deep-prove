FROM python:3.12-slim

WORKDIR /app

# Instalar dependências do sistema (para WeasyPrint, se necessário)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpango-1.0-0 libharfbuzz0b libpangoft2-1.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY litho_verifier_v310.py .
COPY litho_api.py .
COPY litho_batch.py .
COPY litho_updater.py .

# Criar diretórios para dados
RUN mkdir -p /data/specs /data/reports

EXPOSE 8000

CMD ["uvicorn", "litho_api:app", "--host", "0.0.0.0", "--port", "8000"]
