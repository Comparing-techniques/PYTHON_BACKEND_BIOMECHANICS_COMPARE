FROM python:3.10-alpine

WORKDIR /app

# Dependencias de sistema para compilar paquetes Python
RUN apk add --no-cache build-base

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8001

CMD ["uvicorn", "app.main:app_principal", "--host", "0.0.0.0", "--port", "8001"]
