FROM python:3.10-slim

# working directory
WORKDIR /app

# dependencies copy
COPY requirements.txt .

# install dependencies
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt

# project copy
COPY . .

# port expose
EXPOSE 8000

# run command (assuming FastAPI / Flask)
CMD ["uvicorn", "backend.api:app", "--host", "0.0.0.0", "--port", "8000"]

