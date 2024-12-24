# Base image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy source code
COPY . .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose port
EXPOSE 6000

# Run the app
CMD ["python", "server.py"]
