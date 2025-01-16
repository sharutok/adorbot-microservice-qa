# Stage 1: Base image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Set the working directory inside the container
WORKDIR /app

# Install dependencies
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Install Gunicorn for production
RUN pip install --no-cache-dir gunicorn

# Copy the app code to the container
COPY . /app/

# Expose port 5000 for the Flask app
EXPOSE 5000

# Command to run Gunicorn in production mode
CMD ["gunicorn", "--timeout", "500", "--workers", "3", "-b", "0.0.0.0:5001", "server:app"]
