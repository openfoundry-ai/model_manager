# Use an official Python runtime as a parent image
FROM python:3.12-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Set the working directory in the container
WORKDIR /app

# Copy the project files to the working directory
COPY . .

# Install PDM and use it to install dependencies
RUN pip install --no-cache-dir pdm \
    && pdm install --no-interactive

# Expose port 8000 to the outside world
EXPOSE 8000

# Run uvicorn when the container launches
CMD ["pdm", "run", "uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
