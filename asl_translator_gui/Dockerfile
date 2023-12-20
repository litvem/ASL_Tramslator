# Use an official Python runtime as a base image
FROM python:3.11.2

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt /app/

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the current directory contents into the container
COPY . /app/

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
# Collect static files
RUN python manage.py collectstatic --noinput

# Run the application
# CMD ["gunicorn", "--bind", "0.0.0.0:8000", "asl_translator_gui.wsgi:application"]
CMD ["python", "manage.py", "runserver", "0.0.0.0:8000"]