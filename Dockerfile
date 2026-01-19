FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy files
COPY . /app

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose port
EXPOSE 5000

# Start app with gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app.main:app"]
