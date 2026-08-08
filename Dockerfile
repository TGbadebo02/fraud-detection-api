# Use an official Python runtime as a parent image
FROM python:3.11-slim
# Set the working directory in the container
WORKDIR /app
# Copy the requirements file into the container
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
# Copy the rest of the application code into the container
COPY . .
# Run the model training script to train the fraud detection model
RUN python scripts/train_model.py
# Expose the port that the FastAPI application will run on
EXPOSE 8000
# Start the FastAPI application using Uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

