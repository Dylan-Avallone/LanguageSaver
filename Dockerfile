 
# Use the official Python image as a base
FROM python:3.9

# Set the working directory inside the container
WORKDIR /app

# Copy all project files to the container
COPY . /app

# Install required dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port that the app runs on
EXPOSE 7860

# Command to run the application
CMD ["gunicorn", "-b", "0.0.0.0:7860", "app:app"]
