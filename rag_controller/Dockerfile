# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Make port 8005 available to the world outside this container
EXPOSE 8005

# Define environment variable for the URLs
ENV WEAVIATE_URL=http://weaviate.weaviate.svc.cluster.local:80
ENV VECTORIZE_URL=http://emb-svc.emb.svc.cluster.local:81/vectorize
ENV LLM_API_URL=https://nthaiduong83.serveo.net/complete

# Run app.py when the container launches
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8005"]
