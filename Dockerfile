# Use an official Python runtime as a parent image
FROM python:3.9

# Set the working directory in the container
WORKDIR /app

# Use an official Python runtime as a parent image
FROM python:3.9

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed dependencies specified in requirements.txt
# RUN pip install --no-cache-dir numpy
# RUN pip install --no-cache-dir torch
# RUN pip install --no-cache-dir transformers
# RUN pip install --no-cache-dir streamlit
# RUN pip install --no-cache-dir Pillow
# # RUN pip install --no-cache-dir jax[cpu] jaxlib
# #RUN pip install --no-cache-dir jax-metal==0.0.4
# #RUN pip install --no-cache-dir flax
# RUN pip install --no-cache-dir huggingface_hub
# RUN pip install --no-cache-dir googletrans==4.0.0-rc1
# RUN pip install --no-cache-dir protobuf==3.20
# RUN pip install --no-cache-dir scipy


# RUN pip install --no-cache-dir tensorflow


# Install any needed dependencies specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Perform Hugging Face Hub login using the token
RUN huggingface-cli login --token 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

# Run test.py when the container launches
# CMD ["python", "test.py"]

# Make port 7000 available to the world outside this container
# EXPOSE 7001
EXPOSE 8080

ENV YOUTUBE_API_KEY=AIzaSyBpcyTyN0zpVQ5urZv9m6f9eZDVlNIgc2g
ENV OPEN_API_KEY=sk-proj-ZDINUmeoDvJo6WNibHv0T3BlbkFJdldHxoxC6gWTGIez9kYW
ENV FLASK_ENV=development
ENV TOKENIZERS_PARALLELISM=false
ENV GROQ_API_KEY=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
ENV MISTRAL_API_KEY=GjlP3WJYztpZTn1Zv8DqwVrbjsGOvhk3
ENV FLASK_APP=filter.py
ENV PORT=8080


# Define environment variable
# ENV MODEL_NAME=gpt-3.5-turbo

# Run filter.py when the container launches
# CMD ["python", "filter.py"]
CMD ["flask", "run", "--host=0.0.0.0", "--port=8080"]
