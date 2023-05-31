# Use an official TensorFlow Serving runtime as the base image
FROM tensorflow/serving

# Set an environment variable specifying the model name
ENV MODEL_NAME=model

# Copy the SavedModel to the container
COPY C:/Users/ichsa/OneDrive/Documents/Kamil/Bangkit/Capstone/fertilizer-prediction/my_saved_model/saved_model /models/${MODEL_NAME}

# Expose the gRPC port for TensorFlow Serving
EXPOSE 8501

# Set the default model to serve
CMD ["tensorflow_model_server", "--port=8501", "--model_name=${MODEL_NAME}", "--model_base_path=/models/${MODEL_NAME}"]
