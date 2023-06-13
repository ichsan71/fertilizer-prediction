# Use a base image with Python and Jupyter installed
FROM python:3.9

# Set the working directory to /app
WORKDIR /app

RUN pip install flask
RUN pip install tensorflow
RUN pip install tensorflow_decision_forests
# RUN pip install -r requirements.txt

# Copy the IPython Notebook file to the container
COPY . /app
ENTRYPOINT [ "python" ]
CMD ["main.py"]