FROM jupyter/scipy-notebook

WORKDIR /app

# Copy the files
COPY . .

# Create the environment:
RUN pip install --upgrade pip
RUN pip install mlflow
RUN pip install torch
RUN pip install scikit-learn
RUN pip install nltk

# Run jupyter
CMD jupyter notebook --ip=0.0.0.0 --port=8888 --allow-root
CMD python train.py