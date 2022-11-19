FROM jupyter/scipy-notebook

WORKDIR /app

# Copy the files
COPY . .

# Create the environment:
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Run jupyter
CMD jupyter notebook --ip=0.0.0.0 --port=8888 --allow-root
CMD python _db.py
CMD python _preprocess.py
CMD python _loaders.py
CMD python app.py
