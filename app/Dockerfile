FROM jupyter/scipy-notebook

WORKDIR /app

# Copy the files
COPY . .

# Create the environment:
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Run jupyter
RUN python _db.py
RUN python _preprocess.py
RUN python _loaders.py

CMD ["python", "app.py"]