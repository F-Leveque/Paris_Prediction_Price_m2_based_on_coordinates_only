# Paris_Prediction_Price_m2_based_on_coordinates_only

## Overview
The objectif is to develop an application to predict the price per m² in Paris according to the position (x, y)

##  Architecture

The code is written in Python with the environment [VSC](https://code.visualstudio.com/docs/languages/python). The functions train and predict are both within the same file "application_paris.py".
They can be called by using the needed function as [arguments](https://www.tutorialspoint.com/python/python_command_line_arguments.htm)of the program.

##  Requirements

* [Pandas](https://pandas.pydata.org/docs/getting_started/install.html) (version 1.2.4)
* [Numpy](https://numpy.org/install/) (version 1.20.1)
* [Shapefile](https://pypi.org/project/pyshp/) (version 2.1.3)
* [Sklearn](https://scikit-learn.org/stable/install.html) (version 0.24.1)
* [Requests](https://pypi.org/project/requests/) (version 2.25.1)
* [Joblib](https://joblib.readthedocs.io/en/latest/installing.html) (version 1.0.1)
* [Pickle](https://pypi.org/project/pickle-mixin/) (version 4.0)

##  Call visualization
<p align="center">
<img src="img/train.PNG" alt="drawing" width="700"/>
</p>
<p align="center">
<img src="img/predict.PNG" alt="drawing" width="700"/>
</p>

In order to launch the creation of the Docker image, follow the instructions below :

- Install [docker](https://www.docker.com/) (version 20.10.6) and run the application.

- Download and extract the repository Packaging_classification_MNIST : Code -> Download .zip

- Open a shell in the directory where the file ```Dockerfile``` is saved and run the command :
```
    $ docker build -t mnist-prediction:latest .
```
Wait until the process succeded in the creation of the Docker image.

- Run the command to start the container :
```
    $ docker run -d -p 5000:5000 mnist-prediction
```

- Open your browser and use the following adress :
```
    http://localhost:5000/
```

- Use the Form to select your files and obtain your prediction ! (10 images from the test dataset are available in the folder images_test)

<p align="center">
<img src="img/app.JPG" alt="drawing" width="500"/>
</p>

- You can also obtain the prediction directly with the following command (example with images_test_2.jpg):
```
    $ curl --form file='@images_test_2.jpg' localhost:5000/api >> result.json
```

##  Workflow

At the root of the project, you find the file ```backend.py``` containing the classifier code, the pipeline preprocessing and the training method. 
The Flask application is coded in the file ```app.py```. You find also the Dockerfile (with the requirements) for the creation of the Docker image.

- The docker container launches the Flask application, accessible at the adress [http://localhost:5000](http://localhost:5000/), in your web browser.
- The File Selection Form is available and will send the images selected to the encapsulated server.
- The Classifier (under the ```backend.py``` file) is then called and realize the prediction.
- The server returns the prediction in a json file.

## Data used
Classification model trained (accuracy on test dasaset 98.6%) : SVM deg 4 polynomial + deskewing preprocessing  
Data accessible on the website http://yann.lecun.com/exdb/mnist/  
Train data : train-images-idx3-ubyte.gz + train-labels-idx1-ubyte.gz  
Test data : t10k-images-idx3-ubyte.gz + t10k-labels-idx1-ubyte.gz  

##  References
- https://stackoverflow.com/questions/40427435/extract-images-from-idx3-ubyte-file-or-gzip-via-python
- https://stackoverflow.com/questions/43577665/deskew-mnist-images
- https://aws.amazon.com/fr/ecr/
- https://aws.amazon.com/fr/ecs/?whats-new-cards.sort-by=item.additionalFields.postDateTime&whats-new-cards.sort-order=desc&ecs-blogs.sort-by=item.additionalFields.createdDate&ecs-blogs.sort-order=desc
