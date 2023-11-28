# Realtime-Environment-Monitoring

### Problem Statement
The Air Quality Control Application aims to predict air quality using machine learning algorithms. The project addresses the challenge of determining air quality based on various parameters. The binary classification problem distinguishes whether the air quality issue is caused by specific factors (positive class) or other unrelated factors (negative class).

### Solution Proposed 
This project focuses on the Air Quality Control system, predicting air quality for different locations. The positive class signifies air quality issues related to specific factors, while the negative class represents issues unrelated to air quality. The primary objective is to minimize false predictions, providing accurate assessments of air quality.

## Tech Stack Used
1. Python 
2. FastAPI 
3. Machine learning algorithms
4. Docker
5. MongoDB

## Infrastructure Required.

1. AWS S3
2. AWS EC2
3. AWS ECR
4. Git Actions
5. Terraform

## How to run?
Before we run the project, make sure that you are having MongoDB in your local system, with Compass since we are using MongoDB for data storage. You also need AWS account to access the service like S3, ECR and EC2 instances.


## Project Archietecture
![image](https://user-images.githubusercontent.com/57321948/193536768-ae704adc-32d9-4c6c-b234-79c152f756c5.png)


## Deployment Archietecture
![image](https://user-images.githubusercontent.com/57321948/193536973-4530fe7d-5509-4609-bfd2-cd702fc82423.png)


### Step 1: Clone the repository
```bash
git clone https://github.com/ShubhamGupta19/Real-Time-Environmental-Monitoring.git
```

### Step 2- Create a conda environment after opening the repository

```bash
conda create -n env python=3.7.6 -y
```

```bash
conda activate env
```

### Step 3 - Install the requirements
```bash
pip install -r requirements.txt
```

### Step 4 - Export the environment variable
```bash
export AWS_ACCESS_KEY_ID=<AWS_ACCESS_KEY_ID>

export AWS_SECRET_ACCESS_KEY=<AWS_SECRET_ACCESS_KEY>

export AWS_DEFAULT_REGION=<AWS_DEFAULT_REGION>

export MONGODB_URL="mongodb+srv://<username>:<password>@sensortopic.fulcw0i.mongodb.net/?retryWrites=true&w=majority"

```

### Step 5 - Run the application server
```bash
python app.py
```

### Step 6. Train application
```bash
http://localhost:8080/train

```

### Step 7. Prediction application
```bash
http://localhost:8080/predictdata

```

## Run locally

1. Check if the Dockerfile is available in the project directory

2. Build the Docker image
```
docker build --build-arg AWS_ACCESS_KEY_ID=<AWS_ACCESS_KEY_ID> --build-arg AWS_SECRET_ACCESS_KEY=<AWS_SECRET_ACCESS_KEY> --build-arg AWS_DEFAULT_REGION=<AWS_DEFAULT_REGION> . 

```

3. Run the Docker image
```
docker run -d -p 8080:8080 <IMAGE_NAME>
```

To run the project execute the below commmand.

```
python app.py
```
