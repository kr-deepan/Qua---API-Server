# Qua---API-Server
API server - ML models - AI recommendation system - BentoML

# **Details Related to Development**

###### 

### **Phases of Development**

######  1-Started with collecting variables or attributes (input and output) required for our simulation

######  2-Identified the corresponding input and output attributes for every process at every stages (Primary, Biological, and Tertiary)

######  3-Researched and Identified the correlations between the input and output attributes that are realistic to real Water Treatment Plants

######  4-Based on the collected data dataset with about 10000 datapoints were generated for each process at each stage and the dataset of each process were combined into a single dataset for each stage with engineered correlations between the attributes realistic to Real Water Treatment Plant values

######  5-First data points for Primary stage is generated and Biological stage data points and generated based on Primary stage data points an the Tertiary stage data points are generated based on Biological stage data points

######  6-The correlations between the attributes in every dataset were visualized and analyzed

######  7-Linear Regression models using different ensemble method such as Random Forest, XgBoost and Gradient Boosting on the dataset for each stage

######  8-Gradient Boosting was found to be most efficient for all our three datasets of three stages with R2\_score of 0.82 to 0.85 for all three models

######  9-Exported all three trained ML model objects to individual .pkl format for deployment

######  10-A Rule based Optimization Recommendation Engine was created that provides with recommendations for the change in the process parameters so that we get improved or better efficiency for every process at every stage

######  11-Used BentoML framework to expose the ML models as REST APIs that is to deploy the ML models for production

######  12-Using test datapoints the working of the deployed model

######  13-The predictions are produced and the Optimization Recommendation Engine based the predefined rules applied on the input and the predicted output comes up with Recommendations which is returned along with the output attributes predicted

######  14-Initialized a GitHub repository for the ML deployment API server

######  15-Pushed the source code to the remote repository

######  16-Deployed the application to the cloud using Render (PaaS) for continuous deployment and hosting

######  17-Created a Simple Backend Server to test the API request and response



#### Further Development:

###### &nbsp;18-Created a AI Optimization Engine for Optimization of Process Parameters at 3 Stages for better process performance at each stage for based on Energy usage or Efficiency or Balance of both (Balanced)

###### &nbsp;19-Integrated that to our existing API Server

###### &nbsp;20-Updated the Backend Test Server to test the working of AI Optimization Engine



### Working of AI Optimization Engine:

###### &nbsp;1-Gets the current configurations for the process at a specific stage 

###### &nbsp;2-Works like a What-If simulator using Random Local Search (Monte Carlo)

###### &nbsp;3-Starts from the current configuration of the Process parameters at a stage

###### &nbsp;4-Randomly perturbs only the controllable plant parameters within realistic bounds, keeping water quality inputs fixed

###### &nbsp;5-For each candidate configuration uses trained ML model to predict output parameters and checks if the candidate is feasible then computes a score (efficiency vs energy) depending on the selected mode (balanced, efficiency, or energy)

###### &nbsp;6-Sorts all feasible candidates by score and returns the top K with New configurations(input parameters), Predicted output parameters and AI recommendations (rule-based recommendations) 

###### &nbsp;7-AI Optimizations for Each stage is done separately



###### 

### **Working of API Server:**

######  1-Backend Server converts the input parameters (such as BOD\_Level, Screen\_Size, etc.) set by the user at the simulation pages in to json format

######  2-Backend requests the API server with the json format data of inputs

######  3-API server converts to json format input into data format needed for the models to predict output factors (such as Efficiency\_of\_Process, Energy, etc.)

######  4-Output factors are predicted by the trained model

######  5-Using these input and output parameters the Rule-based Optimization Recommendation Engine based on the predefined rules comes up with the Optimalization Recommendations for each processes at each stage of treatment

######  6-The output attribute values predicted and the recommendation from Optimization Recommendation Engine is returned to the Backend Server in json format as response to the request from Backend Server

######  7-Backend Server parses this response from API server and passes it as the response to the user

#### 

### **Other Details:**

#### 

#### **ML models:**

######   • Primary: Gradient Boosting(Linear Regression)

######   • Biological: Gradient Boosting(Linear Regression)

######   • Tertiary: Gradient Boosting(Linear Regression)

###### 

#### **Dataset:**

######   • Synthetic (generated) with engineered correlations between attributes realistic to Real Water Treatment Plant

###### 

#### **GitHub Link for the Project:** [https://github.com/kr-deepan/Qua---API-Server](https://github.com/kr-deepan/Qua---API-Server)

###### 

#### **Link to the API Server:** [https://qua-api-server.onrender.com/](https://qua-api-server.onrender.com/)

######   • Might take some time for the server deploy as if there is no request for the server for 15 mins the service spins down or sleep (for free web services on render.com)

######   • But automatically redeploys with the first request after the 15th minute mark



#### **Folders:**

######   • **ML MODELS DEPLOYED FOR PRODUCTION**: The ML models trained and deployed using BentoML Framework for Production

######   • **BACKEND API TEST**: The test backend server to test the working of API server hosted on cloud



########################################################################################################################################----- K R DEEPAN




