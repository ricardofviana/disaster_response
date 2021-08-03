# disaster_response
Udacity Data Enginnering Project for Data Science Nano Degree track.

# Description
This project it's focused on helping disaster response organizations. 
When a disaster occurs, people send millions and millions of communications from various sources, either direct or social media. Often it is only one in every thousand messages that might be relevant to disaster response professionals. One organization will take care of water, another will care about blocked roads, another may care about medical supplies. The way that disasters are typically responded to, is that different oranizations will take of different approaches.

So we gona use some neural language processing (NLP) with machine learning models to categorize new messages.

## Repositorie File Structure:

![image](https://user-images.githubusercontent.com/48634609/127741798-7fd305b4-9e80-4530-b8bf-66ef03a8dbed.png)

## Getting Started
Clone this repository:

``` Bash
git clone  https://github.com/ricardofviana/disaster_response.git
```
### Executing Program:
1. Run the following commands in the project's root directory to set up your database and model.
    - To run ETL pipeline that cleans data and stores in database ```python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db```
    - To run ML pipeline that trains classifier and saves ```python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl```
2. Run the following command. ```python app/run.py```
3. A link will show up in the terminal to access the application.

# Acknowledgements

- Udacity the scripts were based on the Data Science Nanodegree program
- Figure Eight for providing messages dataset to train my model
