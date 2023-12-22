## Project name
**ASL Translator tool**

## Description
The **ASL Translator** is an advanced technological tool that leverages deep learning and computer vision techniques by using a camera to detect ASL gestures, including movements of hands, head, and facial expression, and translating them to written text, either in real-time, or based on a pre-recorded video file.<br>
At the same time, the ASL translator provides a restricted access section, accessible only using administrator credentials, which is used to manage the translation model. In this section, the site administrators can see data about existing trained models (i.e. date of training, and train and test accuracy), can select which model is currency active, and have the ability to train a new model by providing additional data points. <br>
This tool is designed to facilitate communication for individuals who are hard of hearing with others who are not familiar with sign language. We use an LSTM deep learning model for designing this tool and the model is trained with [MS-ASL Dataset](https://www.microsoft.com/en-us/download/details.aspx?id=100121). 

## Tech stack
- [Django](https://www.djangoproject.com/)
- [Tensorflow](https://www.tensorflow.org/)
- [Scikit-Learn](https://scikit-learn.org)
- [Numpy](https://numpy.org/)
- [OpenCV](https://opencv.org/)
- [Yt-dlp](https://github.com/yt-dlp/yt-dlp)
- [Matplotlib](https://matplotlib.org/)
- [Mediapipe](https://developers.google.com/mediapipe)
- [Ffmpeg](https://pypi.org/project/python-ffmpeg/)
- [Jsonschema](https://python-jsonschema.readthedocs.io/en/latest/validate/)


## Dataset
We are utilizing the [MS-ASL Dataset](https://www.microsoft.com/en-us/download/details.aspx?id=100121) dataset as our data source, which comprises three JSON files containing training, validation, and test sets. The sizes of these sets are 16054, 5287, and 4172, respectively. However, we have discovered that only roughly 70% of the data set is usable, due to a number of the source videos being made private.
The dataset also includes a comprehensive list of classes representing the words to be identified. Furthermore, a list of synonyms is provided, to enable the model to consider alternative words with similar meanings. Each data point is in the format shown below:

{"org_text": "nice", "clean_text": "nice", "start_time": 0.0, "signer_id": 38, "signer": 52, "start": 0, "end": 92, "file": "nice in American Sign Language", "label": 1, "height": 270.0, "fps": 29.917, "end_time": 3.075, "url": "www.youtube.com/watch?v=3aXS3keR8oY", "text": "nice", "box": [0.013679355382919312, 0.18594351410865784, 1.0, 0.7082176208496094], "width": 480.0}

As shown, each data point comes with a URL to a video of a person performing the corresponding sign. Other data such as the length, start time, end time, bounding boxes, etc. are also provided.


## Interactions with the system
### Retraining workflow
![Retraining workflow](/img/retraining_flow.png)

### Translation workflow
![Translation workflow](/img/translation_flow.png)

## Database schema
![Database schema](/img/db_schema.png)

## Installation guide

### Local run
In order to locally run the application, follow these steps:

1. Clone the repository into your local computer
2. Open the directory in your desired IDE 
3. Navigate to the "main" branch
```
git checkout main
```
4. navigate to the project directory 
```
cd asl_translator_gui
```
5. create a shell
```
pipenv shell
```
6. install the dependencies
```
pip install -r requirements.txt
```
7. run the application
```
python manage.py runserver
```
## Deployment

The application is deployed on Kubernetes engine on Google cloud. To access the website follow this link: http://34.140.181.224:80

The **admin account** credentials:
- <ins>Usename:</ins> admin
- <ins>Password:</ins> winter2024

In case by the time you are testing this, the website is down for running out of free credits we have from Google cloud, [here](https://www.youtube.com/watch?v=MaJpswi22UE) we provide you a video of the website deployed so that you can see the functionality. Note that the website is on a HTTP protocol, is not considered safe for your browser. Hence, the the browser will not allow you to use your camera and consequently not the live translation functionality. However, these workarounds might work on allowing the website to use your camera. The instructions below are provided <ins>only</ins> for Google Chrome browser:
1. Go to: **chrome://flags/#unsafely-treat-insecure-origin-as-secure**
2. Enable Insecure origins treated as secure
3. Add the addresses for which you want to ignore this policy: http://34.140.181.224/live/
4. Restart Chrome

For that the following steps have been taken. Note that the required files and correct project structure is in "cloud-deployment" branch.

### Docker image
1. navigate to the project directory 
```
cd asl_translator_gui
```
2. login to docker hub
```
docker login
```
3. build the docker image
```
docker build -t asl-translation-app:01 .
```
4. tag the build image 
```
docker tag asl-translation-app:01 [username]/[desired name and tag]
```
5. push the image to the hub
```
docker push [username]/[desired name and tag chosen]
```

### Connecting to Kubernetes cluster
1. Make a Kubernetes cluster in your Google cloud account
2. Install Kubectl:
- Follow this link: https://kubernetes.io/docs/tasks/tools/

3. Install Google cloud SDK:
- Follow this link: https://cloud.google.com/sdk/docs/install

4. Initialize gcloud:
```
gcloud init
```

5. Get Kubectl credintials for the project:
```
gcloud container clusters get-credentials [the name of the cluster] --location=[the location of the cluster]
```
6. Install any required plugins

7. Deploy the app:
```
kubectl apply -f polls.yaml
```

## Tests
### Pipeline tests
To run the unit tests for data prepare pipline do the following steps:
1. comment out line 17 (from website.models import *)
2. comment Lines 136 to 144 for testing like below:
```
    #last_uploaded_json_file = json.loads((Training_input.objects.latest('tr_input_id').tr_input_file).read().decode('utf-8'))
    #print(Training_input.objects.latest('tr_input_id').tr_input_file)
    # data_handler = DataHandler(db_file = os.path.abspath('data/data.db'))
    #data_handler.insert_data(json_file=last_uploaded_json_file)
    # clean_texts = [entry.get("clean_text", "") for entry in last_uploaded_json_file]
    #actions_O = np.array(clean_texts)
    #print(actions_O)
    #actions = actions_O

```

3. navigate to the project directory 
```
cd asl_translator_gui
```
4. Run the virtual environemnt by the follwoing command:  
``` 
pipenv shell 
```

5. Run the following command to run the tests:

```
python -m unittest .\pipelines\pipes\test_data_prepare_pipeline.py
```
6. After running the tests, make the pipelines/pipes/test2/MP_data empty.

### Retrain data format/shcema verification

**Note** 
This check has been added after deployment so you can not see this function in the deployed app. To see this test, you need to clone the project and run it locally.

## Usage and visuals
The system differentiates between two distinct groups of users: typical users, which use the system to obtain translation of ASL, and administrators, which maintain and update the deep learning models used to generate such translation.

When a user wishes to utilize the system, they will first navigate to the web address at which the system is currently being hosted. Once there, the user will be greeted with a home page, detailing the functionalities of the system, and the opportunity to log in. If a user doesn’t already have a system account, they must create one first before being able to utilize the system’s functionalities.

Once successfully logged in, the user will be presented with the option to either upload a video file they wish to be translated, or begin live translation. If they select the former, they will be prompted to select and upload a video file, which must be in the .mp4 format, and mustn’t exceed 10Mb in size. The user will be informed if they attempt to upload a file that doesn’t meet those constraints, and prompted to choose a different file to translate. Once the user successfully uploads a video file to be translated, the translation process will automatically begin, and the user will be notified when the translation is finished. At that time, the user will be able download and view the generated text in the form of a .txt file.

Otherwise, if the user opts to rather use the live translation functionality, they will be redirected to a page where they will be able to see the real time feed from their connected camera. When the system detects that the user has performed a sign, that sign will be translated into text, and that text will be overlaid as a caption over the top section of the camera feed. This process will continue until the user exits the live translation page.

On the other hand, if the user logs in using administrator credentials (this can be done from the same place where a regular user would log in), they will additionally be able to access and administrator only section of the site where they will be able to see the details of each currently trained model, provide additional data to train a new model, and select which model is to be used by the system to create translations.



## Project developers
- Yasamin Fazelidehkordi @yasaminf
- Emma Litvin @litvin
- Patrik Samcenko @samcenko
- Amin Mahmoudifard @aminmah

For a detailed breakdown of our contributions, visit this [wiki](https://git.chalmers.se/courses/dit826/2023/group2/ASL-translator/-/wikis/Group-Responsibilities) page.

## License
[MIT license](https://git.chalmers.se/courses/dit826/2023/group2/ASL-translator/-/blob/cloud-deployment/LICENSE)
