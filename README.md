## Start Machine Learning project.
Aim of the project The aim is to build a predictive model to find out the sales of each product at a particular store so that it would help the decision makers at BigMart to find out the properties of any product or store, which play a key role in increasing the overall sales.

Predict sales of each item on particular store

Data Description Variable - Description Item_Identifier - Unique product ID Item_Weight - Weight of product Item_Fat_Content - Whether the product is low fat or not Item_Visibility - The % of total display area of all products in a store allocated to the particular product Item_Type - The category to which the product belongs Item_MRP - Maximum Retail Price (list price) of the product Outlet_Identifier - Unique store ID Outlet_Establishment_Year - The year in which store was established Outlet_Size - The size of the store in terms of ground area covered Outlet_Location_Type - The type of city in which the store is located Outlet_Type - Whether the outlet is just a grocery store or some sort of supermarket Item_Outlet_Sales - Sales of the product in the particular store. This is the outcome variable to be predicted.

Deployed link:https://ineuronstoresalesprediction.herokuapp.com/
### Software and account Requirement.
HLD:[SS Low_Level_Design.docx](https://github.com/codeofelango/ineuron_storesales/files/9238715/SS.Low_Level_Design.docx)
Wirefram:[Wireframe.docx](https://github.com/codeofelango/ineuron_storesales/files/9238716/Wireframe.docx)
Detail report:[Detail Project Report.pptx](https://github.com/codeofelango/ineuron_storesales/files/9238718/Detail.Project.Report.pptx)
Architecture:[Architecture.docx](https://github.com/codeofelango/ineuron_storesales/files/9238719/Architecture.docx)
LLD:[SS High_Level_Design.docx](https://github.com/codeofelango/ineuron_storesales/files/9238720/SS.High_Level_Design.docx)
![predictionj](https://user-images.githubusercontent.com/85941190/182282261-6bd44721-83b9-4cf1-bbf2-e3431925fee9.PNG)
![homepage](https://user-images.githubusercontent.com/85941190/182282267-7525fe24-36d9-4160-aa52-f6379e22a46a.PNG)
![log](https://user-images.githubusercontent.com/85941190/182282268-e3a2396b-ecaa-48c8-afe9-f31b56ba4525.PNG)

1. [Github Account](https://github.com)
2. [Heroku Account](https://dashboard.heroku.com/login)
3. [VS Code IDE](https://code.visualstudio.com/download)
4. [GIT cli](https://git-scm.com/downloads)
5. [GIT Documentation](https://git-scm.com/docs/gittutorial)


Creating conda environment
```
conda create -p venv python==3.7 -y
```
```
conda activate venv/
```
OR 
```
conda activate venv
```

```
pip install -r requirements.txt
```

To Add files to git
```
git add .
```

OR
```
git add <file_name>
```

> Note: To ignore file or folder from git we can write name of file/folder in .gitignore file

To check the git status 
```
git status
```
To check all version maintained by git
```
git log
```

To create version/commit all changes by git
```
git commit -m "message"
```

To send version/changes to github
```
git push origin main
```

To check remote url 
```
git remote -v
```

To setup CI/CD pipeline in heroku we need 3 information
1. HEROKU_EMAIL = anishyadav7045075175@gmail.com
2. HEROKU_API_KEY = <>
3. HEROKU_APP_NAME = ml-regression-app

BUILD DOCKER IMAGE
```
docker build -t <image_name>:<tagname> .
```
> Note: Image name for docker must be lowercase


To list docker image
```
docker images
```

Run docker image
```
docker run -p 5000:5000 -e PORT=5000 f8c749e73678
```

To check running container in docker
```
docker ps
```

Tos stop docker conatiner
```
docker stop <container_id>
```



```
python setup.py install
```


git remote add origin https://github.com/codeofelango/ineuron_storesales.git

git branch -M main 

git add .

git commit -m "File added"

git push -f origin main

git push origin master --force


git push -u origin master

git push -u origin main


git push -f origin main

