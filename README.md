
# Real Estate Recommendation System



## Run Locally

Clone the project

```bash
  git clone https://link-to-project
```

Go to the project directory

```bash
  cd my-project
```
We recommend you to create a new environment using conda by using
```bash
    conda create -n <environment name>
    conda activate <environment name>
```
Once inside the environment install the libraires required used using

```bash
  pip install -r requirements.txt
```
Once all the dependencies are done start, the server
```bash
    cd Server
    python Server.py
```
This will open up the server that will run the necessary sentence embedding transformer model needed for embedding the house description given by the user.

Next we will run the ``Streamlit.py`` using ``streamlit`` to get the 
app running.
```bash
cd ..
streamlit run Streamlit.py
```
This will open up a new window in your default browser and the home page will show up which will look like this.

![HomePage](HomePage.png)


You will need to enter the details as per your requirement of the house in the form. You will be required to give a small description and then double click the ``Generate recommendation`` button.

The recommendations will show up on your screen like one shown below.

![HouseImages1](HouseImages1.png)

![HouseImages2](HouseImages2.png)
## Appendix

Any additional information goes here


## Authors

- [@M Srinivasan](https://github.com/Srini2404)
- [@Siddharth Kothari](https://github.com/siddharth-kothari9403)
- [@Kalyan Ram Munagala](https://github.com/KalyanRam1234)
- [@Sankalp Kothari](https://github.com/SankalpKothari0904)


## Features

- Recommendations based on the user's input.
- Images for each of the listing.
- Fullscreen mode
- Cross platform


## Tech Stack

**UI:** Streamlit

**Server:** Flask


**ML Models** - [BLIP-Salesforce](https://arxiv.org/pdf/2201.12086),[BART-Facebook](https://arxiv.org/pdf/1910.13461),[mxbai-embed-large-v1 model - MixedBreadAI](https://huggingface.co/mixedbread-ai/mxbai-embed-large-v1).


<!--## Badges-->

<!--Add badges from somewhere like: [shields.io](https://shields.io/)-->

<!--[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](https://choosealicense.com/licenses/mit/)-->
<!--[![GPLv3 License](https://img.shields.io/badge/License-GPL%20v3-yellow.svg)](https://opensource.org/licenses/)-->
<!--[![AGPL License](https://img.shields.io/badge/license-AGPL-blue.svg)](http://www.gnu.org/licenses/agpl-3.0)-->




