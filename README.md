# GeoParser -- AGILE 2020 ('Extracting interrogative intents and concepts from geo-analytic questions')

GeoParser is a program to parse geographic questions (including geographic web search questions and analytical questions). 

## Installation
### Requirements
The program is implemented in python (version 3.6). Several libraries should be installed (use pip command to install the following libraries) before running the code:
```bash
pip install xlsxwriter
pip install matplotlib
pip install numpy 
pip install scipy 
pip install pandas
pip install sklearn
pip install wordcloud
pip install pytorch
pip install allennlp
```

## Usage
Run the script and check the results -- results are created in parsing_results, csv, graphs, ngrams folders.
Navigate to project folder and use command-prompt/terminal to run the following command:
```python
python parse.py
python visualization.py
```
Note, the program will taks more than 10 hours to be finished properly. The first command will parse and create parsing results in parsing_results folder.
Next, the second command will use parsing results to generate NGrams and word clouds.

If you are running the program for the first time, the pretrained models are going to be downloaded from AllenNLP website [Link](https://allennlp.org/).

To check the evaluation, navigate to evaluation folder in the project folder, and use command prompt or terminal to run the following command:
```python
python evaluation.py
```
The labelled questions can be found in evaluation folder.
## License
[MIT](https://opensource.org/licenses/MIT)
