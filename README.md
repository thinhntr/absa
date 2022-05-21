# Aspect-Based Sentiment Analysis for Restaurant

## Tool

### Features

- Classify a sentence
- Annotate data (for training, testing...)
- Compare results (between annotators)

### Offline

1. `pip install -r requirements.txt`
2. `streamlit run app.py`

### Online

👉 [Live App](https://share.streamlit.io/thinhntr/absa/main/app.py) 👈

## Note

- `SAEvaluate.java`: official evaluation tool from vlsp.org.vn
- `app.py`: web app to classify, annotate data, and compare annotated results
- `change_format.py`: convert original data to dataframe, and vice versa
- `notebook.ipynb`: training notebook
- `model/pipe.joblib`: trained model (based on Logistic Regression)
- `data/original`: original data
- `data/csv`: new format (converted from original data using `change_format.py`)

## Screenshots

### Classify user sentence
![](https://imgur.com/ZbnsTu1.png)

### Annotate raw data
![](https://imgur.com/ryiAjvC.png)

### Compare labeled results between annotators
![](https://imgur.com/nkCe96x.png)

## References

1. [VLSP 2018 - Aspect Based Sentiment Analysis (VABSA 2018)](https://vlsp.org.vn/vlsp2018/eval/sa)
2. [Annotation Guidelines](https://vlsp.org.vn/sites/default/files/2019-06/Guidelines-SA-Restaurant%20%285-3-2018%29.pdf)
3. [NLP@UIT at VLSP 2018: A SUPERVISED METHOD FOR ASPECT BASED SENTIMENT ANALYSIS](https://drive.google.com/file/d/1OacrdWtr47XlRlTXVsuYN7WhLdaPAeU-/view)
