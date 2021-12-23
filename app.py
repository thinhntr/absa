import re
from io import StringIO
from pathlib import Path

import emoji
import joblib
import sklearn
import unicodedata
import numpy as np
import pandas as pd
import streamlit as st

from functools import partial
from emoji import get_emoji_regexp
from flashtext import KeywordProcessor
from sklearn.base import BaseEstimator, TransformerMixin



st.set_page_config(
    page_title="ABSA Restaurant",
    page_icon="🍣"
)

HASHTAG = 'hashtag'

class TextCleanerBase(BaseEstimator, TransformerMixin):
    def __init__(self):
        super().__init__()

        # Find emojis
        emoji = get_emoji_regexp()

        # Create preprocessing function
        self.remove_emoji      = partial(emoji.sub, '')
        self.normalize_unicode = partial(unicodedata.normalize, 'NFC')

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if not isinstance(X, pd.Series):
            X = pd.Series(X)

        return X.apply(str.lower) \
                .apply(self.remove_emoji) \
                .apply(self.normalize_unicode)
        

class TextCleaner(TextCleanerBase):
    def __init__(self):
        super().__init__()

        # Find hashtag
        hashtag = re.compile('#\S+')

        # Find price tags
        pricetag = '((?:(?:\d+[,\.]?)+) ?(?:nghìn đồng|đồng|k|vnd|d|đ))'
        pricetag = re.compile(pricetag)

        # Find special characters
        specialchar = r"[\"#$%&'()*+,\-.\/\\:;<=>@[\]^_`{|}~\n\r\t]"
        specialchar = re.compile(specialchar)

        # Spelling correction
        rules = {
            "òa":["oà"], "óa":["oá"], "ỏa":["oả"], "õa":["oã"], "ọa":["oạ"],
            "òe":["oè"], "óe":["oé"], "ỏe":["oẻ"], "õe":["oẽ"], "ọe":["oẹ"],
            "ùy":["uỳ"], "úy":["uý"], "ủy":["uỷ"], "ũy":["uỹ"], "ụy":["uỵ"],
            "ùa":["uà"], "úa":["uá"], "ủa":["uả"], "ũa":["uã"], "ụa":["uạ"],
            "xảy":["xẩy"], "bảy":["bẩy"], "gãy":["gẫy"],
            "không":["k", "hông", "ko", "khong"]}

        kp = KeywordProcessor(case_sensitive=False)
        kp.add_keywords_from_dict(rules)

        # Create preprocessing functions
        self.autocorrect          = kp.replace_keywords
        self.normalize_pricetag   = partial(pricetag.sub, 'giá_tiền')
        self.normalize_hashtag    = partial(hashtag.sub, HASHTAG)
        self.remove_specialchar   = partial(specialchar.sub, '')

    def transform(self, X):
        X = super().transform(X)

        return X.apply(self.autocorrect) \
                .apply(self.normalize_pricetag) \
                .apply(self.normalize_hashtag) \
                .apply(self.remove_specialchar)

pipeline_fp = Path('./model/pipe.joblib')

full_pipeline = joblib.load(pipeline_fp)


def classify_sentence(sentence):
    return full_pipeline.predict([sentence])[0].astype(np.uint)


def multioutput_to_multilabel(y):
    if isinstance(y, pd.DataFrame):
        y = y.values
    nrow = y.shape[0]
    ncol = y.shape[1]
    multilabel = np.zeros((nrow, 3 * ncol), dtype=bool)
    for i in range(nrow):
        for j in range(ncol):
            if y[i, j] != 0:
                pos = j * 3 + (y[i, j] - 1)
                multilabel[i, pos] = True
    return multilabel


def custom_f1_score(y_true, y_pred, average='micro', **kwargs):
    y_true = multioutput_to_multilabel(y_true)
    y_pred = multioutput_to_multilabel(y_pred)
    return sklearn.metrics.f1_score(y_true, y_pred, average=average, **kwargs)


aspects = ["FOOD#PRICES",
           "FOOD#QUALITY",
           "FOOD#STYLE&OPTIONS",
           "DRINKS#PRICES",
           "DRINKS#QUALITY",
           "DRINKS#STYLE&OPTIONS",
           "RESTAURANT#PRICES",
           "RESTAURANT#GENERAL",
           "RESTAURANT#MISCELLANEOUS",
           "SERVICE#GENERAL",
           "AMBIENCE#GENERAL",
           "LOCATION#GENERAL"]

all_attrs = [['PRICES', 'QUALITY', 'STYLE&OPTIONS'],
             ['PRICES', 'QUALITY', 'STYLE&OPTIONS'],
             ['PRICES', 'GENERAL', 'MISCELLANEOUS'],
             ['SERVICE', 'AMBIENCE', 'LOCATION']]

sentiments = ['dne', 'negative', 'neutral', 'positive']

entities = ['FOOD', 'DRINKS', 'RESTAURANT', 'OTHERS']


@st.cache
def combine_entity_attr(entities, all_attrs):
    return [f'{entity}#{attr}' for attrs, entity in zip(
        all_attrs, entities) for attr in attrs]


all_keys = combine_entity_attr(entities, all_attrs)


def display_result(result):
    for key, stmi in zip(all_keys, result):
        if not stmi:
            continue
        st.markdown(f'- **{key}**: {sentiments[stmi]}')


def label_encoder(label):
    y = [np.nan] * len(aspects)
    ap_stm = re.findall('{(.+?), ([a-z]+)}', label)

    for aspect, sentiment in ap_stm:
        idx = aspects.index(aspect)
        y[idx] = sentiment

    return y


def txt2df(uploaded_file):
    stringio = StringIO(uploaded_file.getvalue().decode('utf-8'))
    data = stringio.read().split('\n')

    df = pd.DataFrame()
    df['review'] = [review for review in data[1::4]]
    df[all_keys] = [label_encoder(label) for label in data[2::4]]

    return df


def label_decoder(encoded_label):
    label = []
    for ap_idx, sentiment in enumerate(encoded_label):
        if isinstance(sentiment, str):
            aspect = aspects[ap_idx]
            label.append(f'{{{aspect}, {sentiment}}}')
    return ', '.join(label)


@st.cache
def df2txt(df):
    X = df.review.values
    y = df.drop('review', axis=1).values
    rows = []
    for test_id, (review, label) in enumerate(zip(X, y), 1):
        label = label_decoder(label)
        rows.extend((f'#{test_id}', review, label, ''))
    text = '\n'.join(rows[:-1])
    return text


st.title('ABSA for Restaurant Review')

modes = ["Classify sentence",
         "Annotate data",
         "Compare results"]

st.sidebar.selectbox("Select a mode", modes, key='mode')
st.header(st.session_state.mode)

if st.session_state.mode == modes[0]:
    text = st.text_area(
        "Enter your review here and press Control+Enter to classify")
    if text:
        with st.spinner():
            result = classify_sentence(text)
            display_result(result)

elif st.session_state.mode == modes[1]:

    if 'prev_fileid' not in st.session_state:
        st.session_state.prev_fileid = None

    uploaded_file = st.sidebar.file_uploader('Upload texts', type='txt')

    def refresh():
        state = st.session_state
        doc_id = state.doc_id
        if state.df is None:
            return
        row = state.df.loc[doc_id]
        state.update({k: row[k] if isinstance(row[k], str)
                      else 'dne' for k in all_keys})

    if uploaded_file and st.session_state.prev_fileid != uploaded_file.id:
        df = txt2df(uploaded_file)

        st.session_state.df = df
        st.session_state.doc_id = 0
        st.session_state.ndocs = len(df)
        st.session_state.prev_fileid = uploaded_file.id
        refresh()
    elif uploaded_file is None:
        st.session_state.df = None
        st.session_state.doc_id = None

    # --------------------------------------------------

    if st.session_state.df is not None:
        txt = df2txt(st.session_state.df)

        st.sidebar.download_button(
            label='Export ⬇',
            data=txt,
            file_name='annotated_dataset.txt',
            mime='plain')

    # --------------------------------------------------

    # def on_submit():
    #     state = st.session_state
    #     doc_id = state.doc_id
    #     df = st.session_state.df
    #     df.iloc[doc_id, 1:] = [state[k] if state[k]
    #                            != 'dne' else np.nan for k in all_keys]

    def on_change_sentiment(key):
        state = st.session_state
        doc_id = state.doc_id
        df = state.df
        col_index = all_keys.index(key) + 1
        df.iloc[doc_id, col_index] = state[key] if state[key] != 'dne' else np.nan

    def on_next_prev(go_next):
        state = st.session_state
        doc_id = state.doc_id
        if go_next:
            state.doc_id = min(state.ndocs - 1, doc_id + 1)
        else:
            state.doc_id = max(0, doc_id - 1)
        refresh()

    def annotate_all():
        df = st.session_state.df
        y = full_pipeline.predict(df.review).astype(np.object_)
        y[y == 1] = 'negative'
        y[y == 2] = 'neutral'
        y[y == 3] = 'positive'
        y[y == 0] = np.nan
        df.iloc[:, 1:] = y
        refresh()

    def auto():
        df = st.session_state.df
        doc_id = st.session_state.doc_id
        text = df.loc[doc_id, 'review']
        result = classify_sentence(text)
        df.iloc[doc_id, 1:] = [sentiments[result[i]]
                               if result[i] else np.nan for i in range(len(result))]
        refresh()

    # --------------------------------------------------

    if st.session_state.df is not None:
        state = st.session_state
        doc_id = state.doc_id
        df = state.df
        doc_ids = range(state.ndocs)
        st.selectbox('Choose document', doc_ids,
                     key='doc_id', on_change=refresh)
        current_doc = df.loc[doc_id]
        st.write(current_doc[0])
        st.write(f'**{label_decoder(current_doc[1:])}**')

    if st.session_state.df is not None:
        cols = st.columns(4)
        cols[0].button('⏮ Prev', on_click=on_next_prev, args=(False, ))
        cols[1].button('Next ⏭', on_click=on_next_prev, args=(True, ))
        cols[2].button('Auto 🐢', on_click=auto)
        cols[3].button('Auto ⚡', on_click=annotate_all)


        containers = []

        for entity in entities:
            container = st.container()
            container.subheader(entity)
            containers.append(container)

        rows = [container.columns(len(attrs))
                for container, attrs in zip(containers, all_attrs)]

        for row, entity, attrs in zip(rows, entities, all_attrs):
            for column, attr in zip(row, attrs):
                key = f'{entity}#{attr}'
                column.radio(attr, sentiments, key=key,
                             on_change=on_change_sentiment, args=(key,))

        st.dataframe(st.session_state.df)
    else:
        st.write('Upload text file to begin.')
else:
    file1 = st.sidebar.file_uploader("Upload annotator 1's data", type='txt')
    df1 = None
    if file1 is not None:
        df1 = txt2df(file1)

    file2 = st.sidebar.file_uploader("Upload annotator 2's data", type='txt')
    df2 = None
    if file2 is not None:
        df2 = txt2df(file2)

    file3 = st.sidebar.file_uploader("Upload goal data", type='txt')
    df3 = None
    if file3 is not None:
        df3 = txt2df(file3)

    if df1 is not None and df2 is not None and df3 is not None:
        if (len(df1) != len(df2) != len(df3)
                or df1.review[0] != df2.review[0]
                or df1.review[0] != df3.review[0]):

            st.error('The annotated data are different')
        else:
            mapping = {np.nan: 0, 'negative': 1, 'neutral': 2, 'positive': 3}

            y1 = df1.iloc[:, 1:].replace(mapping).astype(np.uint8).values
            y2 = df2.iloc[:, 1:].replace(mapping).astype(np.uint8).values
            y3 = df3.iloc[:, 1:].replace(mapping).astype(np.uint8).values

            y1 = multioutput_to_multilabel(y1)
            y2 = multioutput_to_multilabel(y2)
            y3 = multioutput_to_multilabel(y3)

            interagree = custom_f1_score(y1, y2)
            benchmark1 = custom_f1_score(y3, y1)
            benchmark2 = custom_f1_score(y3, y2)

            st.write("**Inter-Annotator Agreement:**", interagree)
            st.write(f"**{file1.name} benchmark:**", benchmark1)
            st.write(f"**{file2.name} benchmark:**", benchmark2)

            st.subheader(file1.name)
            st.dataframe(df1)
            st.subheader(file2.name)
            st.dataframe(df2)
            st.subheader(file3.name)
            st.dataframe(df3)
    else:
        st.write("Upload 2 annotators' data and goal data to compare")
