import pandas as pd
import string
import re
import numpy as np
import torch

from tqdm import tqdm
from torch.utils.data import Dataset
from sklearn.preprocessing import OneHotEncoder

from transformers import BertTokenizer, BertModel
from sentence_transformers import SentenceTransformer

## Preprocessing
def TextPreprocessing(text):
    try:
        # remove URL (test in regex101)
        text = re.sub(r'https?:\/\/.*', '', text, flags=re.MULTILINE)
        # remove punctation
        text = "".join([i for i in text if i not in string.punctuation])
    except TypeError:
        print("tyeperror... ", text, type(text))
    return text

class ModelDataset(Dataset):
    """
    input
    - df: dataframe from train dataset
    - df_comments: dataframe from comment.csv
    - k: for top-k comments used in this dataset
    - token_max_length: max token length for word embedding. If None, treat this dataset as sentence embedding (SBERT)
    """
    def __init__(self, df:pd.DataFrame, df_comments:pd.DataFrame, k:int,
                 token_max_length:int=None): 
        self.texts = []
        self.labels = []
        self.comments = []
        # top-k comments
        self.k = k
        self.token_max_length = token_max_length
        self.is_sentence_embedding = (token_max_length is None)

        # retrieve comments body with each id
        for idx in tqdm(range(len(df))):
            post_id = df.iloc[idx]["id"]

            # find corresponding comments
            comments = df_comments.loc[df_comments["submission_id"] == post_id]
            comments = comments.sort_values(by=["ups"], ascending=False)
            comments["clean_body"] = comments["body"].apply(lambda x:TextPreprocessing(x))

            # title
            if (self.is_sentence_embedding):
                idx_text = df.iloc[idx]["clean_title"]
            else:
                idx_text = [df.iloc[idx]["clean_title"]]

            idx_comment = []
            # index error if not enough comment
            for i in range(self.k):
                try:
                    idx_comment.append(comments.iloc[i]["clean_body"])
                except IndexError:
                    idx_comment.append("")

            idx_label = df.iloc[idx]["6_way_label"]
            
            self.texts.append(idx_text)
            self.labels.append(idx_label)
            self.comments.append(idx_comment)
            
        # label to one-hot encoding
        self.labels = np.array(self.labels)
        self.labels = self.labels.reshape(-1, 1)
        encoder = OneHotEncoder(categories=[range(6)], sparse_output=False, dtype=np.float64)
        self.labels = encoder.fit_transform(self.labels)

        # LLM encoding
        if (self.is_sentence_embedding):
            self.bert_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        else:
            self.bert_tokenizer = BertTokenizer.from_pretrained("bert-large-uncased")
            self.bert_model = BertModel.from_pretrained("bert-large-uncased")

    def __len__(self):
        return len(self.texts)

    def get_word_embedding(self, text, token_max_length):
        """
        given text,
        produce tokens (1, token_max_length, model_dim:1024) shape
        then pass tokens into Bert Model
        return its last hidden state
        """
        if (self.is_sentence_embedding):
            return None
        
        inputs = self.bert_tokenizer.batch_encode_plus(
            text,
            padding="max_length",          # pad to maximum length
            max_length=token_max_length,          # to make output shape (1, max_length, 1024)
            truncation=True,        # Truncate to max length if necess
            return_tensors='pt',    # return Python tensors
            add_special_tokens=True # Add special tokens (CLS, SEP)
        )

        # Get BERT word embeddings for title
        with torch.no_grad():
            outputs = self.bert_model(**inputs)
        embeddings = outputs.last_hidden_state
        return embeddings

    def get_sentence_embedding(self, title, comments):
        inputs = [title] 
        inputs += comments
        embeddings = self.bert_model.encode(inputs, convert_to_tensor=True)
        comment_embeddings = embeddings[1:(1+self.k), :]
        embeddings = embeddings[:1, :]
        return embeddings, comment_embeddings

    def __getitem__(self, index):
        text = self.texts[index]
        label = self.labels[index]
        comments = self.comments[index]

        if (self.is_sentence_embedding):
            # sentence embedding
            embeddings, comment_embeddings = self.get_sentence_embedding(text, comments)
        else:
            # word embedding
            embeddings = self.get_word_embedding(text, self.token_max_length)
            comment_embeddings = self.get_word_embedding(comments, self.token_max_length)

        # Return the sample (embedding and label)
        return {'embedding': embeddings, 'label': label, 'comment': comment_embeddings,
                'text_plaintext': text, 'comment_plaintext': comments}
    
if __name__ == '__main__':
    pass