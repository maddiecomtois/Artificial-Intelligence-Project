import json
import re
import langid
from autocorrect import Speller
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize


# This function reads in the dataset to X, y, and z.
def read_dataset():
    x1 = []
    x2 = []
    y = []

    input_file = open('data/train.json')
    data = json.load(input_file)
    for item in data:
        x1.append(item['post'])
        gender = item['gender']
        if gender == "female":
            x2.append(1)
        else:
            x2.append(0)
        y.append(item['age'])
    return x1, x2, y


# This checks if a token and the following token forms a contraction.
def is_contraction(token):
    contraction = False
    index = token.find("'")
    if index != -1:
        if token[:index].isalpha() and token[index + 1:].isalpha():
            contraction = True
    return contraction


# This function removes any non-alphabetical tokens
# or tokens that are not contractions, e.g. "n't".
def is_valid_token(token):
    is_valid = False
    if token.isalpha() or is_contraction(token):
        if len(token) == len(token.encode()):
            is_valid = True
    return is_valid


# removes a wide range of emojis + other undesirable symbols. Taken from https://stackoverflow.com/a/58356570/9748476
def remove_emojis(post):
    emoj = re.compile(
        "["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u"\U00002500-\U00002BEF"  # chinese char
        u"\U00002702-\U000027B0"
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        u"\U0001f926-\U0001f937"
        u"\U00010000-\U0010ffff"
        u"\u2640-\u2642"
        u"\u2600-\u2B55"
        u"\u200d"
        u"\u23cf"
        u"\u23e9"
        u"\u231a"
        u"\ufe0f"
        u"\u3030"
        "]+",
        re.UNICODE)
    return re.sub(emoj, '', post)


# This function cleans the data by only using english words,
# autocorrecting, and removing non-alpabetical characters.
# It then stems the words.
def preprocess_data(x,fix_spelling=True,only_alpha=False,no_emojis=False,stop_words=True):
    stemmer = PorterStemmer()
    spell = Speller(fast=True)
    for idx, post in enumerate(x):
        if fix_spelling is True:
            post = spell(post)
        if no_emojis is True:
            remove_emojis(post)
        if langid.classify(post)[0] == "en":
            post = word_tokenize(post)
            valid_review = []
            for token in post:
                if only_alpha is False or is_valid_token(token):
                    valid_review.append(stemmer.stem(token))
            if stop_words:
                valid_review = [
                    word for word in valid_review
                    if word not in set(stopwords.words('english'))
                ]
            post = " ".join(valid_review)
        else:
            post = ""
        x[idx] = post
    return x

# This function categorises an age according to its age group. 
def group_ages(y):
    y_group = []
    for age in y:
        age_num = int(age)
        if age_num <=17:
            y_group.append(1)
        elif age_num <=27:
            y_group.append(2)
        else:
            y_group.append(3)
    return y_group
    
def main():
    x1, x2, y = read_dataset()

    # Preprocessing blog posts.
    print('preprocessing data')

    # There are multiple options for pre-processing automatically set to true.
    # Some examples are the removal of emojis and spelling correction.
    # If these optional pre-processing steps are unwanted, set them to False.
    results = preprocess_data(x1)

    # Grouping the ages into three categories.
    y_group = group_ages(y)
    y_num = [int(y_i) for y_i in y]
    output_data = {"posts": results, "genders": x2, "ages": y_num, "group_ages": y_group}

    # # If you want to store the data as an array of objects.
    # output_data = [{
    #     'post': post,
    #     'gender': gender,
    #     'age': age
    # } for post, gender, age in zip(result, x2, y)]

    print('writing data to file')
    json_file = open("data/output.json", "w")
    json_file.write(json.dumps(output_data))
    json_file.close()


if __name__ == "__main__":
    main()
