import json
import re
import langid
from autocorrect import Speller
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from emot.emo_unicode import EMOTICONS_EMO

emot_words = {}
valid_token = re.compile(r"^[\da-zA-Z0-1\s_\-\(\)\/!\.,\?]+$")


# This function reads in the dataset to X, y, and z.
def read_dataset():
    x1 = []
    x2 = []
    y = []

    input_file = open('../data/train.json')
    data = json.load(input_file)
    for item in data:
        x1.append(item['post'])
        gender = item['gender']
        if gender == "female":
            x2.append(1)
        else:
            x2.append(0)
        y.append(item['age'])
    return x1[0:10000], x2[0:10000], y[0:10000]


# This checks if a token and the following token forms a contraction.
def is_contraction(token):
    contraction = False
    index = token.find("'")
    if index != -1:
        if token[:index].isalpha() and token[index + 1:].isalpha():
            contraction = True
    return contraction


# This function removes any tokens that contain non-alphanumeric
# non-regular (e.g. not brackets, underscores, etc.) characters.
# An exception is made for contractions, e.g. "n't".
def is_valid_token(token):
    return valid_token.match(token) or is_contraction(token)


# replaces a wide range of emoticons with word-versions describing them.
def replace_emoticons(post):
    for emot in emot_words:
        post = re.sub(re.escape(emot), emot_words[emot], post)
    return post


def create_emot_dict():
    for emot in EMOTICONS_EMO:
        repl = " " + "_".join(EMOTICONS_EMO[emot].replace(",",
                                                          "").split()) + " "
        emot_words[emot] = repl


# This function cleans the data by only using english words,
# autocorrecting, and removing non-alpabetical characters.
# It then stems the words.
def preprocess_data(x,
                    fix_spelling=True,
                    only_valid=True,
                    replace_emot=True,
                    stop_words=True):
    stemmer = PorterStemmer()
    spell = Speller(fast=True)
    if replace_emot:
        create_emot_dict()

    for idx, post in enumerate(x):
        if fix_spelling is True:
            post = spell(post)
        if replace_emot is True:
            post = replace_emoticons(post)
        post = word_tokenize(post)
        valid_review = []
        for token in post:
            if only_valid is False or is_valid_token(token):
                valid_review.append(stemmer.stem(token))
        if stop_words:
            valid_review = [
                word for word in valid_review
                if word not in set(stopwords.words('english'))
            ]
        post = " ".join(valid_review)
        x[idx] = post
    return x


# This function categorises an age according to its age group.
def group_ages(y):
    y_group = []
    for age in y:
        if age <= 17:
            y_group.append(1)
        elif age <= 27:
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
    y_num = [int(y_i) for y_i in y]
    y_group = group_ages(y_num)
    output_data = {
        "posts": results,
        "genders": x2,
        "ages": y_num,
        "group_ages": y_group
    }

    # # If you want to store the data as an array of objects.
    # output_data = [{
    #     'post': post,
    #     'gender': gender,
    #     'age': age
    # } for post, gender, age in zip(result, x2, y)]

    print('writing data to file')
    json_file = open("../data/pre_train.json", "w")
    json_file.write(json.dumps(output_data))
    json_file.close()


if __name__ == "__main__":
    main()
