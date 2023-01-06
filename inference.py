import fnmatch
import glob
import os
import pprint
import time
import json
import pytesseract
import spacy
from pdf2image import convert_from_path
from io import BytesIO
from PIL import Image
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
import re
from nltk.corpus import stopwords
from spacy import displacy
from tqdm import tqdm

start_time = time.time()
model = spacy.load("output08102021/model-best")
pytesseract.pytesseract.tesseract_cmd = './tesseract/x64/tesseract.exe'
path_tessdata = './tesseract/tessdata'
custom = '--tessdata-dir {}'.format(path_tessdata)
poppler_path = './poppler/bin'

emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           u"\U00010000-\U0010ffff"
                           "]+", flags=re.UNICODE)
STOPWORDS = set(stopwords.words('english'))
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

files = glob.glob('inference/*.pdf')

for root, dirs, files in os.walk('inference'):
    result = [[os.path.join(root, name), name] for name in files if fnmatch.fnmatch(name, '*.pdf')]

out = []

json_file = open('data.json', 'w')

data = {}
for x, (pdf_path, pdf_name) in enumerate(result):
    pages = convert_from_path(pdf_path, poppler_path=poppler_path)
    data['pdf_name'] = pdf_name
    data['label'] = []
    for x, page in enumerate(pages):
        with BytesIO() as f:
            page.save(f, format="png")
            f.seek(0)
            img_page = Image.open(f)
            txt = pytesseract.image_to_string(img_page, config=custom)
            txt = txt.lower()
            txt = stemmer.stem(txt)
            txt = lemmatizer.lemmatize(txt)
            txt = re.sub('[%s]' % re.escape("""!"$%&'()*./:;<=>?@[\]^_`{|}~"""), ' ', txt)  # remove punctuations
            txt = ' '.join([word for word in txt.split(' ') if word.lower() not in STOPWORDS])  # remove stopwords
            txt = re.sub('\s{2,}', " ", txt)  # remove trailing space
            txt = emoji_pattern.sub(r'', txt)
            txt = txt.replace("\n", ' ')  # remove newline \n
            predict = model(txt)
            # displacy.serve(predict, style="ent")
            for ent in predict.ents:
                # out.append([ent.text, ent.label_])
                data['label'].append([ent.text, ent.label_])
    pprint.pprint(data)
    json.dump(data,json_file)

print(out)
print("--- %s seconds ---" % (time.time() - start_time))
