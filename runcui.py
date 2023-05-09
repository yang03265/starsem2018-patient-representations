import os
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from quickumls import QuickUMLS
import concurrent.futures


nltk.download('stopwords')
filenames = {}
def check_loaded(output_directory_path):
    for filename in os.listdir(output_directory_path):
        filenames[str(filename)] = True
        

def map_text_to_cuis(text_file_path):
    with open(text_file_path, 'r') as file:
        text = file.read()
        stop_words = set(stopwords.words('english'))
        word_tokens = word_tokenize(text)
        filtered_text = [word for word in word_tokens if not word in stop_words]
        filtered_text = ' '.join(filtered_text)
        matcher = QuickUMLS(quickumls_fp='dest')
        cui_mappings = matcher.match(filtered_text, best_match=True, ignore_syntax=False)
        return cui_mappings

def parallelize(input_directory_path, output_directory_path):
    check_loaded('out')
    with concurrent.futures.ThreadPoolExecutor(max_workers=100) as executor:
        for filename in os.listdir(input_directory_path):
            if str(filename) in filenames:
                continue
            if filename.endswith('.txt'):
                print("Ends " + filename)
                input_file_path = os.path.join(input_directory_path, filename)
                output_file_path = os.path.join(output_directory_path, filename)
                future = executor.submit(map_text_to_cuis, input_file_path)
                with open(output_file_path, 'w') as file:
                    for mapping in future.result():
                        for element in mapping:
                            file.write(element['cui'] + ' ')

if __name__ == '__main__':
    parallelize('filter', 'out')
    print('done')
