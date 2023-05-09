# Pre-Requisites


* Obtain access and data from [MIMIC3] (https://physionet.org/content/mimiciii/1.4/) dataset, [i2b2 Obesity Challenge] (https://www.i2b2.org/NLP/Obesity/Main.php) NLP dataset, and [UMLS] (https://www.nlm.nih.gov/research/umls/index.html). Please do this now if you are not pre-approved for access.


# Extract CUIs from MIMIC III patient data

* The notes are in the NOTEVENTS.csv file. 
* Run ParseNOTEEVENTS.ipynb to parse the NOTEEVENTS clinical texts and output to a directory.
* Run filter.py to filter out all words less than 5 characters (for faster mapping later on).
* Run runcui.ipynb to map the clinical texts into UMLS CUIs. Note that this step is going to take approximately 48 hours on a normal M1 laptop. Please run it on the background.

# To train the billing code
* The Codes directory is to use for training billing codes.
* You will cd into Codes, and run the ft.py cuis.cfg. Note that this step is expected to take a lot of resources. A 32 GB machine with Dedicated Graphics card is strongly recommended.

# To run the experiments with i2b2 data:
* The experient data is located at the Comorbidity directory.
* cd into Comorbidity directory, and run svm.py sparse.cfg.  For dense, please run svm.py dense.cfg. 
* For Dense, Please ensure that the maxlen is adjusted in the dense.cfg before you ran. If you don't know the number, simply set it to a random integer and run one time and observe what is the expected number.


