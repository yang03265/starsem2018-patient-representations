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

# Sample Result
* A sample result can be found in the following diagram. You should be getting this number if you are running with the correct data:


| Disease                |   p   |   r   |   f1   |
|------------------------|-------|-------|--------|
| Asthma                 | 0.837 | 0.702 | 0.745  |
| CAD                    | 0.556 | 0.554 | 0.555  |
| CHF                    | 0.73  | 0.675 | 0.694  |
| Diabetes               | 0.83  | 0.8   | 0.812  |
| GERD                   | 0.566 | 0.453 | 0.477  |
| Gallstones             | 0.602 | 0.544 | 0.548  |
| Gout                   | 0.905 | 0.693 | 0.751  |
| Hypercholesterolemia    | 0.771 | 0.772 | 0.772  |
| Hypertriglyceridemia    | 0.896 | 0.599 | 0.65   |
| OA                     | 0.511 | 0.421 | 0.438  |
| OSA                    | 0.523 | 0.437 | 0.461  |
| Obesity                | 0.727 | 0.72  | 0.722  |
| PVD                    | 0.629 | 0.532 | 0.567  |
| Venous Insufficiency   | 0.67  | 0.561 | 0.584  |
| Average p = 0.70189659 |       |       |        |
| Average r = 0.61217949 |       |       |        |
| Average f1 = 0.6343698 |       |       |        |


