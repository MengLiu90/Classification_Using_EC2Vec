# Classification_Using_EC2Vec
Three classification models are provided in this repository:

1. reaction-EC classification model
2. substrate-EC classification model
3. product-EC classification model
## Dependencies
1. numpy 1.19.2
2. sklearn 0.23.2
3. pandas 1.1.3

## Data preparation
The molecules of reactions/substrates/products were embedded using mol2vec (https://github.com/samoturk/mol2vec).

The EC numbers were embedded using EC2Vec (https://github.com/MengLiu90/EC2Vec).

The embeddings were concatenated together as the input features to the classifiers.

Please refer to the data in the ```./Data/``` directory for examples.

## Model usage
To execute the classifiers, prepare the data as per the provided instructions and place it within the ```./Data/``` directory.
The Python scripts ```RF_product_EC_classifier.py```, ```RF_reaction_EC_classifier.py```, and ```RF_substrate_EC_classifier.py``` implement respective classifiers using the train/test split protocol.

The Python scripts ```RF_product_EC_classifier_5_fold_cv.py```, ```RF_reaction_EC_classifier_5_fold_cv.py```, and ```RF_substrate_EC_classifier_5_fold_cv.py``` implement respective classifiers using the train/test split protocol.

Run ```RF_product_EC_classifier_5_fold_cv.py``` to obtain performance metrics for predicting the existence of a product-EC pair.

Run ```RF_reaction_EC_classifier_5_fold_cv.py``` to obtain performance metrics for predicting the existence of a reaction-EC pair.

Run ```RF_substrate_EC_classifier_5_fold_cv.py``` to obtain performance metrics for predicting the existence of a substrate-EC pair.





