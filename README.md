# Classification_Using_EC2Vec
Three classification models are provided in this repository:

1. reaction-EC classification model
2. substrate-EC classification model
3. product-EC classification model
## Dependencies
1. pytorch 1.10.0
3. numpy 1.19.2
4. sklearn 0.23.2
5. pandas 1.1.3

## Data preparation
The molecules of reactions/substrates/products were embedded using mol2vec (https://github.com/samoturk/mol2vec).

The EC numbers were embedded using EC2Vec (https://github.com/MengLiu90/EC2Vec).

The embeddings were concatenated together as the input features to the classifiers.

Please refer to the data in the ```./Data/``` directory for examples.

## Model usage
To run the model, prepare the data following the instructions above and put the data under ```./Data/``` directory.

Run ```RF_product_EC_classifier.py``` to predict if a product-EC pair exists.

Run ```RF_reaction_EC_classifier.py``` to predict if a reaction-EC pair exists.

Run ```RF_substrate_EC_classifier.py``` to predict if a substrate-EC pair exists.





