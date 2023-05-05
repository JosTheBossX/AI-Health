# AI-Health
Physionet-challenge-2022
First run this line of code:
python train_test_split.py training_data out_dir 0.2 65
Here the training_data is the folder having the data, out_dir the folder where test and train split will be saved, 0.2 is the test split value, 65 is the random seed
This will split the training_data to test and train split in the directory out_dir

Then run this code to train the model on train_data:
python train_model.py out_dir/train_data model
Where, out_dir/train_data is the data we created from the train split, model is a output folder saving our trained model.

Then run this code to run the model on test_data and save the output predictions
python run_model.py model out_dir/test_data test_outputs
This will save the predictions as csv in test_outputs

Then run this code to evaluate the model.
python evaluate_model.py out_dir/test_data test_outputs
This will give us all the metrics.



If you want to work on CNN you can take reference from this colab file: https://colab.research.google.com/drive/1CG6DIrnaPH8sWa8VlQAmsSt922-SJRG1?usp=sharing
it needs more better data preprocessing and augmentation.

THANK YOU :) 
