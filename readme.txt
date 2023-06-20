"/configs/pixelsnail_1conv.json"
Please change the number of epochs, learning rate, batch size and print frequency that helps train the model

"CLIC_auto_dataset.py"
This is the custom dataloader to train the Encoder/Decoder pair of the architecture with the CLIC dataset

"CLIC_dataset.py"
This is the custom dataloader to train the video compression model with the CLIC dataset

"convert_to_png.py"
Converting the video files to video frames

"dataset.py"
Python file that has the custom dataloader for the Vimeo dataset

"Kodak_testin_auto_encoder.py"
Python file to test the auto encoder performance on the Kodak dataset

"Meter.py" and "metric.py"
Misc files to calculate the MSSIM metric and compute the avg over multiple metrics

The files listed below needs to be changed to train the autoencoder part of the video compression model
"model_auto_encoder.py", "train_new_model_auto_encoder.py"

These files needs to be changed to train the big RAFT video compression model
"train_new_model.py", "model_rev_raft_new_mod.py"

These files needs to be changed to train the small RAFT video compression model
"train_rev.py", "model_rev_raft.py", and "model_rev_raft_sim_entr.py" (This python file especially contains the simple entropy model that estimates just the mean and not the variance)

To test the small RAFT video compression model we need to change the files
"VTL_testing.py", "model_org_sim_entr.py" 

To test the Big RAFT video compression model we need to change these files 
"VTL_testing_new.py", "model_org_raft_new_model.py"



