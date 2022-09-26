## Instructions for FEMNIST. 
1) Download and unzip "Test_Digits_Upper_60k.dat.zip" and "Train_Digits_Upper_60k.dat.zip". These files include 10,000 and 60,000 samples respectively of the Upper case characters and digits form the Extended MNIST dataset.
2) Run "Manage_Dataset.py" to split data to clients and create corresponding dataloaders.
3) Run "Speeds.py" and "Femnist_Model.py" to create computational speeds for all clients and a common inital model.
4) Run "Femnist_Letters.py" to obtain the experimental results.
5) Run "Plotter.py" to plot the corresponding plots. 
