# README

Currently the colab code gets stuck in the final block due to a device side assert

Eventually the code will use the valid.csv data from EmpathicDialogues, however due to a Tensor layout issue PyTorche's method of creating validation data from a training set is used. Once the current code is running and the model is being trained then the final test.csv file from EmpathicDialogues will be used to test the current accuracy of the model
