This is a repository for the GEPNet MIMO detector https://arxiv.org/pdf/2201.03731.pdf

## Guideline 
### Training 
1. Set all training parameters in the parsers.py.
2. Start GEPNet training by running main_with_genData.py.
3. All training reports are saved in the directory reports --> GEPNet_"Nr"X"Nt_list"_
"num_constell"QAM. There will be four files saved during the training:
  1. log.txt --> record elapsed time in each epoch.
  2. current_acc.txt --> Record current validation SERs.
  3. reportHPC.txt --> record number of epoch,	loss, and training accuracy.
  4. best_epoch.txt --> record the best epoch in terms of validation SER.
4. The trained model is saved in the directory models --> GEPNet_"Nr"X"Nt_list"_
"num_constell"QAM. There will be only one model that corresponds to the best epoch.


### Testing 
1. Set all testing parameters in the parsers.py.
2. Start GEPNet testing by running mainTestScript.py.
3. All testing reports are saved in the directory Test_reports --> GEPNet_"Nr"X"Nt_list"_
"num_constell"QAM. There will be two files saved during the testing:
  1. NumberofTestingData.txt --> record number of testing data for each SNR point.
  2. TestReport.txt --> record testing SNR and SER.
4. The testing is based on the trained model saved in the directory models --> GEPNet_"Nr"X"Nt_list"_"num_constell"QAM. There will be only one model that corresponds to the best epoch.
