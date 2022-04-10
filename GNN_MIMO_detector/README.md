This is a repository for the GNN MIMO detector https://arxiv.org/pdf/2007.05703.pdf

## Guideline 
### Training 
1. Set all training parameters in the parsers.py.
2. Start GNN training by running main_with_genData.py.
3. All training reports are saved in the directory reports --> GNN_MMSE_"true/false"_2*NRX2*NT_SNR_SNR_dB_min_train_SNR_dB_max_train_dB. There will be three files saved during the training:
  1. log.txt --> record elapsed time in each epoch.
  2. report.csv --> record number of the epoch,	training SER, and validation SER.
  3. best_epoch.txt --> record the best epoch in terms of validation SER.
4. The trained model is saved in the directory models --> GNN_MMSE_"true/false"_2*NRX2*NT_SNR_SNR_dB_min_train_SNR_dB_max_train_dB. There will be only one model that corresponds to the best epoch.


### Testing 
1. Set all testing parameters in the parsers.py.
2. Start GNN testing by running mainTestScript.py.
3. All testing reports are saved in the directory reports --> GNN_MMSE_"true/false"_2*NRX2*NT_SNR_SNR_dB_min_train_SNR_dB_max_train_dB. There will be two files saved during the testing:
  1. logTest.txt --> record elapsed time for each SNR point.
  2. TestReport.txt --> record testing SNR and SER.
4. The testing is based on the trained model saved in the directory models --> GNN_MMSE_"true/false"_2*NRX2*NT_SNR_SNR_dB_min_train_SNR_dB_max_train_dB. There will be only one model that corresponds to the best epoch.

