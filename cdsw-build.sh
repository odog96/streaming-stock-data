#!/bin/bash
#pip3 install -r requirements.txt

#pip3 install sklearn
pip3 install pandas
pip3 install yfinance
pip3 install scikit-learn
pip3 install seaborn
pip3 install tensorflow
pip3 install psutil

if command -v pip2 >/dev/null 2>&1; then
   pip2 install sklearn
fi
