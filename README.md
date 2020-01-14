### Developing Environment 
- PyCharm 3.7
- Python 3
### Required Packages
- BP-Neural Network
```python3
import numpy as np
import pandas as pd
from pybrain.tools.shortcuts import buildNetwork
from pybrain.structure import SigmoidLayer
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.datasets import SupervisedDataSet
from sklearn.preprocessing import MinMaxScaler
```
- LightGBM
```python3
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
```
- Random Forest
```python3
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
from sklearn import metrics
import numpy as np
```
### File Content
- BP Neural Network Model is in the file ```BPnetwork.py``` 
- Random Forest Model is in the file ```RandomForest.py```
- LightGBM Model is in the file ```LightGBM.py``` 
### Running Notation
- Add the data file (train.csv and test.csv) to the root folder of the project
- Load the required packages for each model
- Run the model files respectively
- ```model.txt``` saved the training model
- Only the LightGBM model conducted a prediction
- In order to predict the result of different target, you need to change the column name in line 26
```python 3
y = np.array(data.loc[:,'DistanceToFirstStop_p50'])
```
Change ```'DistanceToFirstStop_p50'``` to other target name to obtain the prediction

