import numpy as np
from sklearn.preprocessing import LabelEncoder
import pandas as pd

# Load your data
train_data = pd.read_excel('train_data.xlsx')

# Initialize and fit LabelEncoder
label_encoder = LabelEncoder()
label_encoder.fit(train_data['target_col'])

# Save the classes to a file
np.save('label_classes.npy', label_encoder.classes_)
