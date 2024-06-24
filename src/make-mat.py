import scipy.io as sio
import numpy as np

# Create example data
X = np.random.rand(100, 5)  # 100 samples, 5 features each
y = np.random.randint(0, 2, 100)  # 100 labels (binary classification)

# Save to .mat file
mat_file_path = 'data/ex7data2.mat'
sio.savemat(mat_file_path, {'X': X, 'y': y})
print(f"Data saved to {mat_file_path}")