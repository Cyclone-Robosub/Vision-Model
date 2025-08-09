import numpy as np

# Load the calibration data
calibration_data = np.load('legacy-examples/calibration_data.npz')

# Print all available keys in the file
print("Available data in calibration_data.npz:")
print(f"Keys: {list(calibration_data.keys())}")
print()

# Print each array with its shape and values
for key in calibration_data.keys():
    data = calibration_data[key]
    print(f"{key}:")
    print(f"  Shape: {data.shape}")
    print(f"  Data type: {data.dtype}")
    print(f"  Values:\n{data}")
    print("-" * 50)

# If you want to access specific calibration parameters (common names):
try:
    if 'camera_matrix' in calibration_data:
        print("\nCamera Matrix (K):")
        print(calibration_data['camera_matrix'])
    
    if 'dist_coeffs' in calibration_data:
        print("\nDistortion Coefficients:")
        print(calibration_data['dist_coeffs'])
    
    if 'rvecs' in calibration_data:
        print(f"\nRotation Vectors: {len(calibration_data['rvecs'])} images")
        
    if 'tvecs' in calibration_data:
        print(f"Translation Vectors: {len(calibration_data['tvecs'])} images")
        
except KeyError as e:
    print(f"Key not found: {e}")

# Don't forget to close the file
calibration_data.close()