# Run this as the FIRST cell in your Colab notebook, once per session.

from google.colab import drive
drive.mount('/content/drive')  # log in with your TRAIN+VAL account

# Copy data locally (fast local disk instead of slow Drive I/O during training)
!cp -r "/content/drive/MyDrive/train_data" "/content/train_data"
!cp -r "/content/drive/MyDrive/val_data" "/content/val_data"

# Copy your code files locally too (store them in the same Drive account
# so they travel with the data instead of re-uploading each session)
!cp "/content/drive/MyDrive/Data_loader.py" "/content/Data_loader.py"
!cp "/content/drive/MyDrive/Model.py" "/content/Model.py"
!cp "/content/drive/MyDrive/train.py" "/content/train.py"

# Make sure Colab's Python path can see the copied .py files
import sys
sys.path.append("/content")

print("Setup complete. Data and code are ready in /content.")
