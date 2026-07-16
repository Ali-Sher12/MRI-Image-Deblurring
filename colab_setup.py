# Run this as the FIRST cell in your Colab notebook, once per session.

from google.colab import drive
drive.mount('/content/drive')  # log in with your TRAIN+VAL account

# Copy data locally (fast local disk instead of slow Drive I/O during training)
# after mounting a truly fresh Drive:
!cp "/content/drive/MyDrive/IXI_data.zip" "/content/train_data.zip"
!unzip -q /content/train_data.zip -d /content/train_data

# Copy your code files locally too (store them in the same Drive account
# so they travel with the data instead of re-uploading each session)
!cp "/content/drive/MyDrive/Data_loader.py" "/content/Data_loader.py"
!cp "/content/drive/MyDrive/Model.py" "/content/Model.py"
!cp "/content/drive/MyDrive/train.py" "/content/train.py"

# Make sure Colab's Python path can see the copied .py files
import sys
sys.path.append("/content")

print("Setup complete. Data and code are ready in /content.")
