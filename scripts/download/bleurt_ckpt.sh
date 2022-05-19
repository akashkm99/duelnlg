mkdir -p checkpoints/bleurt
echo "Downloading Bleurt Checkpoint .."
wget https://storage.googleapis.com/bleurt-oss/bleurt-base-128.zip -P checkpoints/bleurt
cd checkpoints/bleurt
echo "Unzipping .."
unzip bleurt-base-128.zip
