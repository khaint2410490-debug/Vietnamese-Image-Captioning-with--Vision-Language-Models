import kagglehub

# Download latest version
path = kagglehub.dataset_download("adityajn105/flickr8k", output_dir="./data/raw/Images")

print("Path to dataset files:", path)