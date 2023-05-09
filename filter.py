import os

# Path to input and output directories
input_dir = 'input_ctakes'
output_dir = 'filter'

# Create output directory if it does not exist
if not os.path.exists(output_dir):
    os.mkdir(output_dir)

# Loop through all files in input directory
for filename in os.listdir(input_dir):
    # Check if file is a text file
    if filename.endswith('.txt'):
        # Create input file path and output file path
        input_file = os.path.join(input_dir, filename)
        output_file = os.path.join(output_dir, filename)
        # Open input file and output file
        with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
            # Loop through each line in the input file
            for line in infile:
                # Split line into words and filter out words less than 4 characters
                words = [word for word in line.split() if len(word) >= 5]
                # Write filtered words to output file
                outfile.write(' '.join(words) + '\n')
