# Input size, number of layers, layer sizes, batch size, activation, quantization
# Example command: python3 benchmark_time.py -l 14 128 128 1 -bs 32 -i 10000 -p both -u False -a tanh

# Input Size
python3 benchmark_time.py -l 1 128 128 1 -bs 32 -i 10000 -p both -u False -a tanh &&
python3 benchmark_time.py -l 2 128 128 1 -bs 32 -i 10000 -p both -u False -a tanh &&
python3 benchmark_time.py -l 7 128 128 1 -bs 32 -i 10000 -p both -u False -a tanh &&
python3 benchmark_time.py -l 14 128 128 1 -bs 32 -i 10000 -p both -u False -a tanh &&



# Number of Layers
python3 benchmark_time.py -l 14 1 -bs 32 -i 10000 -p both -u False -a tanh &&
python3 benchmark_time.py -l 14 128 1 -bs 32 -i 10000 -p both -u False -a tanh &&
python3 benchmark_time.py -l 14 128 128 1 -bs 32 -i 10000 -p both -u False -a tanh &&
python3 benchmark_time.py -l 14 128 128 128 1 -bs 32 -i 10000 -p both -u False -a tanh &&




# Penultimate Layer Size
python3 benchmark_time.py -l 14 128 1 1 -bs 32 -i 10000 -p both -u False -a tanh &&
python3 benchmark_time.py -l 14 128 2 1 -bs 32 -i 10000 -p both -u False -a tanh &&
python3 benchmark_time.py -l 14 128 4 1 -bs 32 -i 10000 -p both -u False -a tanh &&
python3 benchmark_time.py -l 14 128 8 1 -bs 32 -i 10000 -p both -u False -a tanh &&
python3 benchmark_time.py -l 14 128 16 1 -bs 32 -i 10000 -p both -u False -a tanh &&
python3 benchmark_time.py -l 14 128 32 1 -bs 32 -i 10000 -p both -u False -a tanh &&
python3 benchmark_time.py -l 14 128 64 1 -bs 32 -i 10000 -p both -u False -a tanh &&
python3 benchmark_time.py -l 14 128 128 1 -bs 32 -i 10000 -p both -u False -a tanh &&



# Batch Size
python3 benchmark_time.py -l 14 128 128 1 -bs 1 -i 10000 -p both -u False -a tanh &&
python3 benchmark_time.py -l 14 128 128 1 -bs 2 -i 10000 -p both -u False -a tanh &&
python3 benchmark_time.py -l 14 128 128 1 -bs 8 -i 10000 -p both -u False -a tanh &&
python3 benchmark_time.py -l 14 128 128 1 -bs 32 -i 10000 -p both -u False -a tanh &&
python3 benchmark_time.py -l 14 128 128 1 -bs 128 -i 10000 -p both -u False -a tanh &&
python3 benchmark_time.py -l 14 128 128 1 -bs 1024 -i 10000 -p both -u False -a tanh &&



# Activation
python3 benchmark_time.py -l 14 128 128 1 -bs 32 -i 10000 -p both -u False -a tanh &&
python3 benchmark_time.py -l 14 128 128 1 -bs 32 -i 10000 -p both -u False -a ReLU &&
python3 benchmark_time.py -l 14 128 128 1 -bs 32 -i 10000 -p both -u False -a softmax




