# Input Size
python3 plot_st.py -xlab 'Input Size' -ylab 'Forward Pass Time (ms)' -x 1 2 7 14 -y 0.268 0.280 0.333 0.350 &&
python3 plot_st.py -xlab 'Input Size' -ylab 'Backward Pass Time (ms)' -x 1 2 7 14 -y 0.574 0.584 0.568 0.591 &&

# Number of Layers
python3 plot_st.py -xlab 'Number of Layers' -ylab 'Forward Pass Time (ms)' -x 1 2 7 14 -y 0.0649 0.255 0.369 0.455 &&
python3 plot_st.py -xlab 'Number of Layers' -ylab 'Backward Pass Time (ms)' -x 1 2 7 14 -y 0.183 0.366 0.621 0.821 &&

# Layer Size
python3 plot_st.py -xlab 'Layer Size' -ylab 'Forward Pass Time (ms)' -x 1 2 4 8 16 32 64 128 -y 0.230 0.251 0.241 0.272 0.283 0.256 0.289 0.386 &&
python3 plot_st.py -xlab 'Layer Size' -ylab 'Backward Pass Time (ms)' -x 1 2 4 8 16 32 64 128 -y 0.319 0.347 0.416 0.452 0.446 0.497 0.566 0.638 &&

# Batch Size
python3 plot_st.py -xlab 'Batch Size' -ylab 'Forward Pass Time (ms)' -x 1 2 8 32 128 256 512 -y 0.165 0.160 0.277 0.353 0.779 2.03 5.57 &&
python3 plot_st.py -xlab 'Batch Size' -ylab 'Backward Pass Time (ms)' -x 1 2 8 32 128 256 512 -y 0.432 0.425 0.532 0.601 1.36 2.37 5.67 

