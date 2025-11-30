import numpy as np
import matplotlib.pyplot as plt

# --- Configuration ---
FILE_PATH = "Sample Data.bin"
FRAME_SIZE = 1024       
MATRIX_DIM = 32         
EXPECTED_NUM_FRAMES = 12000
OUTPUT_IMAGE_FILE = 'eit_analysis_output.png' 

print(f"--- EIT Data Analysis Script ---")
print(f"Reading file: {FILE_PATH}")

# 1. Read byte stream: Read as np.float64 for maximum precision
try:
    data = np.fromfile(FILE_PATH, dtype=np.float64)
except FileNotFoundError:
    print(f"Error: The file '{FILE_PATH}' was not found.")
    exit()

# Adjust number of frames based on actual file size
if data.size < EXPECTED_NUM_FRAMES * FRAME_SIZE:
    actual_frames = data.size // FRAME_SIZE
    print(f"Warning: File size is smaller than expected. Found {actual_frames} frames instead of {EXPECTED_NUM_FRAMES}.")
else:
    actual_frames = EXPECTED_NUM_FRAMES

# 2. Store frames in matrix
eit_data_matrix = data[:actual_frames * FRAME_SIZE].reshape(actual_frames, FRAME_SIZE)

# Replace NaN, +Inf, and -Inf with 0.0 to ensure min/max/sum calculations are valid.
eit_data_matrix_cleaned = np.nan_to_num(eit_data_matrix, nan=0.0, posinf=0.0, neginf=0.0)

# --- Normalize/Scale the data to prevent overflows in plotting ---
# Subtract the overall minimum from all data points. This is the new baseline.
min_val_all = eit_data_matrix_cleaned.min()
eit_data_matrix_scaled = eit_data_matrix_cleaned - min_val_all

# Optional: Scale down very large numbers to prevent Matplotlib internal overflows.
max_diff = eit_data_matrix_scaled.max()
if max_diff > 1000.0:
    scale_factor = max_diff / 1000.0
    eit_data_matrix_scaled /= scale_factor
    print(f"Data scaled down by a factor of {scale_factor:.2f} for robust plotting.")

print(f"Successfully loaded and scaled data into a {eit_data_matrix_scaled.shape} matrix.")

# 3. Put this in a data file (Task 4)
output_file = "processed_eit_data_scaled.npy"
np.save(output_file, eit_data_matrix_scaled)
print(f"Processed matrix saved to: {output_file}")


# --- Time Series Analysis and Statistics ---

# Calculate sum of all pixel values for each frame (Tidal Volume Proxy)
frame_sums = eit_data_matrix_scaled.sum(axis=1)

# 4. Find the min, max, med (Task 9)
min_val = np.min(frame_sums)
max_val = np.max(frame_sums)
median_val = np.median(frame_sums)

print("\n--- Statistics of Frame Sums (Tidal Volume Proxy) ---")
print(f"Minimum Value: {min_val:.2f}")
print(f"Maximum Value: {max_val:.2f}")
print(f"Median Value:  {median_val:.2f}")


# 5. Classify each of the values into inspiration and expiration (Task 10)

# A. Smooth the signal for reliable slope detection
window_size = 20
frame_sums_smooth = np.convolve(frame_sums, np.ones(window_size)/window_size, mode='same')

# B. Classification: Inspiration = rising slope, Expiration = falling slope
slope = np.diff(frame_sums_smooth, prepend=frame_sums_smooth[0])
is_inspiration = slope > 0
time_index = np.arange(actual_frames)


# --- Plotting and Graphical Representation (Task 7 & 11) ---

fig = plt.figure(figsize=(14, 18))
plt.subplots_adjust(hspace=0.4, wspace=0.1)
plt.suptitle("EIT Data Processing and Visualization (Inspiration/Expiration Analysis)", fontsize=18, y=0.95)

# 1. EIT Heatmaps (Task 7)
min_idx = np.argmin(frame_sums)
max_idx = np.argmax(frame_sums)
intermediate_idx = [
    int(actual_frames * 0.25), 
    int(actual_frames * 0.50), 
    min_idx, 
    max_idx
]

for i, idx in enumerate(intermediate_idx):
    ax = fig.add_subplot(3, 4, i + 1)
    frame_2d = eit_data_matrix_scaled[idx].reshape(MATRIX_DIM, MATRIX_DIM)
    
    # Graphical Representation: High value (red), Low value (blue)
    im = ax.imshow(frame_2d, cmap='jet', origin='lower')
    ax.set_title(f"EIT Image @ Frame {idx}", fontsize=10)
    ax.axis('off')

cbar_ax = fig.add_axes([0.92, 0.68, 0.015, 0.22]) 
fig.colorbar(im, cax=cbar_ax, label='Relative Impedance Change (a.u.)')


# 2. Time Series Plot: Values vs Time (Task 8 & 11)
ax2 = fig.add_subplot(3, 1, 2)
ax2.plot(time_index, frame_sums, label='Raw Frame Sums (Values)', color='green', linewidth=1, alpha=0.6)
ax2.plot(time_index, frame_sums_smooth, label='Smoothed Sums (Mimics second line)', color='red', linewidth=1.5)

ax2.set_title('EIT Time Series (Values vs Index/Time)', fontsize=14)
ax2.set_xlabel('Index')
ax2.set_ylabel('Values (a.u.)')
ax2.legend(loc='upper left')
ax2.grid(True, linestyle='--')


# 3. Classification Plot: Inspiration vs Expiration (Task 10 & 11)
ax3 = fig.add_subplot(3, 1, 3)

# Separate values into inspiration (red) and expiration (blue) segments
insp_values = np.where(is_inspiration, frame_sums, np.nan)
exp_values = np.where(~is_inspiration, frame_sums, np.nan)

ax3.plot(time_index, insp_values, color='red', label='Inspiration', linewidth=1)
ax3.plot(time_index, exp_values, color='blue', label='Expiration', linewidth=1)

ax3.set_title('Respiration Phase Classification (Sums vs Index)', fontsize=14)
ax3.set_xlabel('Index')
ax3.set_ylabel('Sums (a.u.)')
ax3.legend(loc='upper left')
ax3.grid(True, linestyle='--')

# Save the plot to a file
plt.savefig(OUTPUT_IMAGE_FILE, dpi=300) 

print(f"\n--- Plotting Complete. Image saved to: {OUTPUT_IMAGE_FILE} ---")
