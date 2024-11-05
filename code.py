import numpy as np
import matplotlib.pyplot as plt
import h5py
from scipy.stats import linregress
import os
import argparse
import json

# Function to extract the peak and tail slope
def extract_peak_and_tail_slope(waveform):
    # Find the peak index and value
    peak_index = np.argmax(waveform)
    peak_value = waveform[peak_index]

    # Calculate tail slope using the last segment of the waveform 
    tail_segment = waveform[-500:]  # Using the last 500 samples for tail slope calculation
    time = np.arange(len(tail_segment))  # Create a time index for the tail segment
    slope, intercept, _, _, _ = linregress(time, tail_segment)

    return peak_index, peak_value, slope

# Function to process waveforms from HDF5 files
def process_waveforms(file_dir, output_dir, files, save_results):
    # To store results for later analysis
    results = []

    # Loop over each file in the list
    for file_name in files:
        file_path = os.path.join(file_dir, file_name)

        # Check if the file exists before processing it
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            continue

        with h5py.File(file_path, 'r') as file:
            # Load the raw waveforms and labels
            raw_waveform = np.array(file["raw_waveform"])
            energy_label = np.array(file["energy_label"])
            psd_label_low_avse = np.array(file["psd_label_low_avse"])
            psd_label_high_avse = np.array(file["psd_label_high_avse"])
            psd_label_dcr = np.array(file["psd_label_dcr"])
            psd_label_lq = np.array(file["psd_label_lq"])
            tp0 = np.array(file["tp0"])
            detector = np.array(file["detector"])
            run_number = np.array(file["run_number"])
            id = np.array(file["id"])

            # Select a random index
            random_index = np.random.choice(raw_waveform.shape[0])

            # Get the random waveform
            random_waveform = raw_waveform[random_index]

            # Access the labels for the selected index
            energy_value = energy_label[random_index]
            psd_low_avse_value = psd_label_low_avse[random_index]
            psd_high_avse_value = psd_label_high_avse[random_index]
            psd_dcr_value = psd_label_dcr[random_index]
            psd_lq_value = psd_label_lq[random_index]
            tp0_value = tp0[random_index]
            detector_value = detector[random_index]
            run_number_value = run_number[random_index]
            id_value = id[random_index]

        # Extract parameters: peak and tail slope
        peak_index, peak_value, tail_slope = extract_peak_and_tail_slope(random_waveform)

        # Store results for analysis
        result = {
            "file_name": file_name,
            "random_index": random_index,
            "energy": energy_value,
            "psd_low_avse": psd_low_avse_value,
            "psd_high_avse": psd_high_avse_value,
            "psd_dcr": psd_dcr_value,
            "psd_lq": psd_lq_value,
            "tp0": tp0_value,
            "detector": detector_value,
            "run_number": run_number_value,
            "id": id_value,
            "peak_index": peak_index,
            "peak_value": peak_value,
            "tail_slope": tail_slope
        }
        results.append(result)

        # Visualization
        plt.figure(figsize=(10, 5))
        plt.plot(random_waveform, label='Waveform')
        plt.title(f'Random Raw Waveform (Index: {random_index}) - {file_name}')
        plt.xlabel('Time Index (μs)')
        plt.ylabel('ADC Counts')

        textstr = (
            f"Peak Index: {peak_index}\n"
            f"Peak Value: {peak_value}\n"
            f"Tail Slope: {tail_slope}\n"
            f"Energy Label: {energy_value}\n"
            f"PSD Label Low Avse: {psd_low_avse_value}\n"
            f"PSD Label High Avse: {psd_high_avse_value}\n"
            f"PSD Label DCR: {psd_dcr_value}\n"
            f"PSD Label LQ: {psd_lq_value}\n"
            f"Start of Rising Edge: {tp0_value}\n"
            f"Detector: {detector_value}\n"
            f"Run Number: {run_number_value}\n"
            f"ID: {id_value}"
        )
        plt.gcf().text(0.45, 0.2, textstr, fontsize=10)
        plt.axvline(x=tp0_value, color='orange', linestyle='--', label='Rising Edge (tp0)')
        plt.legend()

        # Save the plot as a PNG file
        plot_filename = f"{file_name.replace('.hdf5', '')}_plot_{random_index}.png"
        if output_dir:
            plot_filename = os.path.join(output_dir, plot_filename)
        plt.savefig(plot_filename)
        plt.close()

        print(f"Processed {file_name} and saved plot: {plot_filename}")

    # Optionally, save the results to a file for later analysis
    if save_results:
        with open(os.path.join(output_dir, 'results.json'), 'w') as f:
            json.dump(results, f, indent=4)
        print("Results saved to 'results.json'.")

# Command-line argument parsing
def parse_args():
    parser = argparse.ArgumentParser(description="Analyze waveform data from HDF5 files.")
    parser.add_argument('--file-dir', type=str, required=True, help="Directory containing the HDF5 files.")
    parser.add_argument('--output-dir', type=str, default='.', help="Directory to save the plots and results (default: current directory).")
    parser.add_argument('--files', type=str, nargs='*', help="List of files to process. If not provided, all files in the directory will be processed.")
    parser.add_argument('--save-results', action='store_true', help="Save the results to a JSON file.")
    
    return parser.parse_args()

# Entry point of the script
if __name__ == "__main__":
    args = parse_args()

    # List all files in the specified directory if no files are provided
    if not args.files:
        args.files = [f for f in os.listdir(args.file_dir) if f.endswith('.hdf5')]

    process_waveforms(args.file_dir, args.output_dir, args.files, args.save_results)
