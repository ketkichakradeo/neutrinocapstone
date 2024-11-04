import numpy as np
import matplotlib.pyplot as plt
import h5py
from scipy.stats import linregress

# Function to extract peak energy and tail slope
def extract_peak_and_tail_slope(waveform):
    # Find the peak index and value
    peak_index = np.argmax(waveform)
    peak_value = waveform[peak_index]

    # Calculate tail slope using the last segment of the waveform (adjust as needed)
    tail_segment = waveform[-500:]  # Using the last 500 samples for tail slope calculation
    time = np.arange(len(tail_segment))  # Create a time index for the tail segment
    slope, intercept, _, _, _ = linregress(time, tail_segment)

    return peak_index, peak_value, slope

# Main function to load data and visualize results
def main():
    # List of HDF5 files
    files = [
        "MJD_NPML_1.hdf5",
        "MJD_NPML_2.hdf5",
        "MJD_Test_0.hdf5",
        "MJD_Test_1.hdf5",
        "MJD_Test_2.hdf5",
        "MJD_Test_3.hdf5",
        "MJD_Test_4.hdf5",
        "MJD_Test_5.hdf5",
        "MJD_Train_0.hdf5",
        "MJD_Train_1.hdf5",
        "MJD_Train_2.hdf5",
        "MJD_Train_3.hdf5",
        "MJD_Train_4.hdf5",
        "MJD_Train_5.hdf5",
        "MJD_Train_6.hdf5",
        "MJD_Train_7.hdf5",
        "MJD_Train_8.hdf5",
        "MJD_Train_9.hdf5",
        "MJD_Train_10.hdf5",
        "MJD_Train_11.hdf5",
        "MJD_Train_12.hdf5",
        "MJD_Train_13.hdf5",
        "MJD_Train_14.hdf5",
        "MJD_Train_15.hdf5"
    ]

    # Select the file to analyze
    file_path = 'data/' + files[2]  # Change index to analyze different files

    # Load data from HDF5 file
    with h5py.File(file_path, 'r') as file:
        # Load the raw waveforms
        raw_waveform = np.array(file["raw_waveform"])

        # Load other labels
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

    # Visualization
    plt.figure(figsize=(10, 5))
    plt.plot(random_waveform, label='Waveform')
    plt.title(f'Random Raw Waveform (Index: {random_index})')
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
    plt.show()

# Entry point of the script
if __name__ == "__main__":
    main()

