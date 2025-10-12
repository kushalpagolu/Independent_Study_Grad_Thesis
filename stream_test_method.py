@staticmethod
    def test_data_processing():
        """
        Standalone method to test the data processing pipeline with sample EEG data.
        This method will load data from an Excel file in the /data directory.
        """
        try:
            # Load data from Excel file
            file_path = "/Users/kushalpagolu/Documents/Code/epoch_tello_RL_3DBrain/data/Combined_EEG_Gyro_Data.xlsx"
            data = pd.read_excel(file_path)

            # Ensure the file contains the required columns
            required_columns = ['AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4']
            if not all(col in data.columns for col in required_columns):
                raise ValueError(f"Excel file must contain the following columns: {required_columns}")

            # Extract a single row of data for testing
            sample_data = data.iloc[0][required_columns].to_numpy()
            print(f"Sample Data Shape: {sample_data.shape}")

            # Apply preprocessing techniques
            print("\n--- Applying Preprocessing Techniques ---")
            eeg_data = apply_notch_filter(sample_data, fs=256)
            print(f"After Notch Filter Shape: {eeg_data.shape}")

            eeg_data = apply_bandpass_filter(eeg_data, lowcut=1.0, highcut=50.0, sampling_rate=256)
            print(f"After Bandpass Filter Shape: {eeg_data.shape}")

            eeg_data = common_average_reference(eeg_data)
            print(f"After Common Average Reference Shape: {eeg_data.shape}")

            eeg_data = apply_ica(eeg_data)
            print(f"After ICA Shape: {eeg_data.shape}")

            eeg_data = apply_hanning_window(eeg_data)
            print(f"After Hanning Window Shape: {eeg_data.shape}")

            eeg_data = apply_dwt_denoising(eeg_data)
            print(f"After DWT Denoising Shape: {eeg_data.shape}")

            # Extract features
            print("\n--- Extracting Features ---")
            band_power_features = compute_band_power(eeg_data, fs=256)
            print(f"Band Power Features: {band_power_features}")

            hjorth_features = compute_hjorth_parameters(eeg_data)
            print(f"Hjorth Parameters: {hjorth_features}")

            spectral_entropy = compute_spectral_entropy(eeg_data, fs=256)
            print(f"Spectral Entropy: {spectral_entropy}")

            fractal_dimension = higuchi_fractal_dimension(eeg_data)
            print(f"Higuchi Fractal Dimension: {fractal_dimension}")

            # Combine features into a feature vector
            feature_vector = np.concatenate((
                list(band_power_features.values()),
                hjorth_features,
                [spectral_entropy],
                [fractal_dimension]
            ))
            print(f"Feature Vector: {feature_vector}, Shape: {feature_vector.shape}")

        except Exception as e:
            print(f"Error in test_data_processing: {e}")