import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load the ECG image
image_path = 'norm_2x.png'
img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)


# Function to extract lead signal
def extract_lead_signal(img, roi):
    # Crop the image to the selected ROI
    x, y, w, h = roi
    lead_img = img[y:y + h, x:x + w]

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(lead_img, (5, 5), 0)

    # Thresholding to create a binary image
    _, binary = cv2.threshold(blurred, 128, 255, cv2.THRESH_BINARY_INV)

    # Morphological operations to enhance the ECG waveform
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(binary, kernel, iterations=1)
    eroded = cv2.erode(dilated, kernel, iterations=1)

    # Find contours
    contours, _ = cv2.findContours(eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create an empty list to store ECG points
    ecg_points = []

    # Iterate through contours to get the coordinates
    for contour in contours:
        for point in contour:
            px, py = point[0]
            ecg_points.append((x + px, y + h - py))  # Flip the y-coordinate

    # Sort the points by the x-coordinate to get the correct order of the ECG waveform
    ecg_points = sorted(ecg_points, key=lambda point: point[0])

    return ecg_points


# Function to select ROI and extract lead signals
def select_and_extract_leads(img):
    lead_data = {}

    print("Select the ROI for each lead and press ENTER or SPACE.")

    lead_names = ['Lead I',  'V6']
    for lead in lead_names:
        roi = cv2.selectROI(f"Select ROI for {lead}", img)
        if roi[2] == 0 or roi[3] == 0:
            print(f"Skipping {lead} due to invalid ROI.")
            continue
        lead_data[lead] = extract_lead_signal(img, roi)
        cv2.destroyAllWindows()

    return lead_data


# Extract ECG signals for each lead
lead_data = select_and_extract_leads(img)

# Convert the lead data to a DataFrame and save to CSV
all_points = []
for lead, points in lead_data.items():
    for x, y in points:
        all_points.append([lead, x, y])

df = pd.DataFrame(all_points, columns=['Lead', 'X', 'Y'])
output_csv_path = 'ecg_waveforms.csv'
df.to_csv(output_csv_path, index=False)

print(f'ECG waveform points saved to {output_csv_path}')

# Visualize the extracted points for each lead (optional)
plt.figure(figsize=(12, 8))
for lead, points in lead_data.items():
    x = [point[0] for point in points]
    y = [point[1] for point in points]
    plt.plot(x, y, label=lead)

plt.title('Extracted ECG Waveforms')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.show()
