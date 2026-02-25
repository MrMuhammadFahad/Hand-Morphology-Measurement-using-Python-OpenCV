import cv2
import numpy as np
import matplotlib.pyplot as plt


# -----------------------------
# 1. Load and Preprocess Image
# -----------------------------
def load_image(path):
    image = cv2.imread(path)
    if image is None:
        raise FileNotFoundError("Image not found. Check file path.")
    
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # binary conversion (extra processing)
    _, binary = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY)

    cv2.imwrite("gray_hand.jpg", gray)
    cv2.imwrite("binary_hand.jpg", binary)

    return rgb, gray


# -----------------------------
# 2. Distance Calculation
# -----------------------------
def calculate_distance(p1, p2):
    return np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)


# -----------------------------
# 3. Manual Point Selection
# -----------------------------
def select_two_points(image, title):
    while True:
        plt.imshow(image, cmap='gray')
        plt.title(title)
        plt.axis('on')

        print("Click TWO points then press ENTER.")

        points = plt.ginput(2, timeout=0)
        plt.close()

        if len(points) == 2:
            return points
        else:
            print("Invalid selection. Please try again.")

# -----------------------------
# 4. Finger Measurement
# -----------------------------
def measure_fingers(gray_image):
    lengths = []
    base_widths = []
    mid_widths = []

    for i in range(5):
        print(f"\n--- Finger {i+1} ---")

        # Length
        tip_base = select_two_points(
            gray_image,
            f"Finger {i+1}: Select TIP then BASE"
        )
        length = calculate_distance(tip_base[0], tip_base[1])
        lengths.append(length)

        # Base width
        base_pts = select_two_points(
            gray_image,
            f"Finger {i+1}: Select BASE WIDTH"
        )
        base_width = calculate_distance(base_pts[0], base_pts[1])
        base_widths.append(base_width)

        # Mid width
        mid_pts = select_two_points(
            gray_image,
            f"Finger {i+1}: Select MIDPOINT WIDTH"
        )
        mid_width = calculate_distance(mid_pts[0], mid_pts[1])
        mid_widths.append(mid_width)

    return lengths, base_widths, mid_widths


# -----------------------------
# 5. Data Analysis
# -----------------------------
def analyze_data(lengths, base_widths, mid_widths):
    print("\n========== RESULTS ==========")
    print("Finger Lengths (pixels):", lengths)
    print("Base Widths (pixels):", base_widths)
    print("Mid Widths (pixels):", mid_widths)

    print("\nAverage Length:", np.mean(lengths))
    print("Average Base Width:", np.mean(base_widths))
    print("Average Mid Width:", np.mean(mid_widths))


# -----------------------------
# 6. Main Program
# -----------------------------
def main():
    rgb, gray = load_image("myHand.jpeg")
    lengths, base_widths, mid_widths = measure_fingers(gray)
    analyze_data(lengths, base_widths, mid_widths)


if __name__ == "__main__":
    main()