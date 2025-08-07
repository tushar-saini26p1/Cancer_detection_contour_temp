import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Cursor


def extract_color_temp_mapping(color_bar):
    # Convert color bar to HSV for better segmentation
    hsv = cv2.cvtColor(color_bar, cv2.COLOR_BGR2HSV)
    h, w, _ = color_bar.shape

    # Extract colors along the bar and assume a linear gradient mapping
    colors = [color_bar[i, w // 2] for i in range(h)]
    colors = np.array(colors, dtype=np.uint8)

    # Generate a corresponding temperature scale (you need to adjust min/max temp)
    min_temp, max_temp = 20, 100  # Modify as per your color bar range
    temps = np.linspace(min_temp, max_temp, h)

    return colors, temps


def match_temp_to_image(thermal_img, color_bar):
    colors, temps = extract_color_temp_mapping(color_bar)

    # Convert thermal image to HSV and reshape
    thermal_hsv = cv2.cvtColor(thermal_img, cv2.COLOR_BGR2HSV)
    h, w, _ = thermal_img.shape
    thermal_reshaped = thermal_hsv.reshape(-1, 3)

    # Find closest temperature for each pixel
    mapped_temps = np.zeros((h, w))
    for i, pixel in enumerate(thermal_reshaped):
        differences = np.linalg.norm(colors - pixel, axis=1)
        min_idx = np.argmin(differences)
        mapped_temps[i // w, i % w] = temps[min_idx]

    return mapped_temps


def interactive_display(image, temp_map):
    fig, ax = plt.subplots(figsize=(10, 8))
    cax = ax.imshow(image)
    cursor = Cursor(ax, useblit=True, color='red', linewidth=1)

    def on_mouse_move(event):
        if event.xdata is not None and event.ydata is not None:
            x, y = int(event.xdata), int(event.ydata)
            temp = temp_map[y, x]
            ax.set_title(f'Temperature: {temp:.2f} °C')
            fig.canvas.draw()

    fig.canvas.mpl_connect('motion_notify_event', on_mouse_move)
    plt.show()


def main():
    # Use absolute paths to ensure correct file loading
    thermal_img_path = r"C:\Users\USER\Desktop\Tushar\MICE\Rec-000250.jpg"
    color_bar_path = r"C:\Users\USER\Desktop\Tushar\MICE\Rec-000250.jpg"

    thermal_img = cv2.imread(thermal_img_path)
    color_bar = cv2.imread(color_bar_path)

    # Check if images are loaded properly
    if thermal_img is None:
        print(f"Error: Unable to load thermal image from {thermal_img_path}")
        return
    if color_bar is None:
        print(f"Error: Unable to load color bar from {color_bar_path}")
        return

    mapped_temps = match_temp_to_image(thermal_img, color_bar)
    interactive_display(cv2.cvtColor(thermal_img, cv2.COLOR_BGR2RGB), mapped_temps)

if __name__ == "__main__":
    main()

