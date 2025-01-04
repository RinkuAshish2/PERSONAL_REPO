import SimpleITK as sitk
import numpy as np
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import Toplevel

def register_images(fixed_image_path, moving_image_path, output_path="registered_image.tiff"):
    """
    Registers two images using SimpleITK with a translation transformation.
    
    Parameters:
    - fixed_image_path: Path to the fixed (reference) image.
    - moving_image_path: Path to the moving (target) image to be aligned.
    - output_path: Path where the registered image will be saved.
    """
    # Load images
    fixed_image = sitk.ReadImage(fixed_image_path, sitk.sitkFloat32)
    moving_image = sitk.ReadImage(moving_image_path, sitk.sitkFloat32)
    
    # Perform image registration using ImageRegistrationMethod
    registration_method = sitk.ImageRegistrationMethod()
    
    # Set up the metric (MeanSquares)
    registration_method.SetMetricAsMeanSquares()
    
    # Set up the optimizer (RegularStepGradientDescent)
    registration_method.SetOptimizerAsRegularStepGradientDescent(learningRate=1.0, minStep=0.001, numberOfIterations=100)
    
    # Set up the transform (Translation)
    translation_transform = sitk.TranslationTransform(fixed_image.GetDimension())
    registration_method.SetInitialTransform(translation_transform)
    
    # Perform the registration
    result_transform = registration_method.Execute(fixed_image, moving_image)
    
    # Apply the resulting transform to the moving image
    resampled_image = sitk.Resample(moving_image, fixed_image, result_transform, sitk.sitkLinear, 0.0, moving_image.GetPixelID())
    
    # Convert the result to a NumPy array and save using PIL
    nda_result = sitk.GetArrayFromImage(resampled_image)
    nda_result = nda_result.astype(np.uint8)
    
    im_result = Image.fromarray(nda_result)
    im_result.convert("L").save(output_path, "TIFF")
    print(f"Registered image saved to {output_path}")

def image_cleanup(img_array, threshold, aggressiveness):
    """
    Cleans up artifacts in the registered image using neighboring pixel averaging.

    Parameters:
    - img_array: NumPy array of the image to clean.
    - threshold: Intensity threshold to determine which pixels to process.
    - aggressiveness: Aggressiveness of artifact removal.
    """
    def is_artifact(arr, i, j, aggr):
        avg_pix = np.mean([arr[i, j+1], arr[i, j-1], arr[i+1, j], arr[i-1, j]])
        return avg_pix < aggr

    def average_neighbors(arr, i, j):
        return np.mean([arr[i, j+1], arr[i, j-1], arr[i+1, j], arr[i-1, j]])
    
    img_cleaned = img_array.copy()
    for i in range(1, img_array.shape[0] - 1):
        for j in range(1, img_array.shape[1] - 1):
            if img_array[i, j] >= threshold and is_artifact(img_array, i, j, aggressiveness):
                img_cleaned[i, j] = average_neighbors(img_array, i, j)
    return img_cleaned.astype(np.uint8)

def show_image_in_popup(image):
    """
    Displays the given image in a Tkinter pop-up window.
    """
    root = tk.Tk()
    root.title("Cleaned Image")

    # Convert the PIL image to a format Tkinter can use
    img_tk = ImageTk.PhotoImage(image)

    # Create a label widget with the image
    label = tk.Label(root, image=img_tk)
    label.pack()

    # Run the Tkinter event loop
    root.mainloop()

def main():
    # Paths to your images
    fixed_image_path = "BrainProtonDensity_mod.png"
    moving_image_path = "BrainProtonDensity2_mod.png"
    output_registered_path = "registered_image.tiff"
    output_cleaned_path = "cleaned_registered_image.tiff"
    
    # Step 1: Register the images
    register_images(fixed_image_path, moving_image_path, output_registered_path)
    
    # Step 2: Load registered image and apply cleanup
    registered_image = Image.open(output_registered_path)
    registered_array = np.array(registered_image)
    
    # Clean up artifacts in the registered image
    cleaned_array = image_cleanup(registered_array, threshold=254, aggressiveness=128)
    
    # Save the cleaned image
    cleaned_image = Image.fromarray(cleaned_array)
    cleaned_image.convert("L").save(output_cleaned_path, "TIFF")
    print(f"Cleaned registered image saved to {output_cleaned_path}")

    # Step 3: Show the cleaned image in a pop-up window
    show_image_in_popup(cleaned_image)

if __name__ == "__main__":
    main()
