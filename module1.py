import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk

import cv2
import mediapipe as mp
import matplotlib.pyplot as plt
import numpy as np

import requests
from PIL import Image, ImageEnhance
from io import BytesIO

class ImageMergerApp:

    def __init__(self, root):
        self.root = root
        self.root.title("Image Merger App")
        
        # Create Frames
        self.frame1 = tk.Frame(root, pady=20)
        self.frame1.pack(fill=tk.X)
        self.frame2 = tk.Frame(root, pady=20)
        self.frame2.pack(fill=tk.X)
        self.frame3 = tk.Frame(root, pady=20)
        self.frame3.pack(fill=tk.X)

        # Image upload buttons
        self.upload_btn1 = tk.Button(self.frame1, text="Upload Human Image", command=self.upload_image1)
        self.upload_btn1.pack(side=tk.LEFT, padx=10)
        self.upload_btn2 = tk.Button(self.frame2, text="Upload Star Image", command=self.upload_image2)
        self.upload_btn2.pack(side=tk.LEFT, padx=10)

        # Placeholders for uploaded images
        self.human_image_path = None
        self.star_image_path = None
        
        # Image labels for preview
        self.label_img1 = tk.Label(self.frame1)
        self.label_img1.pack(side=tk.RIGHT, padx=10)
        self.label_img2 = tk.Label(self.frame2)
        self.label_img2.pack(side=tk.RIGHT, padx=10)

        # Resultant image label
        self.label_result = tk.Label(self.frame3, text="Resultant Image will appear here")
        self.label_result.pack(padx=10, pady=10)

        # Merge button
        self.merge_btn = tk.Button(root, text="Merge Images", command=self.merge_images)
        self.merge_btn.pack(pady=20)

    def upload_image1(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            self.human_image_path = file_path
            img = Image.open(file_path)
            img = img.resize((100, 100))
            img = ImageTk.PhotoImage(img)
            self.label_img1.config(image=img, text="")
            self.label_img1.image = img

    def upload_image2(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            self.star_image_path = file_path
            img = Image.open(file_path)
            img = img.resize((100, 100))
            img = ImageTk.PhotoImage(img)
            self.label_img2.config(image=img, text="")
            self.label_img2.image = img

    def merge_images(self):
        if self.human_image_path and self.star_image_path:
            img1 = np.array(Image.open(self.human_image_path))
            img2 = np.array(Image.open(self.star_image_path).resize(img1.size))
            
            #img1 = np.array(self.label_img1.image)
            #img2 = np.array(self.label_img2.image)
            
            human_image_path = self.human_image_path
            star_image_path = self.star_image_path

            # Initialize MediaPipe Holistic
            mp_holistic = mp.solutions.holistic
            holistic = mp_holistic.Holistic()

            # Load and process the image
            image_path = human_image_path
            image = img1
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = holistic.process(image_rgb)

            # Extract x and y coordinates of pose landmarks
            pose_landmarks = results.pose_landmarks

            if pose_landmarks:
                x_coords, y_coords = [], []
                for idx, landmark in enumerate(pose_landmarks.landmark):
                    x = int(landmark.x * image.shape[1])  # Scale x-coordinate to image width
                    y = int(landmark.y * image.shape[0])  # Scale y-coordinate to image height
                    x_coords.append(x)
                    y_coords.append(y)


                # Define the skeleton connections (indices of landmarks to be connected)
                skeleton_connections = [(0, 5), (0, 2), (5, 8), (2, 7), (10, 9), (11, 12), (12, 24), (11, 13), (13, 15), (15, 17),
                                        (17, 19), (19, 15), (15, 21), (11, 23),
                                        (12, 14), (14, 16), (16, 18), (18, 20), (20, 16), (16, 22), (23, 24), (23, 25), (25, 27), (27, 29)
                                        , (29, 31), (31, 27), (24, 26),
                                       (26, 28), (28, 30), (30, 32), (32, 28)]


            # Release resources
            holistic.close()


            # Load the image
            image = img2  # Replace with the path to your image

            # Resize the image to a lower resolution for faster processing
            scale_percent = 50  # Change this value as needed (50 means 50% of the original size)
            width = int(image.shape[1] * scale_percent / 100)
            height = int(image.shape[0] * scale_percent / 100)
            resized_image = cv2.resize(image, (width, height))

            # Convert the resized image to grayscale
            grayscale = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)

            # Define the color range for non-black pixels
            lower_bound = np.array([50, 50, 50], dtype=np.uint8)  # Define the lower bound for non-black colors
            upper_bound = np.array([255, 255, 255], dtype=np.uint8)  # Define the upper bound for non-black colors

            # Create a binary mask for non-black pixels
            mask = cv2.inRange(resized_image, lower_bound, upper_bound)

            # Find the coordinates of the center of non-black regions
            non_black_centers = []
            for y in range(mask.shape[0] - 1):
                for x in range(mask.shape[1] - 1):
                    block = mask[y:y + 1, x:x + 1]
                    if np.mean(block) > 0:
                        center_x = x
                        center_y = y
                        non_black_centers.append((center_x, center_y))


            #print(non_black_centers)

            starlist = non_black_centers

            humanlist = []
            for i in range (len(x_coords)):
                humanlist.append((x_coords[i]/12, y_coords[i]/12))

            special_points = []
            for i in range (len(humanlist)):
                minimum =[]
                for j in range (len(starlist)):
                    a = abs(humanlist[i][0]-starlist[j][0])
                    b = abs(humanlist[i][1]-starlist[j][1])
                    minimum.append(a+b)
                min_index = minimum.index(min(minimum))
                special_points.append(starlist[min_index])

            # Load the original image
            original_image = img2  # Replace with the path to your image

            # Resize the image to a lower resolution for faster processing
            scale_percent = 50  # Change this value as needed (50 means 50% of the original size)
            width = int(original_image.shape[1] * scale_percent / 100)
            height = int(original_image.shape[0] * scale_percent / 100)
            resized_image = cv2.resize(original_image, (width, height))

            # Define specific coordinates (normalized to the processed image)
            specific_coordinates_normalized = special_points

            # Scale back the specific coordinates to match the original image's scale
            specific_coordinates_original = []
            for (x, y) in specific_coordinates_normalized:
                x_original = int(x * (original_image.shape[1] / resized_image.shape[1]))
                y_original = int(y * (original_image.shape[0] / resized_image.shape[0]))
                specific_coordinates_original.append((x_original, y_original))

            # Define the skeleton connections (indices of landmarks to be connected)
            skeleton_connections = [(0, 5), (0, 2), (5, 8), (2, 7), (10, 9), (11, 12), (12, 24), (11, 13), (13, 15), (15, 17),
                                    (17, 19), (19, 15), (15, 21), (11, 23), (12, 14), (14, 16), (16, 18), (18, 20),
                                    (20, 16), (16, 22), (23, 24), (23, 25), (25, 27), (27, 29), (29, 31), (31, 27),
                                    (24, 26), (26, 28), (28, 30), (30, 32), (32, 28)]

            # Display the original image
            plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))

            # Display the specific points in red and skeleton connections in green
            for connection in skeleton_connections:
                plt.plot([specific_coordinates_original[connection[0]][0], specific_coordinates_original[connection[1]][0]],
                         [specific_coordinates_original[connection[0]][1], specific_coordinates_original[connection[1]][1]],
                         color='grey', linewidth=2)



            # Capture the figure content into a NumPy array
            fig = plt.gcf()
            fig.canvas.draw()
            img_array = np.array(fig.canvas.renderer._renderer)

            # Close the plt figure to free up memory
            plt.close()

            # Now, `img_array` is a NumPy array representing the final image

            import requests
            from PIL import Image, ImageEnhance
            from io import BytesIO


            def remove_bg_and_get_image(image_path, api_key):
                api_url = "https://api.remove.bg/v1.0/removebg"

                with open(image_path, 'rb') as image_file:
                    response = requests.post(
                        api_url,
                        headers={'X-Api-Key': api_key},
                        files={"image_file": image_file},
                        data={"size": "auto"}
                    )
                response.raise_for_status()

                # Convert the response content to an Image object
                result_image = Image.open(BytesIO(response.content))
                return result_image

            api_key = '2eFZ4nwmSBb878cUxU5CV44h'  # Replace with your API key
            input_image_path = human_image_path  # Replace with your image path

            segmented_image = remove_bg_and_get_image(input_image_path, api_key)

            def adjust_transparency(image, alpha_factor):
                if image.mode != 'RGBA':
                    image = image.convert('RGBA')
                r, g, b, alpha = image.split()
                alpha = ImageEnhance.Brightness(alpha).enhance(alpha_factor)
                image.putalpha(alpha)
                return image

            def resize_to_fit(img, base_shape):
                if isinstance(img, Image.Image):  # Check if img is a PIL Image
                    aspect = img.width / img.height
                else:  # It's a numpy ndarray
                    aspect = img.shape[1] / img.shape[0]

                base_aspect = base_shape[1] / base_shape[0]

                if aspect > base_aspect:
                    new_width = base_shape[1]
                    new_height = int(new_width / aspect)
                else:
                    new_height = base_shape[0]
                    new_width = int(new_height * aspect)

                if isinstance(img, Image.Image):
                    return img.resize((new_width, new_height), Image.ANTIALIAS)
                else:  # Convert ndarray back to PIL Image for resizing and return ndarray
                    return np.array(Image.fromarray(img).resize((new_width, new_height), Image.ANTIALIAS))

            def overlay_images(base_img_array, overlay_img_pil, shift_left=0, shift_up=0):
                # Convert overlay_img_pil to numpy array
                overlay_img_array = np.array(overlay_img_pil)

                # Determine the position to place overlay_img_array on base_img_array
                y1, y2 = max(0, shift_up), min(base_img_array.shape[0], overlay_img_array.shape[0] + shift_up)
                x1, x2 = max(0, shift_left), min(base_img_array.shape[1], overlay_img_array.shape[1] + shift_left)

                overlay_area_base = base_img_array[y1:y2, x1:x2]
                overlay_area_overlay = overlay_img_array[max(0, -shift_up):min(overlay_img_array.shape[0], base_img_array.shape[0] - shift_up),
                                                         max(0, -shift_left):min(overlay_img_array.shape[1], base_img_array.shape[1] - shift_left)]

                # Check if there's an alpha channel in the overlay image
                if overlay_area_overlay.shape[2] == 4:
                    alpha = overlay_area_overlay[:, :, 3] / 255.0
                    for channel in range(3):
                        base_img_array[y1:y2, x1:x2, channel] = (1 - alpha) * overlay_area_base[:, :, channel] + alpha * overlay_area_overlay[:, :, channel]
                else:
                    base_img_array[y1:y2, x1:x2] = overlay_area_overlay

                return base_img_array

            mp_holistic = mp.solutions.holistic
            holistic = mp_holistic.Holistic()

            # Convert the PIL Image to a NumPy array
            image = np.array(segmented_image)

            # Convert the image color from BGR to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Load and process the image
            #image_path = human_image_path
            #image = segmented_image
            #image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = holistic.process(image_rgb)

            # Extract x and y coordinates of pose landmarks
            pose_landmarks = results.pose_landmarks

            if pose_landmarks:
                x_coords_segmented, y_coords_segmented = [], []
                for idx, landmark in enumerate(pose_landmarks.landmark):
                    x = int(landmark.x * image.shape[1])  # Scale x-coordinate to image width
                    y = int(landmark.y * image.shape[0])  # Scale y-coordinate to image height
                    x_coords_segmented.append(x)
                    y_coords_segmented.append(y)


                # Define the skeleton connections (indices of landmarks to be connected)
                skeleton_connections = [(0, 5), (0, 2), (5, 8), (2, 7), (10, 9), (11, 12), (12, 24), (11, 13), (13, 15), (15, 17),
                                        (17, 19), (19, 15), (15, 21), (11, 23),
                                        (12, 14), (14, 16), (16, 18), (18, 20), (20, 16), (16, 22), (23, 24), (23, 25), (25, 27), (27, 29)
                                        , (29, 31), (31, 27), (24, 26),
                                       (26, 28), (28, 30), (30, 32), (32, 28)]

            # Release resources
            holistic.close()

            c1 = sum(x_coords_segmented) / len(x_coords_segmented)
            c2 = sum(y_coords_segmented) / len(y_coords_segmented)

            x_normalized = []
            y_normalized = []
            for i in range (len(x_coords_segmented)):
                x = int(x_coords_segmented[i]/1.8-40)
                y = int(y_coords_segmented[i]/1.8-40)
                x_normalized.append(x)
                y_normalized.append(y)

            c=abs(y_normalized[32]-y_normalized[1])
            d=abs(specific_coordinates_original[32][1]-specific_coordinates_original[1][1])
            resized_parameter = (1/((d/c)*1.8))
            print ("resized_parameter", resized_parameter)

            e = sum(x_normalized) / len(x_normalized)
            f = sum(y_normalized) / len(y_normalized)

            if isinstance(transparent_image, Image.Image):  # Check if it's a PIL Image
                ref_point_x, ref_point_y = transparent_image.size
            elif isinstance(transparent_image, np.ndarray):  # Check if it's a numpy array
                ref_point_y, ref_point_x = transparent_image.shape[:2]
            else:
                raise ValueError("transparent_image should either be a PIL Image or a numpy array.")


            x_reference_scale = abs((ref_point_x-e)/ref_point_x)/4
            y_reference_scale = abs((ref_point_y-f)/ref_point_y)/4
            print ("x_reference_scale", x_reference_scale)

            x_original_list = []
            y_original_list = []
            for i in range (len(specific_coordinates_original)):
                x_original_list.append(specific_coordinates_original[i][0])
                y_original_list.append(specific_coordinates_original[i][1])

            g = sum(x_original_list) / len(x_original_list)
            h = sum(y_original_list) / len(y_original_list)

            x_shift=int(((e-g)/resized_parameter+40)*1.8)
            y_shift=int(((f-h)/resized_parameter+40)*1.8)

            copy_segmented_image = segmented_image.copy()
            transparent_image = adjust_transparency(copy_segmented_image, 0.5)

            copy_img_array = img_array.copy()

            def overlay_images(base_img, overlay_img, shift_left=0, shift_up=0):
                # Copy the base image to avoid in-place modifications
                result_img = base_img.copy()

                # Determine the position to place overlay_img on base_img
                y1, y2 = max(0, shift_up), min(base_img.shape[0], overlay_img.shape[0] + shift_up)
                x1, x2 = max(0, shift_left), min(base_img.shape[1], overlay_img.shape[1] + shift_left)

                # Calculate the overlay area in both base and overlay images
                overlay_area_base = result_img[y1:y2, x1:x2]
                overlay_area_overlay = overlay_img[max(0, -shift_up):min(overlay_img.shape[0], base_img.shape[0] - shift_up),
                                                   max(0, -shift_left):min(overlay_img.shape[1], base_img.shape[1] - shift_left)]

                # Check if there's an alpha channel in the overlay image
                if overlay_area_overlay.shape[2] == 4:
                    alpha = overlay_area_overlay[:, :, 3] / 255.0
                    for channel in range(3):
                        result_img[y1:y2, x1:x2, channel] = (1 - alpha) * overlay_area_base[:, :, channel] + alpha * overlay_area_overlay[:, :, channel]
                else:
                    result_img[y1:y2, x1:x2] = overlay_area_overlay

                return result_img

            # Assuming transparent_image is loaded using PIL, convert it to a NumPy array
            transparent_image = np.asarray(transparent_image)

            # Ensure you have defined or loaded all the required variables, such as:
            # transparent_image, resized_parameter, x_reference_scale, y_reference_scale, x_shift, y_shift, img_array

            # Scale down the transparent_image by the resizing parameter
            scaled_width = int(transparent_image.shape[1] * resized_parameter)
            scaled_height = int(transparent_image.shape[0] * resized_parameter)
            scaled_transparent_image = cv2.resize(transparent_image, (scaled_width, scaled_height))

            # Define the reference point as 10% above the center
            ref_point_x = c1
            ref_point_y = c2

            # Adjust shift values to keep the reference point the same after scaling
            adjusted_shift_left = int(-x_shift) + int(ref_point_x - scaled_width * (ref_point_x / transparent_image.shape[1]))
            adjusted_shift_up = int(-y_shift) + int(ref_point_y - scaled_height * (ref_point_y / transparent_image.shape[0]))

            # Overlay the scaled image onto the original img_array with adjusted shift values
            resultant_image = overlay_images(copy_img_array, scaled_transparent_image, shift_left=int(adjusted_shift_left), shift_up=int(adjusted_shift_up))

            # Display the overlaid image
            plt.figure(figsize=(10, 10))
            plt.axis('on')
            plt.imshow(resultant_image)
            plt.show()

            # This is a simple example of merging, you can customize as needed
            resultant_image = resultant_image
            
            resultant_image_tk = ImageTk.PhotoImage(resultant_image)
            self.label_result.config(image=resultant_image_tk, text="")
            self.label_result.image = resultant_image_tk

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageMergerApp(root)
    root.mainloop()

