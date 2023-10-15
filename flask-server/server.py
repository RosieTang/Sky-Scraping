from flask import Flask, request, jsonify
from flask_cors import CORS
import base64
import cv2
import mediapipe as mp
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

human_path = '/Users/tangyunxi/Library/CloudStorage/OneDrive-Personal/文档/WUSTL/activities/HackWashu23/skyscraping/flask-server/test8.jpg'
star_path = '/Users/tangyunxi/Library/CloudStorage/OneDrive-Personal/文档/WUSTL/activities/HackWashu23/skyscraping/flask-server/star1.jpg'

app = Flask(__name__)
CORS(app)  # This will allow requests from any origin by default. Fine for development, but lock this down in production!

@app.route('/upload', methods=['POST'])
def upload_image():
    uploaded_file = request.files.get('photo')
    if not uploaded_file:
        return jsonify({"error": "No file uploaded!"}), 400
    
    #processed_image_path = # 
    process_image(uploaded_file)
    
    # # Convert the processed image to base64
    # with open(processed_image_path, "rb") as img_file:
    #     b64_string = base64.b64encode(img_file.read()).decode()

    # # Return the base64 encoded image
    # return jsonify({'image': b64_string})

def process_image(human_path):
    run_code(human_path, star_path)


if __name__ == '__main__':
    app.run(port=5173)



def run_code(human_image_path, star_image_path):
    # Initialize MediaPipe Holistic
    mp_holistic = mp.solutions.holistic
    holistic = mp_holistic.Holistic()

    # Load and process the image
    image_path = human_image_path
    image = cv2.imread(image_path)
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

        # Visualize pose landmarks and skeleton connections using Matplotlib
        plt.figure(figsize=(8, 6))
        # plt.imshow(image_rgb)
        plt.scatter(x_coords, y_coords, color='red', s=10)  # Use smaller dots for landmarks
        for connection in skeleton_connections:
            plt.plot([x_coords[connection[0]], x_coords[connection[1]]],
                     [y_coords[connection[0]], y_coords[connection[1]]], color='green', linewidth=2)
        plt.title('Pose Landmarks with Skeleton Connections')
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        #plt.gca().invert_yaxis()  # Invert y-axis to match the image coordinates
        plt.grid(True)  # Show grid for better reference
        # plt.show()
    else:
        print("No pose landmarks detected in the image.")

    # Release resources
    holistic.close()
    print (x_coords)
    print (y_coords)

    """# Star Recognition"""

    from matplotlib import pyplot as plt

    # Load the image
    image = cv2.imread(star_image_path)  # Replace with the path to your image

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

    # Display the points on the resized image
    # plt.imshow(cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB))
    for center in non_black_centers:
        plt.scatter(center[0], center[1], c='red', s=1)  # Draw a red dot at each point
    plt.axis('off')  # Hide axis for better visualization
    # plt.show()

    #print(non_black_centers)

    starlist = non_black_centers
    print (starlist[0][0])

    humanlist = []
    for i in range (len(x_coords)):
        humanlist.append((x_coords[i]/12, y_coords[i]/12))
    print (humanlist)

    special_points = []
    for i in range (len(humanlist)):
        minimum =[]
        for j in range (len(starlist)):
            a = abs(humanlist[i][0]-starlist[j][0])
            b = abs(humanlist[i][1]-starlist[j][1])
            minimum.append(a+b)
        min_index = minimum.index(min(minimum))
        special_points.append(starlist[min_index])



    # Load the image
    image = cv2.imread(star_image_path)  # Replace with the path to your image

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

    # Specific coordinates to highlight in different colors
    specific_coordinates = special_points

    # Display the points on the resized image
    # plt.imshow(cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB))
    for center in non_black_centers:
        if center in specific_coordinates:
            plt.scatter(center[0], center[1], c='red', s=1)  # Draw specific points in red
        else:
            plt.scatter(center[0], center[1], c='blue', s=1)  # Draw other points in blue
    plt.axis('off')  # Hide axis for better visualization

    # Define the skeleton connections (indices of landmarks to be connected)
    skeleton_connections = [(0, 5), (0, 2), (5, 8), (2, 7), (10, 9), (11, 12), (12, 24), (11, 13), (13, 15), (15, 17),
                            (17, 19), (19, 15), (15, 21), (11, 23),
                            (12, 14), (14, 16), (16, 18), (18, 20), (20, 16), (16, 22), (23, 24), (23, 25), (25, 27), (27, 29),
                            (29, 31), (31, 27), (24, 26),
                            (26, 28), (28, 30), (30, 32), (32, 28)]

    # Visualize pose landmarks and skeleton connections using Matplotlib
    for connection in skeleton_connections:
        plt.plot([specific_coordinates[connection[0]][0], specific_coordinates[connection[1]][0]],
                 [specific_coordinates[connection[0]][1], specific_coordinates[connection[1]][1]], color='green', linewidth=2)

    # plt.show()



    # Load the original image
    original_image = cv2.imread(star_image_path)  # Replace with the path to your image

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
    # plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))

    # Display the specific points in red and skeleton connections in green
    for connection in skeleton_connections:
        plt.plot([specific_coordinates_original[connection[0]][0], specific_coordinates_original[connection[1]][0]],
                 [specific_coordinates_original[connection[0]][1], specific_coordinates_original[connection[1]][1]],
                 color='grey', linewidth=2)

    #for point in specific_coordinates_original:
        #plt.scatter(point[0], point[1], c='red', s=5)  # Draw specific points in red

    plt.axis('on')  # Hide axis for better visualization
    plt.savefig("output_image.png", bbox_inches='tight', pad_inches=0, dpi=300)


    # Capture the figure content into a NumPy array
    fig = plt.gcf()
    fig.canvas.draw()
    img_array = np.array(fig.canvas.renderer._renderer)

    # Close the plt figure to free up memory
    plt.close()

    # Now, `img_array` is a NumPy array representing the final image

    plt.figure(figsize=(8, 8))  # Adjust the figure size as needed
    # plt.imshow(img_array)
    plt.axis('off')  # To hide the axes
    # plt.show()

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

    def adjust_transparency(image, alpha_factor):
        """
        Adjust the transparency of an image.
        :param image: Image object to adjust.
        :param alpha_factor: Factor to adjust the alpha (transparency).
                             0.0: fully transparent, 1.0: original image
        :return: Adjusted image.
        """
        if image.mode != 'RGBA':
            image = image.convert('RGBA')

        # Extract alpha channel
        r, g, b, alpha = image.split()
        alpha = ImageEnhance.Brightness(alpha).enhance(alpha_factor)

        # Merge back channels
        image.putalpha(alpha)
        return image

    api_key = 'm5HS9uTNULvJcx8EKoKAcudG'  # Replace with your API key (please ensure you're not sharing your actual API key publicly)
    input_image_path = human_image_path  # Replace with your image path

    segmented_image = remove_bg_and_get_image(input_image_path, api_key)

    # Adjust transparency
    transparent_image = adjust_transparency(segmented_image, 0.5)  # 0.5 will make it semi-transparent

    # Display the image in Jupyter Notebook
    # plt.imshow(transparent_image)
    plt.axis('off')
    # plt.show()



    # Function to remove the background and get the image
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
        result_image = Image.open(BytesIO(response.content))
        return result_image

    # Initialize MediaPipe Holistic
    mp_holistic = mp.solutions.holistic
    holistic = mp_holistic.Holistic()

    # Load and process the image using the remove_bg_and_get_image function
    api_key = 'm5HS9uTNULvJcx8EKoKAcudG'
    image_path = human_image_path
    segmented_image = remove_bg_and_get_image(image_path, api_key)

    # Convert PIL Image to numpy array (if it's a PIL image)
    if isinstance(segmented_image, Image.Image):
        image = np.array(segmented_image)
    else:
        image = segmented_image  # Assuming it's already a numpy array

    # Convert RGB to BGR
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    results = holistic.process(image_bgr)

    # Extract x and y coordinates of pose landmarks
    pose_landmarks = results.pose_landmarks
    if pose_landmarks:
        x_coords_segmented, y_coords_segmented = [], []
        for landmark in pose_landmarks.landmark:
            x = int(landmark.x * image.shape[1])
            y = int(landmark.y * image.shape[0])
            x_coords_segmented.append(x)
            y_coords_segmented.append(y)

        # Define the skeleton connections (indices of landmarks to be connected)
        skeleton_connections = [(0, 5), (0, 2), (5, 8), (2, 7), (10, 9), (11, 12), (12, 24), (11, 13), (13, 15), (15, 17),
                                (17, 19), (19, 15), (15, 21), (11, 23),
                                (12, 14), (14, 16), (16, 18), (18, 20), (20, 16), (16, 22), (23, 24), (23, 25), (25, 27), (27, 29)
                                , (29, 31), (31, 27), (24, 26),
                               (26, 28), (28, 30), (30, 32), (32, 28)]  # Continue your list here

        # Visualize pose landmarks and skeleton connections using Matplotlib
        plt.figure(figsize=(8, 6))
        # plt.imshow(image)
        plt.scatter(x_coords_segmented, y_coords_segmented, color='red', s=10)
        for i, (x, y) in enumerate(zip(x_coords_segmented, y_coords_segmented)):
            plt.annotate(str(i), (x, y), fontsize=8, ha="right")
        for connection in skeleton_connections:
            plt.plot([x_coords_segmented[connection[0]], x_coords_segmented[connection[1]]],
                     [y_coords_segmented[connection[0]], y_coords_segmented[connection[1]]], color='green', linewidth=2)
        plt.title('Pose Landmarks with Skeleton Connections')
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        plt.grid(True)
        # plt.show()
    else:
        print("No pose landmarks detected in the image.")

    # Release resources
    holistic.close()

    print (x_coords_segmented[1])
    print (x_coords_segmented[32])
    print (specific_coordinates_original[1][0])
    print (specific_coordinates_original[32][0])

    print (y_coords_segmented[1])
    print (y_coords_segmented[32])
    print (specific_coordinates_original[1][1])
    print (specific_coordinates_original[32][1])

    c=abs(y_coords_segmented[32]-y_coords_segmented[1])
    d=abs(specific_coordinates_original[32][1]-specific_coordinates_original[1][1])
    resized_parameter = d/c+0.2
    print ("resized_parameter", resized_parameter)

    e = sum(x_coords_segmented) / len(x_coords_segmented)
    f = sum(y_coords_segmented) / len(y_coords_segmented)

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

    x_shift=int(e-g)
    y_shift=int(f-h)
    print (x_shift)
    print (y_shift)


    def overlay_images(base_img, overlay_img, shift_left=0, shift_up=0):
        # Copy the base image to avoid in-place modifications
        result_img = base_img.copy()

        # Convert overlay_img to numpy array if it's a PIL Image object
        if isinstance(overlay_img, Image.Image):
            overlay_img = np.array(overlay_img)

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

    # Overlay the transparent image onto the original img_array with specified shift
    resultant_image = overlay_images(img_array, transparent_image, shift_left=int(-x_shift*0.25), shift_up=int(-y_shift*0.25))

    # Display the overlaid image
    plt.figure(figsize=(10, 10))
    plt.axis('on')
    # plt.imshow(resultant_image)
    # plt.show()

    # Determine the center of the image
    if isinstance(transparent_image, Image.Image):  # Check if it's a PIL Image
        ref_point_x, ref_point_y = transparent_image.size
    elif isinstance(transparent_image, np.ndarray):  # Check if it's a numpy array
        ref_point_y, ref_point_x = transparent_image.shape[:2]
    else:
        raise ValueError("transparent_image should either be a PIL Image or a numpy array.")

    ref_point_x /= 2  # center x
    ref_point_y = ref_point_y / 2 - 0.3 * ref_point_y  # a little above the center y

    # Convert transparent_image to numpy array if it's a PIL Image object
    if isinstance(transparent_image, Image.Image):
        transparent_image = np.array(transparent_image)

    # Scale down the image
    scale_factor = resized_parameter
    scaled_transparent_image = cv2.resize(transparent_image, (int(transparent_image.shape[1] * scale_factor), int(transparent_image.shape[0] * scale_factor)))

    # Calculate the shift to keep the reference point the same
    shift_left = ref_point_x - scale_factor * ref_point_x
    shift_up = ref_point_y - scale_factor * ref_point_y

    # Overlay the scaled image onto the original img_array
    resultant_image = overlay_images(img_array, scaled_transparent_image, shift_left=int(shift_left), shift_up=int(shift_up))

    # Display the overlaid image
    plt.figure(figsize=(10, 10))
    plt.axis('on')
    # plt.imshow(resultant_image)
    # plt.show()


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
    ref_point_x = transparent_image.shape[1] // 2 + int(x_reference_scale * transparent_image.shape[1])
    ref_point_y = transparent_image.shape[0] // 2 - int(y_reference_scale * transparent_image.shape[0])

    # Adjust shift values to keep the reference point the same after scaling
    adjusted_shift_left = int(-x_shift*0.25) + int(ref_point_x - scaled_width * (ref_point_x / transparent_image.shape[1]))
    adjusted_shift_up = int(-y_shift*0.25) + int(ref_point_y - scaled_height * (ref_point_y / transparent_image.shape[0]))

    # Overlay the scaled image onto the original img_array with adjusted shift values
    resultant_image = overlay_images(img_array, scaled_transparent_image, shift_left=adjusted_shift_left, shift_up=adjusted_shift_up)

    # Display the overlaid image
    plt.figure(figsize=(10, 10))
    plt.axis('on')
    # plt.imshow(resultant_image)
    # plt.show()


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

    # Scale down the transparent_image by 0.4
    scaled_width = int(transparent_image.shape[1] * resized_parameter)
    scaled_height = int(transparent_image.shape[0] * resized_parameter)
    scaled_transparent_image = cv2.resize(transparent_image, (scaled_width, scaled_height))

    # Define the reference point as 10% above the center
    ref_point_x = g
    ref_point_y = h

    # Adjust shift values to keep the reference point the same after scaling
    adjusted_shift_left = int(-x_shift*0.25) + int(ref_point_x - scaled_width * (ref_point_x / transparent_image.shape[1]))
    adjusted_shift_up = int(-y_shift*0.25) + int(ref_point_y - scaled_height * (ref_point_y / transparent_image.shape[0]))

    # Overlay the scaled image onto the original img_array with adjusted shift values
    resultant_image = overlay_images(img_array, scaled_transparent_image, shift_left=adjusted_shift_left, shift_up=adjusted_shift_up)

    # Display the overlaid image
    plt.figure(figsize=(10, 10))
    plt.axis('on')
    # plt.imshow(resultant_image)
    # plt.show()

    def overlay_images(base_img, overlay_img):
        # Copy the base image to avoid in-place modifications
        result_img = base_img.copy()

        # Determine the top-left position to place overlay_img on base_img
        x_offset = (result_img.shape[1] - overlay_img.shape[1]) // 2
        y_offset = (result_img.shape[0] - overlay_img.shape[0]) // 2

        # Overlay the images
        if overlay_img.shape[2] == 4:  # Check if the overlay image has an alpha channel
            alpha = overlay_img[:, :, 3] / 255.0
            for channel in range(3):
                result_img[y_offset:y_offset+overlay_img.shape[0], x_offset:x_offset+overlay_img.shape[1], channel] = \
                (1 - alpha) * result_img[y_offset:y_offset+overlay_img.shape[0], x_offset:x_offset+overlay_img.shape[1], channel] + \
                alpha * overlay_img[:, :, channel]
        else:
            result_img[y_offset:y_offset+overlay_img.shape[0], x_offset:x_offset+overlay_img.shape[1]] = overlay_img

        return result_img

    # First, resize the transparent_image to 0.44 of its original scale
    resized_transparent_image = cv2.resize(transparent_image, (int(transparent_image.shape[1] * resized_parameter),
                                                        int(transparent_image.shape[0] * resized_parameter)))

    # Overlay the resized transparent image onto the original img_array
    resultant_image = overlay_images(img_array, resized_transparent_image)

    # Display the overlaid image
    plt.figure(figsize=(10, 10))
    # plt.imshow(resultant_image)
    plt.axis('off')
    # plt.show()

    def overlay_images(base_img, overlay_img, shift_left=0, shift_up=0):
        # Copy the base image to avoid in-place modifications
        result_img = base_img.copy()

        # Determine the top-left position to place overlay_img on base_img
        x_offset = (result_img.shape[1] - overlay_img.shape[1]) // 2 - shift_left
        y_offset = (result_img.shape[0] - overlay_img.shape[0]) // 2 - shift_up

        # Check if shifted overlay goes out of bounds and correct if necessary
        if x_offset < 0:
            x_offset = 0
        if y_offset < 0:
            y_offset = 0

        # Overlay the images
        if overlay_img.shape[2] == 4:  # Check if the overlay image has an alpha channel
            alpha = overlay_img[:, :, 3] / 255.0
            for channel in range(3):
                result_img[y_offset:y_offset+overlay_img.shape[0], x_offset:x_offset+overlay_img.shape[1], channel] = \
                (1 - alpha) * result_img[y_offset:y_offset+overlay_img.shape[0], x_offset:x_offset+overlay_img.shape[1], channel] + \
                alpha * overlay_img[:, :, channel]
        else:
            result_img[y_offset:y_offset+overlay_img.shape[0], x_offset:x_offset+overlay_img.shape[1]] = overlay_img

        return result_img

    # First, resize the transparent_image to 0.44 of its original scale
    resized_transparent_image = cv2.resize(transparent_image, (int(transparent_image.shape[1] * resized_parameter),
                                                        int(transparent_image.shape[0] * resized_parameter)))

    # Overlay the resized transparent image onto the original img_array
    resultant_image = overlay_images(img_array, resized_transparent_image, shift_left=x_shift, shift_up=y_shift)

    # Display the overlaid image
    plt.figure(figsize=(10, 10))
    # plt.imshow(resultant_image)
    plt.axis('off')
    # plt.show()
    plt.savefig('Users/tangyunxi/Library/CloudStorage/OneDrive-Personal/文档/WUSTL/activities/HackWashu23/skyscraping/photo-upload-server/returns', format='jpg', dpi=1200)