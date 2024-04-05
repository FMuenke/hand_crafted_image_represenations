import cv2
import numpy as np

def compute_edge_intensity_orientation(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply the Sobel operator to compute gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

    # Compute edge intensity and orientation
    edge_intensity = np.sqrt(sobelx**2 + sobely**2)
    edge_orientation = np.arctan2(sobely, sobelx)

    return edge_intensity, edge_orientation

def create_hog_descriptor(image, num_bins=9, cell_size=8, block_size=2):
    # Compute edge intensity and orientation using the previous function
    edge_intensity, edge_orientation = compute_edge_intensity_orientation(image)
    
    # Compute the number of cells in the x and y directions
    num_cells_x = image.shape[1] // cell_size
    num_cells_y = image.shape[0] // cell_size
    
    # Initialize the histogram of oriented gradients
    hog_descriptor = np.zeros((num_cells_y, num_cells_x, num_bins))
    
    # Iterate over each cell
    for y in range(num_cells_y):
        for x in range(num_cells_x):
            # Get the edge intensity and orientation for the current cell
            cell_intensity = edge_intensity[y*cell_size:(y+1)*cell_size, x*cell_size:(x+1)*cell_size]
            cell_orientation = edge_orientation[y*cell_size:(y+1)*cell_size, x*cell_size:(x+1)*cell_size]
            
            # Compute the histogram for the current cell
            histogram = np.histogram(cell_orientation, bins=num_bins, range=(-np.pi, np.pi), weights=cell_intensity)[0]
            
            # Normalize the histogram
            histogram /= np.sum(histogram)
            
            # Store the histogram in the HOG descriptor
            hog_descriptor[y, x] = histogram
    
    # Normalize the HOG descriptor
    hog_descriptor /= np.sqrt(np.sum(hog_descriptor**2) + 1e-6)
    
    return hog_descriptor