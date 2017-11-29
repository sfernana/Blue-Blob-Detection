#!/usr/bin/env python

""" Blue Blob Detection """

# Standard libraries
import os
import sys
import argparse
import logging

# "Computer vison" libraries 
import numpy as np
import cv2

def create_voronoi_diagram(image, subdiv_2D, voronoi_color) :
    """Create and print a voronoi diagram on
    the image to process.

    :param image: The image to print the Voronoi diagram
    :type image: cv2.imread()

    :param subdiv_2D: The object containing centers for creating Voronoi diagram
    :type subdiv_2D: cv2.subdiv2D

    :return: The polygones's coordinates
    :rtype: np.Array

    """

    facets, centers = subdiv_2D.getVoronoiFacetList([])

    for facet in facets:
        facet_to_int = np.array([facet], np.int)
        cv2.polylines(image, facet_to_int, True, voronoi_color, 1, 8, 0)

    return facets 

def detect_blue_blobs(image, connectivity, detected_blob_color):
    """Detect blue blobs and return their centers for the Voronoi diagram.

    :param image: The image to detect the blue blobs
    :type image: cv2.imread()

    :return: Centers and Container of centers
    :rtype: np.Array, cv2.Subdiv2D 

    """

    blue_channel = image.copy()
    blue_channel[:,:,1] = 0
    blue_channel[:,:,2] = 0

    gray_image = cv2.cvtColor(blue_channel, cv2.COLOR_BGR2GRAY)

    (thresh, bw_image) = cv2.threshold(gray_image, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    # Rectangle for Subdiv2D
    image_width = image.shape[0]
    image_height = image.shape[1]
    rect = (0, 0, image_height, image_width)
        
    # Create a rectangle of type Subdiv2D for Voronoi
    subdiv_2D = cv2.Subdiv2D(rect)

    output = cv2.connectedComponentsWithStats(bw_image, connectivity, cv2.CV_32S)

    # The fourth cell is the centroid matrix
    centroids = output[3]

    centers_to_int = [ ( int(p[0]), int(p[1]) ) for p in centroids]

    # Fill the container of centers
    for center in centers_to_int :
        # Insert points into subdiv
        subdiv_2D.insert(center)
        # Draw colored filled circles on the centers
        cv2.circle( image, center, 3, detected_blob_color, cv2.FILLED, 8, 0 )

    return centroids, subdiv_2D

def process(images,  output_folder_temp, output_folder, connectivity, detected_blob_color, voronoi_color):
    """The process of the image consists on :
    1) Blue Blob detection
    2) Voronoi Diagram creation over these detected blobs

    :param images: The images to process
    :type image: cv2.imread()

    :return: Centers and polygones's coordinates of all images
    :rtype: list of np.Array

    """

    total_facets = []

    for image in images:
        image_filename = os.path.realpath(image)
        if not os.path.exists(image_filename):
            print("The image '%s' does not exist, skipping to the next one..." % image_filename)
            continue

        print("Image '%s' processing..." % image)
        
        image_data = cv2.imread(image)

        # Determine center of blobs
        centroids, subdiv_2D = detect_blue_blobs(image_data, connectivity, detected_blob_color)

        # Create Voronoi diagram given the centers
        facets = create_voronoi_diagram(image_data, subdiv_2D, voronoi_color)

        # Show image
        #cv2.imshow("Blue Blob detection", image_data)
        #cv2.waitKey(0)
        
        # Write processed image
        filename, file_extension = os.path.splitext(image_filename)
        processed_image = "%s_processed%s" % (filename, file_extension)

        final_output_folder = os.path.dirname(image_filename) if not output_folder_temp else output_folder

        cv2.imwrite(os.path.join(final_output_folder, os.path.basename(processed_image)), image_data)

        print("Image '%s' processing successfull" % image)

        total_facets.append(facets)

    print("Image(s) processing finished")
    return total_facets

def main(images=[], output_folder="", connectivity=0, detected_blob_color=None, voronoi_color=None):
    
    parser = argparse.ArgumentParser()

    parser.add_argument("-i", action="store",dest="images",nargs='*',type=str,default=[],help="The images to process ie. -i image_1 image_2 ...")
    parser.add_argument("-o", action="store",dest="output",default="",help="The output folder of processed images.")
    parser.add_argument("--connect", action="store",dest="connectivity", default=8, help="Neighborhood size")
    parser.add_argument("--colorb", action="store",dest="blob_color",nargs=3,type=str,default=(255, 0, 0),help="The color of the detected blob (in RGB)")
    parser.add_argument("--colorv", action="store",dest="voronoi_color",nargs=3,type=str,default=(255, 255, 255),help="The color of the voronoi diagram (in RGB)")
        
    args = parser.parse_args()
 
    images = args.images if not images else images.split(' ')
    output_folder_temp = args.output if not output_folder else output_folder
    output_folder = os.path.realpath(output_folder_temp)
    
    connectivity = int(args.connectivity) if not connectivity else connectivity 
    detected_blob_color = tuple(map(int, args.blob_color)) if not detected_blob_color else detected_blob_color
    voronoi_color = tuple(map(int, args.voronoi_color)) if not voronoi_color else voronoi_color

    if not images:
        print("No images have been provided !")
        quit()

    if not os.path.exists(output_folder):
        print("The output folder '%s' for processed images does not exist !" % output_folder)
        quit()

    facets = process(images, output_folder_temp, output_folder, connectivity, detected_blob_color, voronoi_color)
    return facets

if __name__ == '__main__':
    main()