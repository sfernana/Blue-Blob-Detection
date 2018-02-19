#!/usr/bin/env python

""" Blue Blob Detection

This module provides detection of blue blobs.
Only the blobs have to be blue so it cannot be used for a general blob detector.

In fact, these blue blobs represent centers of cells of an organic tissue.

The aim is to characterize these cells in order to know their diameter etc...

For example a Voroinoi diagram can be drawn over these blobs in order to represent the cells themselves.

Once, the detection done, the results will be given to a simulation model.

"""

# Standard libraries
import os
import sys
import argparse
import math
import csv
import time

# "Computer Vision" libraries
import numpy as np
import cv2
import matplotlib.pyplot as plt

def compute_optimal_neighborhood_depth(centroids):
    """Compute the optimal depth for the neighborhood

    :param centroids: The centers of the cells
    :type centroids: list

    :return: The optimal depth
    :rtype: int

    """

    distances = []

    for centroid in centroids:
        a = centroid
        b = get_closest_point(centroids, a)

        distance = math.sqrt(pow(a[0]-b[0],2) + pow(a[1]-b[1],2))
        distances.append(distance)

    mean_neighbor_distance = int(round(np.mean(distances)))

    return mean_neighbor_distance

def compute_polygone_area(polygon):
    """Compute the area of a polygon given its points

    :param polygone: The points of the polygone
    :type filename: list

    :return: The area of the polygone
    :rtype: float

    """

    area = 0
    for index, point in enumerate(polygon):
        if index < len(polygon) - 1:
           area += (point[0] * polygon[index + 1][1]) - (point[1] * polygon[index + 1][0])
        else:
           area += (point[0] * polygon[0][1]) - (point[1] * polygon[0][0])

    return math.fabs(area/2.0)

def compute_polygone_from_neighborhood(image, centroid, flooding_neighborhood_points, depth, image_height, image_width):
    """Find contours of a neighborhood which gives a "complex" polygone.

    :param image: The image to compute the neighborhood on
    :type image: cv2.imread()

    :param centroids: The centers of cells
    :type centroids: list

    :param flooding_neighborhood_points: The points of the neighborhood
    :type flooding_neighborhood_points: list

    :param depth: The depth of the neighborhood
    :type depth: int

    :param image_height: The height of the image
    :type image_height: int

    :param image_width: The width of the image
    :type image_width: int

    :return: The polygone's coordinates
    :rtype: np.Array

    """

    flooding_neighborhood = np.zeros((image_height, image_width), dtype=int)

    for point in flooding_neighborhood_points:
        x = point[0]
        y = point[1]
        flooding_neighborhood[x][y] = 1

    x_center = centroid[0]
    y_center = centroid[1]

    # starts and stops of the search of contours of the neighborhood matrix

    x_marge_before = x_center - depth
    x_marge_before = x_marge_before if x_marge_before >= 0 else 0

    x_marge_after = x_center + depth
    x_marge_after = x_marge_after if x_marge_after < flooding_neighborhood.shape[1] else flooding_neighborhood.shape[1] - 1

    y_marge_before = y_center - depth
    y_marge_before = y_marge_before if y_marge_before >= 0 else 0

    y_marge_after = y_center + depth
    y_marge_after = y_marge_after if y_marge_after < flooding_neighborhood.shape[0] else flooding_neighborhood.shape[0] - 1


    top = ['horizontal', range(x_marge_before, x_marge_after + 1), range(y_marge_before, y_center)]
    right = ['vertical', range(y_marge_before, y_marge_after + 1), range(x_marge_after, x_center, -1) ]
    bottom = ['horizontal', range(x_marge_after, x_marge_before - 1, -1), range(y_marge_after, y_center, -1)]
    left = ['vertical', range(y_marge_after, y_marge_before - 1, -1), range(x_marge_before, x_center)]

    sides = [top, right, bottom, left]
    polygone = []

    for side in sides:
        scan_type = side[0]
        horizontal_scan = side[1]
        vertical_scan = side[2]
        for a in horizontal_scan:
            for b in vertical_scan:
                if scan_type == 'horizontal':
                    x = a
                    y = b
                else:
                    x = b
                    y = a
                if flooding_neighborhood[y][x] == 1 :
                    point = (y,x)
                    if point not in polygone:
                        polygone.append([y, x])
                        # Draw neighborhood on the image
                        image[y][x] = [0,0,0]
                    break;

    return polygone

def convert_pixel_area_to_micrometer_area(pixel_area):
    """Convert an area in pixel into an area in micrometer

    :param pixel_area: The area in pixel
    :type pixel_area: int

    :return: The area in micrometer
    :rtype: float

    """

    # Scale given in the image
    scale = 2000
    # Length of the scale in pixels
    start_pixel = 3422
    end_pixel = 3746

    scale_to_pixel = end_pixel - start_pixel

    micrometer_area = math.sqrt((pixel_area * pow(scale, 2) / pow(scale_to_pixel, 2)))

    return micrometer_area

def create_flooding_neighborhood(image, centroids):
    """Create a neighborhood depending on neighbor's reservation and
    RGB value of the neighbor.

    The size of the neighborhood depends on the depth.
    e.g A depth of 1 gives a neighborhood of 8.

    :param image: The image to compute the neighborhood on
    :type image: cv2.imread()

    :param centroids: The centers of cells
    :type centroids: list

    :return: The polygones's coordinates
    :rtype: np.Array

    """

    image_height = image.shape[0]
    image_width = image.shape[1]

    # RGB thresholds
    # Shifts used to find a "good" threshold
    shift_blue = 0
    shift_red = 0

    # Min value of blue over all pixels #12
    blue_threshold = np.min(image[:,:,0]) + shift_blue

    # Min value of red over all pixels #30
    red_threshold = np.min(image[:,:,2]) + shift_red

    # Distance from the center to the farest neighbor
    neighborhood_depth = compute_optimal_neighborhood_depth(centroids)

    # A depth value of 1 will give these distances : 3 (8 neighbors)
    # A depth value of 5 will give these distances : 3, 5, 7, 9, 11 (more neighbors)
    neighborhood_ranges = range(3, 100, 2)[:neighborhood_depth]

    neighborhoods = [ np.ones((size,size), dtype=int) for size in neighborhood_ranges ]
    # e.g neighborhoods = [ np.ones((3,3))]
    # neighborhood1= np.array([[1,1,1],
    #                        [1,0,1],
    #                        [1,1,1]])

    # List of pixels reserved by cells during the flooding
    reserved_neighbors = []
    neighborhood_of_centroids = {}

    # Blue and red values of neighbors
    blues = []
    reds = []

    # Iterate over each neighborhood
    for neighborhood_index, neighborhood in enumerate(neighborhoods):

        # To have zeros inside
        neighborhood[1:-1,1:-1] = 0
        # The maximum number of neighbors
        neighborhood_size = len(zip(*np.where(neighborhood == 1)))

        # Iterate over each neighbor
        for neighbor_index in range(neighborhood_size):

            # For each centroid, look for neighbors of neighborhood
            for centroid in centroids:

                x_center = centroid[0]
                y_center = centroid[1]

                centroid = (x_center, y_center)

                # Each cell contains a neighborhood in these dictionaries instanciated once
                if centroid not in neighborhood_of_centroids:
                    neighborhood_of_centroids[centroid] = {'full_neighborhood_points':{}, 'flooding_neighborhood_points': [] }

                if neighborhood_index not in neighborhood_of_centroids[centroid]['full_neighborhood_points']:
                    neighborhood_of_centroids[centroid]['full_neighborhood_points'][neighborhood_index] = []

                    # Truncate neighborhood for cells in extremas

                    height_start = 0
                    height_end = neighborhood.shape[0]

                    width_start = 0
                    width_end = neighborhood.shape[1]

                    neighbor_distance_height_before = y_center  - (neighborhood_index + 1)

                    if neighbor_distance_height_before < 0:
                        height_start = np.abs(neighbor_distance_height_before)

                    neighbor_distance_height_after = y_center  + (neighborhood_index + 1)

                    if neighbor_distance_height_after > image_height - 1:
                        height_end = neighborhood.shape[0] - (neighbor_distance_height_after - (image_height - 1))


                    neighbor_distance_width_before = x_center - (neighborhood_index + 1)

                    if neighbor_distance_width_before < 0:
                        width_start = np.abs(neighbor_distance_width_before)

                    neighbor_distance_width_after = x_center  + (neighborhood_index + 1)

                    if neighbor_distance_width_after > image_width - 1:
                        width_end = neighborhood.shape[1] - (neighbor_distance_width_after - (image_width - 1))

                    # Truncated neighborhood
                    neighborhood_cut = neighborhood[height_start:height_end,width_start:width_end]

                    # Reshape the (truncated or not) neighborhood to the size of the image
                    # in order to obtain absolute coordinates of neighbors
                    height_before = neighbor_distance_height_before
                    height_before = height_before if height_before >= 0 else 0

                    height_after = image_height - y_center - (neighborhood_index + 2)
                    height_after = height_after if height_after >= 0 else 0

                    width_before = neighbor_distance_width_before
                    width_before = width_before if width_before >= 0 else 0

                    width_after = image_width - x_center - (neighborhood_index + 2)
                    width_after = width_after if width_after >= 0 else 0

                    # Reshaping: 0s padding
                    neighborhood_resized = np.pad(neighborhood_cut,((height_before, height_after), (width_before, width_after)) ,mode='constant', constant_values=0)

                    neighborhood_points = zip(*np.where(neighborhood_resized == 1))

                    neighborhood_of_centroids[centroid]['full_neighborhood_points'][neighborhood_index] = neighborhood_points

                neighborhood_points = neighborhood_of_centroids[centroid]['full_neighborhood_points'][neighborhood_index]

                # Maximum number of neighbors may not correspond to the 'real' neighborhood due to extremas !
                if neighbor_index < len(neighborhood_points):
                    neighbor = neighborhood_points[neighbor_index]

                    x = neighbor[0]
                    y = neighbor[1]

                    # Get RB value of the neighbor
                    blue = image[x , y][0]
                    red = image[x , y][2]

                    # Analyse distribution of colors
                    blues.append(blue)
                    reds.append(red)

                    # A neighbor is choosen if it is not reserved and if it has a sufficient RGB threshold
                    if (blue > blue_threshold and red > red_threshold) and neighbor not in reserved_neighbors:
                        neighborhood_of_centroids[centroid]['flooding_neighborhood_points'].append(neighbor)
                        reserved_neighbors.append(neighbor)

    # Once the neighborhood has been computed for each centroid, we try to find the contours
    # which will be our polygone
    facets = []
    for centroid in centroids:
        centroid = (centroid[0], centroid[1])
        flooding_neighborhood_points = neighborhood_of_centroids[centroid]['flooding_neighborhood_points']
        facet = compute_polygone_from_neighborhood(image, centroid, flooding_neighborhood_points, neighborhood_depth, image_height, image_width)
        facet_to_rint = np.array(np.rint(facet), np.int)
        if (facet_to_rint>=0.0).all():
            facets.append(facet)

    return facets

def create_voronoi_diagram(image, subdiv_2D, voronoi_color) :
    """Create and print a Voronoi diagram on
    the image to process.

    :param image: The image to print the Voronoi diagram on
    :type image: cv2.imread()

    :param subdiv_2D: The object containing centers for creating the Voronoi diagram
    :type subdiv_2D: cv2.subdiv2D

    :param voronoi_color: Color of the Voronoi diagram
    :type voronoi_color: tuple (RGB value)

    :return: The polygones's coordinates
    :rtype: np.Array

    """

    treated_facets=[]

    facets, centers = subdiv_2D.getVoronoiFacetList([])

    for facet in facets:
        facet_to_rint = np.array(np.rint(facet), np.int)
        #facet_to_rint = np.array([facet], np.int)
        if (facet_to_rint>=0.0).all():
            treated_facets.append(facet_to_rint)
            cv2.polylines(image, [facet_to_rint], True, voronoi_color, 1, 8, 0)

    return treated_facets

def create_voronoi_from_file(filename):
    """Create a Voronoi Diagram from points into file

    :param filename: Name of the file containing points
    :type filename: str

    :return: The polygone's coordinates
    :rtype: np.Array

    """

    facets = []
    # Get centroids from file
    with open(filename,'r') as f:
        xs, ys = zip(*csv.reader(f))

    centroids = [(float(x),float(y)) for x,y in zip(xs,ys)]
    centroids = np.array(centroids)
    from scipy.spatial import Voronoi, voronoi_plot_2d

    vor = Voronoi(centroids)

    # Get a Finite Voronoi
    regions, vertices = voronoi_finite_polygons_2d(vor)
    fig = plt.figure()

    # colorize
    for region in regions:
        polygon = vertices[region]
        facets.append(polygon)
        plt.fill(*zip(*polygon), alpha=0.4)

    plt.scatter(centroids[:,0], centroids[:,1])
    plt.axis('equal')
    plt.xlim(vor.min_bound[0] - 0.1, vor.max_bound[0] + 0.1)
    plt.ylim(vor.min_bound[1] - 0.1, vor.max_bound[1] + 0.1)

    plt.show()

    return facets, centroids

def detect_blue_blobs(image, detected_blob_color):
    """Detect blue blobs and return their centers for the Voronoi diagram.

    :param image: The image to detect the blue blobs
    :type image: cv2.imread()

    :param detected_blob_color: Color of the detected blobs
    :type detected_blob_color: tuple (RGB value)

    :return: Centers, Contours, Container of centers and Image area
    :rtype: np.Array, list, cv2.Subdiv2D, int

    """

    # Keep only blue channel
    blue_channel = image
    blue_channel[:,:,1] = 0
    blue_channel[:,:,2] = 0

    # Create gray image for the thresholding
    gray_image = cv2.cvtColor(blue_channel, cv2.COLOR_BGR2GRAY)

    (thresh, bw_image) = cv2.threshold(gray_image, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    # Rectangle for Subdiv2D
    image_width = image.shape[1]
    image_height = image.shape[0]

    image_coords = [[0, 0],[image_width, 0],[image_width, image_height],[0, image_height]]
    image_area = compute_polygone_area(image_coords)

    rect = (0, 0, image_width, image_height)
        
    # Create a rectangle of type Subdiv2D for Voronoi
    subdiv_2D = cv2.Subdiv2D(rect)

    contours = []

    # Contour Detection
    print("Contour Detection method in progress...")
    image_cnt, contours, hierarchy = cv2.findContours(bw_image,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    centroids = []

    for contour in contours:
        M = cv2.moments(contour)
        cx = cy = 0
        if M['m00']:
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            centroids.append([cx,cy])


    print("%s blue blobs have been detected" % len(centroids))
    centers_to_int = [ ( int(p[0]), int(p[1]) ) for p in centroids]

    # Fill the container of centers
    for center in centers_to_int :
        # Insert points into subdiv
        subdiv_2D.insert(center)
        # Draw colored filled circles on the centers
        cv2.circle( image, center, 3, detected_blob_color, cv2.FILLED, 8, 0 )
    #return centroids, contours, subdiv_2D, image_area
    return centroids, contours, subdiv_2D, image_area

def get_closest_point(points, coord):
    """Return the closest point between a point and a list of points

    :param points: The list of point
    :type points: list

    :param coord: The point which we find the closest
    :type coord: tuple

    :return: The closest point
    :rtype: tuple

    """

    # List of (dist, point) tuples
    dists = [(pow(point[0] - coord[0], 2) + pow(point[1] - coord[1], 2), point)
              for point in points if point!= coord]
    nearest = min(dists)
    # Return point only and not the distance
    return nearest[1]

def voronoi_finite_polygons_2d(vor, radius=None):
    """
    Reconstruct infinite voronoi regions in a 2D diagram to finite
    regions.
    Parameters
    ----------
    vor : Voronoi
        Input diagram
    radius : float, optional
        Distance to 'points at infinity'.
    Returns
    -------
    regions : list of tuples
        Indices of vertices in each revised Voronoi regions.
    vertices : list of tuples
        Coordinates for revised Voronoi vertices. Same as coordinates
        of input vertices, with 'points at infinity' appended to the
        end.
    """

    if vor.points.shape[1] != 2:
        raise ValueError("Requires 2D input")

    new_regions = []
    new_vertices = vor.vertices.tolist()

    center = vor.points.mean(axis=0)
    if radius is None:
        radius = vor.points.ptp().max()*2

    # Construct a map containing all ridges for a given point
    all_ridges = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        all_ridges.setdefault(p1, []).append((p2, v1, v2))
        all_ridges.setdefault(p2, []).append((p1, v1, v2))

    # Reconstruct infinite regions
    for p1, region in enumerate(vor.point_region):
        vertices = vor.regions[region]

        if all(v >= 0 for v in vertices):
            # finite region
            new_regions.append(vertices)
            continue

        # reconstruct a non-finite region
        ridges = all_ridges[p1]
        new_region = [v for v in vertices if v >= 0]

        for p2, v1, v2 in ridges:
            if v2 < 0:
                v1, v2 = v2, v1
            if v1 >= 0:
                # finite ridge: already in the region
                continue

            # Compute the missing endpoint of an infinite ridge

            t = vor.points[p2] - vor.points[p1] # tangent
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])  # normal

            midpoint = vor.points[[p1, p2]].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, n)) * n
            far_point = vor.vertices[v2] + direction * radius

            new_region.append(len(new_vertices))
            new_vertices.append(far_point.tolist())

        # sort region counterclockwise
        vs = np.asarray([new_vertices[v] for v in new_region])
        c = vs.mean(axis=0)
        angles = np.arctan2(vs[:,1] - c[1], vs[:,0] - c[0])
        new_region = np.array(new_region)[np.argsort(angles)]

        # finish
        new_regions.append(new_region.tolist())

    return new_regions, np.asarray(new_vertices)

def plot_cell_area_distribution(data_to_dist):
    """Plot distribution from data

    :param data_to_dist: Data for the distribution
    :type data_to_dist: list

    """

    fig = plt.figure()

    #n, bins, patches = plt.hist(data_to_dist, bins=np.linspace(0,1000,100))
    plt.hist(data_to_dist, bins=np.linspace(0,0.000000001,100))
    plt.xticks(np.arange(0.0000000001,0.0000000007,0.00000000005))
    #plt.hist(data_to_dist, bins=np.linspace(3.0,9.0,20))
    #plt.xticks(np.arange(3.0,9.0,0.5))
    #print np.sum(n)
    plt.show()

    fig.savefig('CellAreaDistribution.png')

def main(images=[], output_folder="", detected_blob_color=None, voronoi_color=None):
    """Main function defined for the Matlab call of python script
    It checks arguments and run process() method.

    cf process.__doc__

    """

    parser = argparse.ArgumentParser()

    parser.add_argument("-i", action="store",dest="images",nargs='*',type=str,default=[],help="The images to process ie. -i image_1 image_2 ...")
    parser.add_argument("-o", action="store",dest="output",default="",help="The output folder of processed images.")
    parser.add_argument("--colorb", action="store",dest="blob_color",nargs=3,type=str,default=(255, 0, 0),help="The color of the detected blob (in RGB)")
    parser.add_argument("--colorv", action="store",dest="voronoi_color",nargs=3,type=str,default=(255, 255, 255),help="The color of the voronoi diagram (in RGB)")
        
    args = parser.parse_args()
 
    images = args.images if not images else images.split(' ')

    output_folder_temp = args.output if not output_folder else output_folder
    output_folder = os.path.realpath(output_folder_temp)
    
    detected_blob_color = tuple(map(int, args.blob_color)) if not detected_blob_color else detected_blob_color
    voronoi_color = tuple(map(int, args.voronoi_color)) if not voronoi_color else voronoi_color

    if not images:
        print("No images have been provided !")
        return [], []

    if not os.path.exists(output_folder):
        print("The output folder '%s' for processed images does not exist !" % output_folder)
        return [], []

    sorted_facets = process(images, output_folder_temp, output_folder, detected_blob_color, voronoi_color)

    return sorted_facets

def process(images, output_folder_temp, output_folder, detected_blob_color, voronoi_color):
    """The process of the image consists on :
    1) Detecting Blue Blobs which represent centers of the cells (nucleous ?)
    2) Characterizing the cells

    :param images: The images to process
    :type images: cv2.imread()

    :param output_folder_temp: The User output folder of the processed images
    :type output_folder_temp: str

    :param output_folder: The Default output folder of the processed images
    :type output_folder: str

    :param detected_blob_color: Color of the detected blobs
    :type detected_blob_color: tuple (RGB value)

    :param voronoi_color: Color of the Voronoi diagram
    :type voronoi_color: tuple (RGB value)

    :return: Centers and polygones's coordinates of all images
    :rtype: list of np.Array

    """

    total_facets = []
    total_centroids = []

    for image_filename in images:
        image_realpath = os.path.realpath(image_filename)
        if not os.path.exists(image_realpath):
            print("The image '%s' does not exist, skipping to the next one..." % image_filename)
            continue

        print("Image '%s' processing..." % image_filename)

        image = cv2.imread(image_realpath)

        contours = []

        subdiv_2D = []

        image_to_treat = image.copy()

        # Determine center of blobs from image
        centroids, contours, subdiv_2D, image_area = detect_blue_blobs(image_to_treat, detected_blob_color)

        start = time.time()
        facets = create_flooding_neighborhood(image, centroids)
        #facets = create_voronoi_diagram(image, subdiv_2D, voronoi_color)
        #facets, centroids = create_voronoi_from_file('cellCentroids_50000.txt')
        end = time.time()

        elapsed_time = end - start

        print "Elapsed time is about %s" %elapsed_time


        # Write processed image
        file_name, file_extension = os.path.splitext(image_filename)
        processed_image = "%s_processed%s" % (file_name, file_extension)
        facets_file_text = "%s_processed.csv" % file_name

        final_output_folder = os.path.dirname(image_filename) if not output_folder_temp else output_folder

        cv2.imwrite(os.path.join(final_output_folder, os.path.basename(processed_image)), image)

        print("Image '%s' processing successfull" % image_filename)

        print("Image '%s' facets coordinates counter clockwise sorting processing..." % image_filename)

        sorted_facets = []
        cell_areas = []
        cell_edges = []

        for facet, centroid in zip(facets, centroids):
            # Do not consider cells in borders
            sorted_facet = sorted(facet, key=lambda coordinate:math.atan2(coordinate[0] - centroid[0], coordinate[1] - centroid[1]), reverse=True)
            sorted_facets.append(sorted_facet)
            cell_area = compute_polygone_area(facet)
            if cell_area < 1000:
                cell_areas.append(cell_area)
            if len(facet) < 2:
                edges = 0
            elif len(facet) == 2:
                edges = 1
            else:
                edges = len(facet)

            cell_edges.append(edges)

        cell_area = np.median(cell_areas)
        cell_std = np.std(cell_areas)
        cell_mean = np.mean(cell_areas)
        edges_median = np.median(cell_edges)
        edges_mean = np.mean(cell_edges)
        edges_std = np.std(cell_edges)

        print("Image '%s' facets coordinates counter clockwise sorting successfull" % image_filename)
        print("Mean of cell area is of '%s' micrometer" % convert_pixel_area_to_micrometer_area(cell_mean))
        print("Median of cell area is of '%s' micrometer" % convert_pixel_area_to_micrometer_area(cell_area))
        print("Standard deviation of cell area is of '%s' micrometer" % convert_pixel_area_to_micrometer_area(cell_std))
        print("Mean of cell edges is of '%s'" % edges_mean)
        print("Median of cell edges is of '%s'" % edges_median)
        print("Standard deviation of cell edges is of '%s'" % edges_std)
        print("Image '%s' facets coordinates saving to file processing..." % image_filename)

        plot_cell_area_distribution(cell_areas)

        sorted_facets_to_text = []
        for index, facet in enumerate(sorted_facets):
            suite = ''
            for point in facet:
                suite += ',%s,%s' % (point[0], point[1])

            facet_line = "%s%s" % (index, suite)
            sorted_facets_to_text.append(facet_line)

        with open(facets_file_text, 'w') as text_file:
            text_file.write("\n".join(sorted_facets_to_text))

        print("Image '%s' facets coordinates saving to file successfull" % image)

        total_facets.append(sorted_facets)

    print("Image(s) processing finished")
    return total_facets


if __name__ == '__main__':
    main()
