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
from itertools import combinations

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

def get_diameter(pixel_area):
    """Convert an area in pixel into diameter in micrometer

    :param pixel_area: The area in pixel
    :type pixel_area: int

    :return: The diameter in micrometer
    :rtype: float

    """

    # Scale given in the image
    scale = 2000
    # Length of the scale in pixels
    start_pixel = 3422
    end_pixel = 3746

    scale_to_pixel = end_pixel - start_pixel

    #micrometer_area = math.sqrt((pixel_area * pow(scale, 2) / pow(scale_to_pixel, 2)))

    pixel_in_micrometer = scale / scale_to_pixel
    rayon = math.sqrt( (pixel_area * pow(pixel_in_micrometer, 2)/ math.pi ) )
    diameter = 2 * rayon

    return diameter

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

    image_width = image.shape[1]
    image_height = image.shape[0]

    # RGB thresholds
    # Shifts used to find "good" thresholds
    shift_blue = 0
    shift_red = 0

    # Min value of blue over all pixels #12
    blue_threshold = np.min(image[:,:,0]) + shift_blue

    # Min value of red over all pixels #30
    red_threshold = np.min(image[:,:,2]) + shift_red

    # Distance from the center to the farest neighbor
    neighborhood_depth = compute_optimal_neighborhood_depth(centroids)
    #neighborhood_depth = 2

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
    points_belongs_to = {}

    # Blue and red values of neighbors
    blues = []
    reds = []

    # Iterate over each neighborhood
    for neighborhood_index, neighborhood in enumerate(neighborhoods):
        # To have zeros inside
        neighborhood[1:-1,1:-1] = 0

        neighborhood_full_points = zip(*np.where(neighborhood == 1))

        # Iterate over each neighbor
        for relative_neighbor in neighborhood_full_points:

            # For each centroid, look for neighbors of neighborhood
            for centroid in centroids:

                x_center = centroid[0]
                y_center = centroid[1]

                centroid = (x_center, y_center)

                # Each cell contains a neighborhood in these dictionaries instanciated once
                if centroid not in neighborhood_of_centroids:
                    neighborhood_of_centroids[centroid] = {'mine':[], 'commons':{}}

                neighbor_distance_width_before = x_center - (neighborhood_index + 1)

                neighbor_distance_height_before = y_center  - (neighborhood_index + 1)

                # neighbor's coordinates in neighborhood matrix
                relative_neighbor_x = relative_neighbor[0]
                relative_neighbor_y = relative_neighbor[1]

                # neighbor's coordinates in image
                absolute_neighbor_x = relative_neighbor_x + neighbor_distance_width_before
                absolute_neighbor_y = relative_neighbor_y + neighbor_distance_height_before

                # Be carefull of cells located at extremas !
                if (absolute_neighbor_x < (image_width - 1) and absolute_neighbor_x >=0 ) and (absolute_neighbor_y < (image_height -1) and absolute_neighbor_y >= 0):

                    absolute_neighbor = (absolute_neighbor_x, absolute_neighbor_y)
                    #print image_width, image_height

                    #print absolute_neighbor
                    # Get RB value of the neighbor
                    #print absolute_neighbor_x, image_width, image_height
                    blue = image[absolute_neighbor_y , absolute_neighbor_x][0]
                    red = image[absolute_neighbor_y , absolute_neighbor_x][2]

                    # Analyse distribution of colors
                    #blues.append(blue)
                    #reds.append(red)

                    # A neighbor is choosen if it is not reserved and if it has a sufficient RGB threshold
                    if (blue > blue_threshold and red > red_threshold):

                        if absolute_neighbor in reserved_neighbors:
                            # Each centroid know which points it has in common
                            # And each point know which cell it belongs to (see later: points_belongs_to)
                            centroid_owner = points_belongs_to[absolute_neighbor]

                            if centroid_owner not in neighborhood_of_centroids[centroid]['commons']:
                                neighborhood_of_centroids[centroid]['commons'][centroid_owner] = []

                            neighborhood_of_centroids[centroid]['commons'][centroid_owner].append(absolute_neighbor)

                        else:

                            # Compute contour of the cell by deleting inside/predecessor neighborhood
                            # and keep the greatest one, which will result in a polygone
                            # The first neighborhood does not have predecessor (there is only centroid in its center)
                            if neighborhood_index > 0:
                                predecessor_relative_neighbor_x = -1
                                predecessor_relative_neighbor_y = -1

                                # We take predecessor of certain neighbors
                                # e.g
                                # We have this (depth = 2):
                                # *****
                                # *****
                                # **0**
                                # *****
                                # *****
                                # Here we look only predecessors of -:
                                # *---*
                                # -***-
                                # -*0*-
                                # -***-
                                # *---*
                                # We delete them and the result is:
                                # *****
                                # *   *
                                # * 0 *
                                # *   *
                                # *****

                                if relative_neighbor_x > 0 and relative_neighbor_x < (neighborhood_index + 1)*2 and relative_neighbor_y == 0:
                                    predecessor_relative_neighbor_x = relative_neighbor_x
                                    predecessor_relative_neighbor_y = relative_neighbor_y + 1
                                elif relative_neighbor_y > 0 and relative_neighbor_y < (neighborhood_index + 1)*2 and relative_neighbor_x == 0:
                                    predecessor_relative_neighbor_x = relative_neighbor_x + 1
                                    predecessor_relative_neighbor_y = relative_neighbor_y
                                elif relative_neighbor_x > 0 and relative_neighbor_x < (neighborhood_index + 1)*2 and relative_neighbor_y == (neighborhood_index + 1)*2:
                                    predecessor_relative_neighbor_x = relative_neighbor_x
                                    predecessor_relative_neighbor_y = relative_neighbor_y - 1
                                elif relative_neighbor_y > 0 and relative_neighbor_y < (neighborhood_index + 1)*2 and relative_neighbor_x == (neighborhood_index + 1)*2:
                                    predecessor_relative_neighbor_x = relative_neighbor_x - 1
                                    predecessor_relative_neighbor_y = relative_neighbor_y

                                if predecessor_relative_neighbor_x >= 0 and predecessor_relative_neighbor_y >= 0:
                                    predecessor_absolute_neighbor_x = predecessor_relative_neighbor_x + neighbor_distance_width_before
                                    predecessor_absolute_neighbor_y = predecessor_relative_neighbor_y + neighbor_distance_height_before

                                    predecessor_absolute_neighbor = (predecessor_absolute_neighbor_x, predecessor_absolute_neighbor_y)

                                    if predecessor_absolute_neighbor in neighborhood_of_centroids[centroid]['mine']:
                                        predecessor_absolute_neighbor_index = neighborhood_of_centroids[centroid]['mine'].index(predecessor_absolute_neighbor)
                                        del neighborhood_of_centroids[centroid]['mine'][predecessor_absolute_neighbor_index]
                                        #image[predecessor_absolute_neighbor_y][predecessor_absolute_neighbor_x] = [255,255,255]

                            #neighborhood_of_centroids[centroid][relative_neighbor] = absolute_neighbor
                            neighborhood_of_centroids[centroid]['mine'].append(absolute_neighbor)

                            #image[absolute_neighbor_y][absolute_neighbor_x] = [0,0,0]

                            reserved_neighbors.append(absolute_neighbor)

                            # For each point, we know which cell it belongs to
                            points_belongs_to[absolute_neighbor] = centroid


    #neighborhood_of_centroids = {(2,4):{'mine':[(2,5),(3,6),(4,7),(100,100)], 'commons':{(12,14):[(13,15), (12,16),(30,100)]}},(12,14):{'mine':[(13,15),(12,16),(13,12),(30,100)], 'commons':{(2,4):[(3,6),(4,7),(100,100)]}} }
    neighborhood_of_centroids = put_facets_in_common(neighborhood_of_centroids)
    # result =
    #{(2, 4): {'mine': [[2, 5], [3, 6], [100, 100]], 'commons': {(12, 14): [[13, 15], [12, 16], [30, 100]]}}, (12, 14): {'mine': [[13, 12], [3, 6], [100, 100]], 'commons': {}}}

    facets = []
    for centroid in neighborhood_of_centroids:
        facet = neighborhood_of_centroids[centroid]['mine']
        #facet_to_rint = np.array(np.rint(facet), np.int)
        #if (facet_to_rint>=0.0).all():
            #facets.append(facet)
        facets.append(facet)
    return facets

def square_distance(x,y):
    """Compute the square distance between points.

    :param x: The x coordinates
    :type x: int

    :param y: The y coordinates
    :type y: int

    :return: The square distance
    :rtype: int

    """

    return sum([(xi-yi)**2 for xi, yi in zip(x,y)])

def get_farthest_points(edges):
    """Given some edges by a list of points,
    The function finds the farthest points and return one edge (2 points).
    Like that, it minimizes the number of edges of contours.

    :param edges: The point of the edges
    :type edges: list

    :return: The farthest points
    :rtype: list

    """

    # Find farthest(2) points in a list of at least 3 points
    if len(edges) < 3:
        return edges

    A = np.array(edges)
    max_square_distance = 0

    for pair in combinations(A,2):
        distance = square_distance(*pair)
        if distance > max_square_distance:
            max_square_distance = distance
            max_pair = [tuple(pair[0]), tuple(pair[1])]

    return max_pair

def put_facets_in_common(neighborhood_of_centroids):
    """The flooding finds contours but they have
    no edges in common.

    This function use 'history' parameter 'commons' and
    put in common contours by deleting points of the second
    visited contour and add to it common points of the first visited contour.

    e.g
    Suppose we have 2 centroids with their contour delimited by 'mine' points
    and the points they have in common defined in 'commons':
    C1:mine:ABCDE        C2:mine:FGHIJ
      :commons:GHI         :commons:BCD

    At the end we would like to have:
    C1:mine:ABCDE        C2:mine:FBCDJ
      :commons:GHI         :commons:BCD

    And if we want to keep only one edge in common between the 2 centroids,
    we would like to have:

    C1:mine:AB*DE        C2:mine:FB*DJ
      :commons:G*I         :commons:B*D

    We remove intermediate points (make diagram to understand)

    :param neighborhood_of_centroids: The centroids and their neighborhood
    :type neighborhood_of_centroids: dict

    :return: The centroids and their neighborhood put in common
    :rtype: dict


    """

    for centroid in neighborhood_of_centroids:

        # We get centroid's info like points it has in common
        centroid_info = neighborhood_of_centroids[centroid]
        # Points in common
        centroid_commons = centroid_info['commons']
        # Own points
        centroid_mine = centroid_info['mine']

        # Look points in common
        for centroid_owner in centroid_commons:
            # Get the owner of the point

            for centroid_common in centroid_commons[centroid_owner]:

                # We delete this common point in the owner's mine list
                if centroid_common in neighborhood_of_centroids[centroid_owner]['mine']:
                    common_index = neighborhood_of_centroids[centroid_owner]['mine'].index(centroid_common)
                    del neighborhood_of_centroids[centroid_owner]['mine'][common_index]

            # Delete intermediate points to keep only one edge before adding them to 2nd visited contour
            if centroid in neighborhood_of_centroids[centroid_owner]['commons']:
                centroid_owner_commons = neighborhood_of_centroids[centroid_owner]['commons'][centroid]
                centroid_owner_commons_minimized = get_farthest_points(centroid_owner_commons)
                neighborhood_of_centroids[centroid_owner]['mine'] += centroid_owner_commons_minimized

                # Intermediate points deleted have to be removed also from 1st visited contour (update)
                centroid_mines_to_delete = list(set(map(tuple, centroid_owner_commons)).symmetric_difference(set(map(tuple, centroid_owner_commons_minimized))))

                for centroid_mine_to_delete in centroid_mines_to_delete:
                    if centroid_mine_to_delete in neighborhood_of_centroids[centroid]['mine']:
                        mine_index = neighborhood_of_centroids[centroid]['mine'].index(centroid_mine_to_delete)
                        del neighborhood_of_centroids[centroid]['mine'][mine_index]

                # Because 2nd visited contour will be visited, do not check it commons and make 2 times the same job
                # of 'commonization'
                del neighborhood_of_centroids[centroid_owner]['commons'][centroid]


            # Like that we have 2 centroids now with common points
            # We choose the ones of the centroid we start

    return neighborhood_of_centroids


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

    #areas
    #plt.hist(data_to_dist, bins=np.linspace(0,0.000000001,100))
    #plt.xticks(np.arange(0.0000000001,0.0000000007,0.00000000005))

    #diameter
    plt.hist(data_to_dist, bins=np.linspace(0,100,10))
    plt.xticks(np.arange(0,100,10))

    fig.savefig('CellDiameterDistribution.png')

    plt.clf()

    #edges
    plt.hist(data_to_dist, bins=np.linspace(3.0,30.0,20))
    plt.xticks(np.arange(3.0,30.0,2))

    #edges
    #plt.hist(data_to_dist, bins=np.linspace(3.0,9.0,20))
    #plt.xticks(np.arange(3.0,9.0,0.5))
    #print np.sum(n)
    #plt.show()

    fig.savefig('CellEdgeDistribution.png')

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
        cell_diameters = []
        cell_edges = []

        for facet, centroid in zip(facets, centroids):
            # Do not consider cells in borders
            sorted_facet = sorted(facet, key=lambda coordinate:math.atan2(coordinate[0] - centroid[0], coordinate[1] - centroid[1]), reverse=True)
            sorted_facets.append(sorted_facet)
            cell_area = compute_polygone_area(facet)
            cell_diameter = get_diameter(cell_area)
            cell_diameters.append(cell_diameter)
            if cell_area < 1000:
                cell_areas.append(cell_area)
            if len(facet) < 2:
                edges = 0
            elif len(facet) == 2:
                edges = 1
            else:
                edges = len(facet)

            cell_edges.append(edges)

        # areas
        cell_area_median = np.median(cell_areas)
        cell_area_std = np.std(cell_areas)
        cell_area_mean = np.mean(cell_areas)

        # diameters
        cell_diameter_median = np.median(cell_diameters)
        cell_diameter_std = np.std(cell_diameters)
        cell_diameter_mean = np.mean(cell_diameters)

        # edges
        cell_edges_median = np.median(cell_edges)
        cell_edges_mean = np.mean(cell_edges)
        cell_edges_std = np.std(cell_edges)

        print("Image '%s' facets coordinates counter clockwise sorting successfull" % image_filename)
        print("Mean of cell diameter is of '%s' micrometer" % cell_diameter_mean)
        print("Median of cell diameter is of '%s' micrometer" % cell_diameter_median)
        print("Standard deviation of cell diameter is of '%s' micrometer" % cell_diameter_std)
        print("Mean of cell edges is of '%s'" % cell_edges_mean)
        print("Median of cell edges is of '%s'" % cell_edges_median)
        print("Standard deviation of cell edges is of '%s'" % cell_edges_std)
        print("Image '%s' facets coordinates saving to file processing..." % image_filename)

        plot_cell_area_distribution(cell_diameters)

        sorted_facets_to_text = []
        for index, facet in enumerate(sorted_facets):
            suite = ''
            for point in facet:
                suite += ',%s,%s' % (point[0], point[1])

            facet_line = "%s%s" % (index, suite)
            sorted_facets_to_text.append(facet_line)

        with open(facets_file_text, 'w') as text_file:
            text_file.write("\n".join(sorted_facets_to_text))

        print("Image '%s' facets coordinates saving to file successfull" % image_filename)

        total_facets.append(sorted_facets)

    print("Image(s) processing finished")
    return total_facets


if __name__ == '__main__':
    main()
