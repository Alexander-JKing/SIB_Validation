"""sib_main.sib_sub.sib_validation_imports
========
Sub operations imported and called into sib_validation_main.
"""

import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.geometry import Polygon, Point
from shapely.geometry.multipolygon import MultiPolygon
from shapely.geometry.multipoint import MultiPoint


def confidence_filter(model_detected_points, conf_filter=10):

    """A confidence filter function automatically called during svm.preprocessing.

        This function takes the input value as the percentile at which to find the cutoff point for confidence
        values in the model_detected_points GeoDataFrame.

        Parameters
        ----------
        conf_filter : int
                      This is an integer value which determines the percentile cutoff point for the confidence distribution
                      of the model_detected_points.

        model_detected_points : GeoDataFrame
                                These points must be imported after having been converted from YOLO txt coordinates,
                                to a shapefile format.

        Returns
        -------
        float
             Returns a float value representing the percentile cutoff point of whatever parameter was entered.
    """

    conf = model_detected_points['confidence'].to_numpy()
    percentile = np.percentile(conf, conf_filter)

    remaining_percentage = 100.0 - conf_filter

    print("{}% of values above = {}".format(remaining_percentage, percentile))

    percentile_float = float(percentile)

    return percentile_float


def geometry_check(gdf):

    """A geometry check automatically called during svm.preprocessing.

        This function checks that the geometry type is POINT and not MULTIPOINT for a specified geometry.


        Parameters
        ----------
        gdf : GeoDataFrame
              These points can either be the ground_truth_points or model_detected_points.

        Returns
        -------
        GeoDataFrame
                    Returns the GeoDataFrame that was passed into the function parameters, unaltered.
    """

    geometry_type = gdf.geom_type

    if geometry_type[0] == 'Point':
        print("Accepted Geometries: 'POINT'")
    elif geometry_type[0] == 'MultiPoint':
        print("Invalid Geometry type: 'MULTIPOINT' - Must enter a 'POINT' Geometry")
        return None

    return gdf


def multi_to_single(gdf):

    """A MultiPoint geometry type to a Point geometry type converter.

        This function converts all MultiPoint geometries into Point geometries by way of creating centroids,
        then swapping the geometry column with the new centroid-geometry column.

        Parameters
        ----------
        gdf : GeoDataFrame
              These points can either be the ground_truth_points or model_detected_points.

        Returns
        -------
        GeoDataFrame
                    Returns the GeoDataFrame that was passed into the function parameters with a Point geometry type.
    """

    gdf_centroid = gdf.geometry.centroid

    gdf = gdf.drop('geometry', axis=1)

    gdf['geometry'] = gdf_centroid

    return gdf


def single_to_multi(gdf, geometry_type='Point'):

    """A single geometry type to a multi-geometry type converter.

        This function converts all Point or Polygon geometries into MultiPoint or MultiPolygon geometries, by way
        of swapping the geometries during a for loop in a list comprehension.

        Parameters
        ----------
        gdf : GeoDataFrame
              These points can either be the ground_truth_points or model_detected_points.

        geometry_type : str
                        This string represents the geometry type to change to Multi. It can be either 'Point' or 'Polygon'.
                        The default is 'Point'.

        Returns
        -------
        GeoDataFrame
                    Returns the GeoDataFrame that was passed into the function parameters with a multi-geometry type.
    """

    # https://gis.stackexchange.com/questions/311320/casting-geometry-to-multi-using-geopandas

    if geometry_type == 'Point':
        gdf["geometry"] = [MultiPoint([feature]) if type(feature) == Point
                           else feature for feature in gdf["geometry"]]

    elif geometry_type == 'Polygon':
        gdf["geometry"] = [MultiPolygon([feature]) if type(feature) == Polygon
                           else feature for feature in gdf["geometry"]]

    else:
        print("Invalid arguments")

    return gdf


def create_buffer(point_gdf, buffer_size=2):

    """A Buffer Generator.

        This function creates a buffer of a given radius around an input Point geometry..

        Parameters
        ----------
        point_gdf : GeoDataFrame
                    These points can either be the ground_truth_points or model_detected_points.

        buffer_size : int
                      This int value specifies the size of the buffer to be created. Default is 2m

        Returns
        -------
        GeoDataFrame
                    Returns a new buffer GeoDataFrame of the points passed in the parameter.
    """

    # Create a 2m buffer
    point_gdf = point_gdf.to_crs("epsg:29902")
    buffer_geometry = point_gdf['geometry'].buffer(buffer_size)

    # Assemble a GeoDataFrame with that buffer geometry
    buffer_df = gpd.GeoDataFrame(point_gdf, geometry=buffer_geometry)

    return buffer_df


def intersection_count(geometry_a, geometry_b):

    """Counts the number of intersections between two geometries.

        This function calculates the number of times that geometry_a intersects geometry_b.
        geometry_a must be a Polygon geometry, for example a buffer.

        Parameters
        ----------
        geometry_a : GeoDataFrame
                     This must be a Polygon geometry.

        geometry_b : GeoDataFrame
                     This can either be a Point, Line or Polygon geometry.

        Returns
        -------
        GeoDataFrame
                    Returns the original geometry_a GeoDataFrame with the number of intersections as a new column.
    """

    intersections = lambda x: np.sum(geometry_b.intersects(x))
    n = geometry_a['geometry'].apply(intersections)
    geometry_a['n'] = n

    return geometry_a


def drop_indices(points_gdf, outcome_gdf, outcome_value='true'):

    """Drops rows in a DataFrame.

        This function extracts the indices of a DataFrame based on a condition, then returns those indices in the form of a list.
        The condition is dependent on the outcome_gdf passed, which will either be the 'true positives' or the 'false positives' DataFrame.
        The default outcome_value is set to 'true', but can be modified to 'false'.

        Parameters
        ----------
        points_gdf : GeoDataFrame
                     This will be the original point GeoDataFrame used to extract the outcome_df.
                     For example, the model_detected_points were used in this validation.

        outcome_gdf : GeoDataFrame
                      This can either be the true_positives GeoDataFrame extracted or the False Positives GeoDataFrame extracted.

        outcome_value : str
                        This is a switch condition which determines whether the function calculates based off of true or false positives.
                        Options are 'true' and 'false'. The default value is 'true'.

        Returns
        -------
        GeoDataFrame
                    Returns the original GeoDataFrame used to calculate the outcome_gdf with the appropriate rows dropped.
    """

    # check for duplicates
    try:
        outcome_gdf = outcome_gdf.drop_duplicates(keep='first')
    except Exception:
        pass

    if outcome_value == 'true':
        try:
            outcome_gdf = outcome_gdf[['det_ID', 'TP']]
            points_gdf = points_gdf.merge(outcome_gdf, how='left', on='det_ID')
            index_list = points_gdf.index[points_gdf['TP'] == 1.0].tolist()
            points_gdf = points_gdf.drop(index_list)
            points_gdf = points_gdf[['det_class', 'det_ID', 'geometry']]
            print("Length of 'model_det' after TP rows dropped: {}\n".format(len(points_gdf)))
        except KeyError:
            print("No more TP points to detect within 2m")

    elif outcome_value == 'false':
        try:
            outcome_gdf = outcome_gdf[['det_ID', 'FP']]
            points_gdf = points_gdf.merge(outcome_gdf, how='left', on='det_ID')
            index_list = points_gdf.index[points_gdf['FP'] == 1.0].tolist()
            points_gdf = points_gdf.drop(index_list)
            points_gdf = points_gdf[['det_class', 'det_ID', 'geometry']]
            print("Length of 'model_det' after FP rows dropped: {}\n".format(len(points_gdf)))
        except KeyError:
            print("No more FP points to detect within 2m")

    elif outcome_value == 'ground':
        
        if 'TP' in outcome_gdf.columns:
            try:
                outcome_gdf = outcome_gdf[['gt_ID', 'TP']]
                points_gdf = points_gdf.merge(outcome_gdf, how='left', on='gt_ID')
                index_list = points_gdf.index[points_gdf['TP'] == 1.0].tolist()
                points_gdf = points_gdf.drop(index_list)
                points_gdf = points_gdf[['class', 'gt_ID', 'geometry']]
                print("Length of 'Ground Truth points' after TP rows dropped: {}\n".format(len(points_gdf)))
            except KeyError:
                print("No more TP points within 2m of 'ground_truth'")
        
        else:
            try:
                outcome_gdf = outcome_gdf[['gt_ID', 'FP']]
                points_gdf = points_gdf.merge(outcome_gdf, how='left', on='gt_ID')
                index_list = points_gdf.index[points_gdf['FP'] == 1.0].tolist()
                points_gdf = points_gdf.drop(index_list)
                points_gdf = points_gdf[['class', 'gt_ID', 'geometry']]
                print("Length of 'Ground Truth points' after FP rows dropped: {}\n".format(len(points_gdf)))
            except KeyError:
                print("No more TP points within 2m of 'ground_truth'")
            
    else:
        print('Invalid argument given')

    return points_gdf


def extract_points_outside_2m(ground_truth_points, model_detected_points):

    """Extracts all False Positives outside of 2m from the ground_truth_points.

        This function extracts the False Positives outside of a 2m Buffer from the ground_truth_points.
        It does this by performing a spatial join with a ground_truth_point buffer. The model_detected_points will be
        the left GeoDataFrame. Duplicates found are dropped and any NaN values for the joined GeoDataFrame are extracted
        as the False Positives outside.

        Parameters
        ----------
        ground_truth_points : GeoDataFrame
                              These points must be imported after being manually created.
                              They represent the actual ground truth class, as verified by the user.

        model_detected_points : GeoDataFrame
                                These points must be imported after having been converted from YOLO txt coordinates,
                                to a shapefile format.

        Returns
        -------
        GeoDataFrame
                    Returns a Point GeoDataFrame of all False Positives located 2m outside of the ground_truth_points
    """

    gt_buffer = create_buffer(ground_truth_points, buffer_size=2)
    model_detected_points = gpd.sjoin(model_detected_points, gt_buffer, how="left", predicate='intersects')
    model_detected_points = model_detected_points.drop_duplicates(subset='det_ID')
    model_detected_points['gt_ID'].fillna('outside', inplace=True)
    fp2 = model_detected_points.loc[model_detected_points.gt_ID == 'outside']
    fp2 = fp2[['det_class', 'det_ID', 'geometry']]
    fp2['FP'] = 1

    return fp2


def n_max(ground_truth_points, model_detected_points):

    """Finds the maximum number of points located within a 2m buffer of ground_truth_points.

        This function counts the number of points that intersect a buffer geometry, in this case a ground_truth_buffer.
        It returns the maximum count found so that an end iteration for a for loop can be assigned.

        Parameters
        ----------
        ground_truth_points : GeoDataFrame
                                  These points must be imported after being manually created.
                                  They represent the actual ground truth class, as verified by the user.

        model_detected_points : GeoDataFrame
                                    These points must be imported after having been converted from YOLO txt coordinates,
                                    to a shapefile format.

        Returns
        -------
        int
            Returns an integer  of all False Positives located 2m outside of the ground_truth_points
    """

    # Create a 2m buffer
    gt_buffer = create_buffer(ground_truth_points, buffer_size=2)

    # Call the intersection_count on the buffer
    count = intersection_count(gt_buffer, model_detected_points)

    # Get the index elements of the buffer GeoDataFrame in a list.
    gt_buffer_index = gt_buffer.index

    # Find the max number of model_detected_points within a buffer
    condition = np.max(gt_buffer['n'])
    max_within_buffer = gt_buffer_index[condition]

    return max_within_buffer


def create_global_GeoDataFrame(global_list):

    """Turns a list of dictionaries into a GeoDataFrame.

        This function takes in a list of Dictionaries originating from either a True Positive or False Positive DataFrame
        It constructs a Pandas DataFrame from this and then returns a GeoPandas GeoDataFrame.

        Parameters
        ----------
        global_list : list
                      This list contains a list of dictionaries of True Positive or False Positive DataFrame rows.

        Returns
        -------
        GeoDataFrame
                    Returns an GeoDataFrame of all True Positives or False Positives.
    """

    global_df = pd.concat(pd.DataFrame(row) for row in global_list)
    global_df_geometry = global_df.geometry

    global_GeoDF = gpd.GeoDataFrame(global_df, geometry=global_df_geometry, crs="EPSG:29902")
    global_GeoDF = global_GeoDF.reset_index(drop=True)

    return global_GeoDF


def tp_first_pass(nearest_neighbour):

    """Turns a list of dictionaries into a GeoDataFrame.

        This function creates and extracts a unique column from the result of the call_calculate_nearest function.
        It returns this column in the form of a DataFrame, consisting of all the records validated as a True Positive on the previous pass
        and their corresponding ID values stored in the 'gt_ID' column..

        Parameters
        ----------
        nearest_neighbour : GeoDataFrame
                            This list contains a list of dictionaries of True Positive or False Positive DataFrame rows.

        Returns
        -------
        GeoDataFrame
                    Returns the original GeoDataFrame for nearest_neighbour with all the rows containing True Positives marked.

        DataFrame
                 Returns a DataFrame containing all the ground_truth_points ID's with a True Positive identified on the first pass.
    """

    nearest_neighbour['contains_TP'] = 'NO'
    contains_tp = nearest_neighbour.loc[nearest_neighbour['TP'] == 1.0]
    contains_tp = contains_tp["contains_TP"].replace({"NO": "YES"})
    nearest_neighbour.update(contains_tp)
    contains_tp = nearest_neighbour[['gt_ID', 'contains_TP']]

    return nearest_neighbour, contains_tp

