"""sib_main.sib_sub.sib_validation_main
=========
Main operations for the SIB validation process.
"""

from sib_main.sib_sub.sib_validation_imports import *
import pandas as pd
import geopandas as gpd
import numpy as np
from shapely.ops import nearest_points
import copy

IRENET95 = "epsg:29902"


def preprocessing(ground_truth_points, model_detected_points, conf_percentile=10):

    """Preprocessing GeoDataFrame before analysis.

    This function checks that the following criteria are met in the GeoDataFrame:
        a) checks geometry type
        b) filters the confidence values by a chosen percentile
        c) verifies the correct crs
        d) establishes a unique ID for each GeoDataFrame
        e) renames the relevant columns to a standard format
        f) drops any unrelated columns and attributes

    Parameters
    ----------
    ground_truth_points : GeoDataFrame
                          These points must be imported after being manually created.
                          They represent the actual ground truth class, as verified by the user.

    model_detected_points : GeoDataFrame
                            These points must be imported after having been converted from YOLO txt coordinates,
                            to a shapefile format.

    conf_percentile : int
                      This is an integer value which determines the percentile cutoff point for the confidence distribution
                      of the model_detected_points.

    Returns
    -------
    GeoDataFrame
                Returns the ground_truth_points preprocessed

    GeoDataFrame
                Returns the model_detected_points preprocessed
    """

    # Call the 'geometry_check' function on the ground_truth_points and the model_detected_points.
    ground_truth_points = geometry_check(ground_truth_points)
    model_detected_points = geometry_check(model_detected_points)

    # Call the confidence_filter function on the model_detected_points and return only the values greater than the chosen percentile
    percentile = confidence_filter(model_detected_points, conf_percentile)
    model_detected_points = model_detected_points.loc[model_detected_points.confidence >= percentile]

    # Establish CRS as IRENET95
    ground_truth_points = ground_truth_points.to_crs("epsg:29902")
    model_detected_points = model_detected_points.to_crs("epsg:29902")

    # Create a Unique ID for the Ground Truth Points
    ground_truth_points['gt_ID'] = range(0, len(ground_truth_points))

    # Create a Unique ID for the Model Detected Points
    model_detected_points['det_ID'] = range(0, len(model_detected_points))

    # Drop any unrelated columns
    try:
        ground_truth_points_columns = list(ground_truth_points.columns)
        column_name = [x for x in ground_truth_points_columns if 'lass' in x]
        column_name = column_name[0]
        ground_truth_points = ground_truth_points.rename({column_name: 'class'}, axis=1)
        ground_truth_points = ground_truth_points[['class', 'gt_ID', 'geometry']]
    except Exception:
        pass

    try:
        model_detected_points_columns = list(model_detected_points.columns)
        column_name = [x for x in model_detected_points_columns if 'lass' in x]
        column_name = column_name[0]
        model_detected_points = model_detected_points.rename({column_name: 'det_class'}, axis=1)
        model_detected_points = model_detected_points[['det_class', 'det_ID', 'geometry']]
    except Exception:
        print("Invalid column name for the Class - convert to lower case 'det_class'")

    return ground_truth_points, model_detected_points


def calculate_nearest(row, second_gdf, point_column='geometry', val_col='geometry'):

    """Calculates the distance between two points from two different GeoDataFrames.

        This function finds the nearest point to another specified point, in this case our
        ground_truth_points and model_detected_points. The 'point_column' and 'val_col' refer to the
        two separate GeoDataFrames the function will operate on. The 'val_col' does not always
        have to be a geometry, for instance it can also be used with 'ID' as a unique identifier.

        Parameters
        ----------
        row : iterable
              This is an iterable of the rows in the DataFrame that will be iterated over.

        second_gdf : GeoDataFrame
                     This is the second GeoDataFrame that the function will look to in order to find its nearest point.

        point_column : str
                       This is a string value representing the column on which to operate in the original DataFrame.
                       Note, it must be a geometry column in order to perform a geometric operation.

        val_col : str
                  This is a string value representing the column from the opposing DataFrame upon which to operate on.
                  This can be set to any column, as the origin DataFrame will take whatever value in the column specified
                  to be the 'nearest'.

        Returns
        -------
        GeoSeries / Series
                          Returns the nearest values to the origin DataFrame in the form of a GeoSeries or Series,
                          depending on what kind of val_col was specified.
    """

    # 1 - Create a unary union
    other_points = second_gdf['geometry'].unary_union

    # 2 - Find closest point
    nearest_geoms = nearest_points(row[point_column], other_points)

    # 3 - Find the corresponding geom
    nearest_data = second_gdf.loc[second_gdf['geometry'] == nearest_geoms[1]]

    # 4 - Get the corresponding value (ID)
    nearest_value = nearest_data[val_col].to_numpy()[0]

    return nearest_value


def call_calculate_nearest(ground_truth_points, model_detected_points):

    """Calls the calculate_nearest function on the ground_truth_points and model_detected_points.

       This function calls the calculate_nearest function to create a new GeoDataFrame with the following
       information for each ground_truth_point:
            a) nearest point (geometry)
            b) nearest ID (int)
            c) nearest class (str)

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
                   Returns a new GeoDataFrame for the ground_truth_points (nearest_neighbour)
    """

    try:
        # Nearest Geometry
        ground_truth_points['nearest_point'] = ground_truth_points.apply(
            calculate_nearest, second_gdf=model_detected_points, point_column='geometry', val_col='geometry', axis=1)

        # Nearest ID
        ground_truth_points['n_ID'] = ground_truth_points.apply(
            calculate_nearest, second_gdf=model_detected_points, point_column='geometry', val_col='det_ID', axis=1)

        # Nearest Detected Class
        ground_truth_points['n_det_class'] = ground_truth_points.apply(
            calculate_nearest, second_gdf=model_detected_points, point_column='geometry', val_col='det_class', axis=1)

    except IndexError:
        print("index error: index 0 is out of bounds for axis 0 with size 0. Most likely a result of 'MULTIPOINT' geometry entered, instead of 'POINT' geometry.")
        raise

    return ground_truth_points


def calculate_distance(nearest_neighbour):

    """Calculates the distance between two points within the same GeoDataFrame.

       This function calculates the distance between a ground_truth point and its nearest model detected point.
       It must separate the GeoDataFrame into two GeoDataFrames, since the two geometries required to calculate the distance
       are originally contained in the same GeoDataFrame. Once the distance has been calculated, it is reassembled into one GeoDataFrame again.

       Parameters
       ----------
       nearest_neighbour : GeoDataFrame
                           This GeoDataFrame must have two geometry columns after being returned from the call_calculate_nearest function.
                           The second geometry represents the nearest point to the ground_truth_point.

       Returns
       -------
       GeoDataFrame
                   Returns the nearest_neighbour GeoDataFrame with the distance between the two geometry columns (row-wise) calculated.
    """

    # 1). Separate into 2 GeoDataFrames - Keep the 'n_ID' in both, will need this to reassemble them.
    #     *Note - 'nearest_point' column is a geometry column.
    nearest_point = nearest_neighbour[['nearest_point', 'n_ID', 'n_det_class']]
    nearest_neighbour = nearest_neighbour.drop(['nearest_point', 'n_det_class'], axis=1)

    # 2). Clean up the New DataFrame and calculate the Distance between each point in both DataFrames
    np_geometry = nearest_point['nearest_point']  # extract the geometry as its own variable
    nearest_point = gpd.GeoDataFrame(nearest_point, geometry=np_geometry, crs=IRENET95)  # create a GeoDataFrame
    nearest_point['distance'] = nearest_neighbour.distance(nearest_point)  # calculate distance with ground_truth

    nearest_point = nearest_point.drop('nearest_point', axis=1)  # Drop the 'nearest_point' column - this is a duplicate
    nearest_point = nearest_point.rename(columns={'geometry': 'nearest_point'})  # change the 'geometry' name (don't want it overlapping).

    # 3). Merge the Two DataFrames back together again based on their shared 'n_ID'
    nearest_neighbour = nearest_neighbour.merge(nearest_point, how='left', on='n_ID')
    nearest_neighbour = nearest_neighbour.drop_duplicates(subset="gt_ID", keep='first')

    return nearest_neighbour


def calculate_tp(nearest_neighbour):

    """Calculates the True Positives within a nearest_neighbour GeoDataFrame.

        This function examines whether a model_detected_point is a True Positive:
        In order for it to be a True Positive, it must be within a 2m radius of a ground_truth point
        and the 'class' must match the nearest_point 'n_det_class'.

        Parameters
        ----------
        nearest_neighbour : GeoDataFrame
                            This GeoDataFrame must have two geometry columns after being returned from the call_calculate_nearest function.
                            The second geometry represents the nearest point to the ground_truth_point.

        Returns
        -------
        GeoDataFrame
                    Returns the nearest_neighbour GeoDataFrame with the True Positives (row-wise) calculated.

        GeoDataFrame
                    Returns a separate True Positive GeoDataFrame from all the True Positive rows calculated in the nearest_neighbour parameter.
    """

    nearest_neighbour['TP'] = np.where((nearest_neighbour['class'] == nearest_neighbour['n_det_class'])
                                           & (nearest_neighbour['distance'] < 2.0), 1, 0)

    TP = nearest_neighbour.loc[nearest_neighbour['TP'] == 1]
    print("Initial True Positives, Count: {}".format(len(TP)))

    # Returns a DataFrame to be merged with the next 'call_calculate_nearest' object
    nearest_neighbour, contains_tp = tp_first_pass(nearest_neighbour)

    TP = TP[['gt_ID', 'n_ID', 'n_det_class', 'nearest_point', 'distance', 'TP']]
    TP = TP.rename({'n_ID': 'det_ID', 'n_det_class': 'det_class', 'nearest_point': 'geometry'}, axis=1)
    TP_geometry = TP.geometry
    TP = gpd.GeoDataFrame(TP, geometry=TP_geometry, crs=IRENET95)

    return nearest_neighbour, TP, contains_tp


def calculate_fp(nearest_neighbour):

    """Calculates the False Positives within a nearest_neighbour GeoDataFrame.

        This function examines whether a model_detected_point is a False Positive:
        In order for it to be a False Positive, it must be either within a 2m distance and the 'class' must not match
        the nearest_point 'n_det_class', or be outside the 2m distance.

        Parameters
        ----------
        nearest_neighbour : GeoDataFrame
                            This GeoDataFrame must have two geometry columns after being returned from the call_calculate_nearest function.
                            The second geometry represents the nearest point to the ground_truth_point.

        Returns
        -------
        GeoDataFrame
                    Returns the nearest_neighbour GeoDataFrame with the False Positives (row-wise) calculated.

        GeoDataFrame
                    Returns a separate False Positive GeoDataFrame from all the False Positive rows calculated in the nearest_neighbour parameter.
    """

    # Check within the 2m radius to see what classes don't match.
    nearest_neighbour['FP'] = np.where((nearest_neighbour['class'] != nearest_neighbour['n_det_class'])
                                       & (nearest_neighbour['distance'] < 2.0), 1, 0)

    # Extract the False Positives within the 2m radius of the ground_truth_points
    FP1 = nearest_neighbour.loc[nearest_neighbour['FP'] == 1]

    if len(FP1) > 0:
        print("Initial False Positives, Count: {}".format(len(FP1)))

        # Make sure the columns are the same in each DataFrame before concatenating them
        FP1 = FP1[['n_det_class', 'n_ID', 'nearest_point', 'FP']]
        FP1 = FP1.rename({'n_det_class': 'det_class', 'n_ID': 'det_ID', 'nearest_point': 'geometry'}, axis=1)

        try:
            FP1 = FP1.drop_duplicates(keep='first')
        except Exception:
            pass

        return nearest_neighbour, FP1

    elif len(FP1) == 0:
        print("There were no incorrect classifications (FP1) detected on this pass.")

        FP1 = FP1[['n_det_class', 'n_ID', 'nearest_point', 'FP']]
        FP1 = FP1.rename({'n_det_class': 'det_class', 'n_ID': 'det_ID', 'nearest_point': 'geometry'}, axis=1)

        return nearest_neighbour, FP1


def calculate_fn(ground_truth_points, model_detected_points):

    """Calculates the False Negatives within a nearest_neighbour GeoDataFrame.

        This function examines whether a ground_truth_point is a False Negative:
        In order for it to be a False Negative, it must be a ground_truth_point with no model_detected_point
        within a 2m radius of it.

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
                    Returns a separate False Negatives GeoDataFrame.
    """

    gt_buffer = create_buffer(ground_truth_points)

    FN = intersection_count(gt_buffer, model_detected_points)

    FN = FN.loc[FN['n'] == 0]

    return FN


def create_validation_dataframe(VHRValidation_samples, true_positive_points, false_positive_points, false_negative_points):

    """Creates a validation dataframe containing all the metrics needed to assess the performance of the livestock detection model.

        This function takes the true positive, false positive and false negative points and calculates the
        precision, recall and F1 scores for each VHRValidation_V4_sample parcel. A single DataFrame is returned at the end
        with all the scores contained within.

        Parameters
        ----------
        VHRValidation_samples : GeoDataFrame
                                These polygons must be imported.
                                They represent the parcel boundaries of each area of interest containing the samples of livestock.

        true_positive_points : GeoDataFrame
                               These points are the global points for all True Positives calculated.

        false_positive_points : GeoDataFrame
                                These points are the global points for all False Positives calculated.

        false_negative_points : GeoDataFrame
                                These points are the global points for all False Negatives calculated.

        Returns
        -------
        GeoDataFrame
                    Returns a VHRValidation_samples dataframe with the precision, recall and F1 scores calculated for each polygon.
    """

    # TP
    TP_Count = intersection_count(VHRValidation_samples, true_positive_points)
    TP_Count = TP_Count['n']
    VHRValidation_samples['TP_Count'] = TP_Count

    # FP
    FP_Count = intersection_count(VHRValidation_samples, false_positive_points)
    FP_Count = FP_Count['n']
    VHRValidation_samples['FP_Count'] = FP_Count

    # FN
    FN_Count = intersection_count(VHRValidation_samples, false_negative_points)
    FN_Count = FN_Count['n']
    VHRValidation_samples['FN_Count'] = FN_Count

    # Add the Precision, Recall & F1 columns to the DataFrame
    # Precision = TP/(TP+FP)
    VHRValidation_samples['precision'] = (VHRValidation_samples.TP_Count /
                                             (VHRValidation_samples.TP_Count + VHRValidation_samples.FP_Count)).replace(np.inf, 0)

    # Recall = TP/(TP+FN)
    VHRValidation_samples['recall'] = (VHRValidation_samples.TP_Count /
                                          (VHRValidation_samples.TP_Count + VHRValidation_samples.FN_Count)).replace(np.inf, 0)

    # F1 = 2*(Recall * Precision) / (Recall + Precision)
    VHRValidation_samples['f1'] = 2 * ((VHRValidation_samples.recall * VHRValidation_samples.precision) /
                                          (VHRValidation_samples.recall + VHRValidation_samples.precision)).replace(np.inf, 0)

    return VHRValidation_samples


def export_GeoDataFrame(local_filepath, gdf, file_name="", extension='.shp'):

    """Calculates a GeoDataFrame as a shapefile or a csv.

        This function exports a GeoDataFrame to a given file path.
        Its default extension is a shapefile, but can be modified to export to CSV.
        There is also the option ot give the file a name, default is an empty string.

        Parameters
        ----------
        local_filepath : str
                         String url of file path to save shapefile or csv file.

        gdf : GeoDataFrame
              Chosen GeoDataFrame to export.

        file_name : str
                    String of chosen file name. Default is empty string.

        extension : str
                    String of chosen extension type. Options are '.shp' or '.csv' - Default is '.shp'
    """

    if extension == '.shp':
        gdf.to_file(local_filepath + "/" + file_name, driver='ESRI Shapefile')
    elif extension == '.csv':
        gdf.drop('geometry', axis=1).to_csv(local_filepath + "/" + file_name + extension)
    else:
        print("Invalid extension")
        pass

    print("Exported to Local Drive")


