# 1) IMPORT THE RELEVANT FILES AND LIBRARIES

import sib_main.sib_sub.sib_validation_main as svm
import sib_main.sib_sub.sib_validation_imports as svi

import copy
import pandas as pd
import geopandas as gpd
import numpy as np
from shapely.geometry import Polygon, Point
from shapely.geometry.multipolygon import MultiPolygon
from shapely.geometry.multipoint import MultiPoint
from shapely.ops import nearest_points
from datetime import datetime

ground_truth = gpd.read_file("")
model_det = gpd.read_file("")
VHRsamples = gpd.read_file("")

export_path = ""

# 2) PREPROCESS THE DATA

# Find out the confidence threshold
svi.confidence_filter(model_det)

# Change 'geometry' column from 'MULTIPOINT' to 'POINT'
ground_truth = svi.multi_to_single(ground_truth)
model_det = svi.multi_to_single(model_det)

# Preprocess the data
ground_truth, model_det = svm.preprocessing(ground_truth, model_det, conf_percentile=10)
print("Number of Ground Truth Points: {}".format(len(ground_truth)))
print("Number of Model Detected Points: {}\n".format(len(model_det)))

# 3) FIND THE MAXIMUM VALUE OF DETECTED POINTS WITHIN A 2M RADIUS OF A GROUND TRUTH POINT.
n_ = svi.n_max(ground_truth, model_det)

# 4) RUN THE BODY OF THE SCRIPT.

startTime = datetime.now()

global_tp = []
global_fp = []

for i in range(0, n_):

    # 1) CALCULATE NEAREST
    nearest_neighbour = svm.call_calculate_nearest(ground_truth, model_det)

    # 2) CALCUALTE DISTANCE
    nearest_neighbour = svm.calculate_distance(nearest_neighbour)

    # 3) CALCULATE TP
    nearest_neighbour, tp, tp_previous_pass = svm.calculate_tp(nearest_neighbour)
    tp_dict = tp.to_dict('list')
    try:
        global_tp.append(tp_dict)
    except AttributeError:
        print("No more True Positives to append")
        pass

    # 4) CALCULATE FP
    nearest_neighbour, fp = svm.calculate_fp(nearest_neighbour)
    fp_dict = fp.to_dict('list')
    try:
        global_fp.append(fp_dict)
    except AttributeError:
        print("No more False Positives to append")
        pass

    # 5) DROP INDICES
    # DROP TP INDICES
    model_det = svi.drop_indices(model_det, tp, outcome_value='true')

    # DROP FP INDICES
    model_det = svi.drop_indices(model_det, fp, outcome_value='false')

    # DROP GROUND TRUTH INDICES
    ground_truth = svi.drop_indices(ground_truth, tp, outcome_value='ground')

# 6) CREATE GEODATAFRAMES FOR TRUE POSITIVES AND FALSE POSITIVES
global_tp_df = svi.create_global_GeoDataFrame(global_tp)
global_fp_df = svi.create_global_GeoDataFrame(global_fp)
print("True Positives GeoDataFrame Constructed\n")

# 7) CONSTRUCT THE COMPLETE FALSE POSITIVES DATAFRAME
fp2 = svi.extract_points_outside_2m(ground_truth, model_det)
global_fp_df = global_fp_df.append(fp2)
print("False Positives GeoDataFrame Constructed\n")

# 7) CALCULATE FN
global_fn_df = svm.calculate_fn(ground_truth, model_det)
print("False Negatives GeoDataFrame Constructed\n")

# 8) CREATE_VALIDATION_DATAFRAME
VHRsamples = svm.create_validation_dataframe(VHRsamples, global_tp_df, global_fp_df, global_fn_df)
print("VHRValidation GeoDataFrame Constructed\n")

print(datetime.now() - startTime)

# 5) EXPORT RESULTS

svm.export_GeoDataFrame(export_path, VHRsamples, file_name='VHRSamples', extension='.shp')
svm.export_GeoDataFrame(export_path, VHRsamples, file_name='VHRSamples', extension='.csv')
svm.export_GeoDataFrame(export_path, global_tp_df, file_name='TP', extension='.shp')
svm.export_GeoDataFrame(export_path, global_fp_df, file_name='FP', extension='.shp')
svm.export_GeoDataFrame(export_path, global_fn_df, file_name='FN', extension='.shp')
svm.export_GeoDataFrame(export_path, ground_truth, file_name='ground_truth_OUTPUT', extension='.shp')
svm.export_GeoDataFrame(export_path, model_det, file_name='model_det_OUTPUT', extension='.shp')