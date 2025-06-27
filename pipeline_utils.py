"""
Functions for running the snow classification workflow on the Google Earth Engine (GEE) server side.
Rainey Aberle
2025
"""

import ee
import geedim as gd
import datetime
import numpy as np

# Grab current datetime for default file name if none is provided
current_datetime = datetime.datetime.now()
current_datetime_str = str(current_datetime).replace(' ','').replace(':','').replace('-','').replace('.','')

def query_gee_for_dem(aoi):
    """
    Query GEE for digital elevation model (DEM) over study site. If the study site is within the ArcticDEM coverage,
    use the ArcticDEM V3 2m mosaic. Otherwise, use NASADEM.

    Parameters
    ----------
    aoi: ee.Geometry
        Area of interest (AOI) to query for DEM.
    
    Returns
    ----------
    dem: ee.Image
        Digital elevation model (DEM) image.
    """
    
    print('\nQuerying GEE for DEM')

    # Determine whether to use ArcticDEM or NASADEM
    # Check for ArcticDEM coverage
    arcticdem_coverage = ee.FeatureCollection('projects/ee-raineyaberle/assets/glacier-snow-cover-mapping/ArcticDEM_Mosaic_coverage')
    intersects = arcticdem_coverage.geometry().intersects(aoi).getInfo()
    if intersects:        
        # make sure there's data (some areas are have empty or patchy coveraage even though they're within the ArcticDEM coverage geometry)
        dem = ee.Image("UMN/PGC/ArcticDEM/V4/2m_mosaic").clip(aoi)
        dem_area = dem.mask().multiply(ee.Image.pixelArea()).reduceRegion(
            reducer=ee.Reducer.sum(),
            geometry=aoi,
            scale=10,
            maxPixels=1e9,
            bestEffort=True
            ).get('elevation')
        dem_percent_coverage = ee.Number(dem_area).divide(ee.Number(aoi.area())).multiply(100).getInfo()
        print(f"ArcticDEM coverage = {int(dem_percent_coverage)} %")
        if dem_percent_coverage >= 90:
            dem_name = "ArcticDEM Mosaic"
            dem_string = "UMN/PGC/ArcticDEM/V4/2m_mosaic"
        else:
            dem_name = "NASADEM"
            dem_string = "NASA/NASADEM_HGT/001"
        
    # Otherwise, use NASADEM
    else:
        dem_name = "NASADEM"
        dem_string = "NASA/NASADEM_HGT/001"
        print('No ArcticDEM coverage')
    print(f"Using {dem_name}")

    # Get the DEM, clip to AOI
    dem = ee.Image(dem_string).select('elevation').clip(aoi)

    # Reproject to the EGM96 geoid if using ArcticDEM
    if dem_name=='ArcticDEM Mosaic':
        geoid = ee.Image('projects/ee-raineyaberle/assets/glacier-snow-cover-mapping/us_nga_egm96_15')
        dem = dem.subtract(geoid)
    dem = dem.set({'vertical_datum': 'EGM96 geoid'})

    return dem


def split_date_range(aoi_area, dataset, date_start, date_end, month_start, month_end):
    """
    Split a date range into smaller chunks to mitigate computation time-out based on the area of the AOI:
        - AOI < 500 km2: split by month
        - 700 km2 <= AOI < 1100 km2: split by week
        - AOI >= 1100 km2: split by day

    Enforces dataset availability windows:
        - Sentinel-2_TOA: available from 2016
        - Sentinel-2_SR: available from 2019
        - Landsat 8/9: available from 2013

    Parameters
    ----------
    aoi_area : float or int
        AOI area in square meters (e.g., from aoi.area().getInfo()).
    dataset : str
        Image dataset name. Supported: "Sentinel-2_TOA", "Sentinel-2_SR", or "Landsat".
    date_start : str
        Start date in 'YYYY-MM-DD' format.
    date_end : str
        End date in 'YYYY-MM-DD' format.
    month_start : int
        Start month (1–12).
    month_end : int
        End month (1–12).

    Returns
    -------
    ranges : list of tuples
        List of (start_date, end_date) strings for each valid time window.
    """
    # Convert string inputs to datetime objects
    date_start = datetime.datetime.strptime(date_start, "%Y-%m-%d").date()
    date_end = datetime.datetime.strptime(date_end, "%Y-%m-%d").date()

    # Enforce dataset availability
    dataset_start_years = {
        "Sentinel-2_TOA": 2016,
        "Sentinel-2_SR": 2019,
        "Landsat": 2013,
    }

    if dataset not in dataset_start_years:
        raise ValueError(f"Unsupported dataset: {dataset}")

    min_year = dataset_start_years[dataset]
    date_start = max(date_start, datetime.date(min_year, 1, 1))  # Clamp to dataset availability

    # List to hold date range tuples
    ranges = []

    # Determine splitting strategy
    if aoi_area < 500e6:
        print('AOI area < 500 km2 — splitting date range by month.')
        for year in range(date_start.year, date_end.year + 1):
            for month in range(1, 13):
                if (year == date_start.year and month < month_start) or (year == date_end.year and month > month_end):
                    continue
                start = datetime.date(year, month, 1)
                end = datetime.date(year + 1, 1, 1) - datetime.timedelta(days=1) if month == 12 else datetime.date(year, month + 1, 1) - datetime.timedelta(days=1)
                start = max(start, date_start)
                end = min(end, date_end)
                if start <= end:
                    ranges.append((start.isoformat(), end.isoformat()))

    elif aoi_area < 1100e6:
        print('500 km2 <= AOI < 1100 km2 — splitting date range by week.')
        current = max(date_start, datetime.date(date_start.year, month_start, 1))
        end_limit = min(date_end, datetime.date(date_end.year, month_end, 28) + datetime.timedelta(days=4)) # max end-of-month buffer

        while current <= date_end:
            if month_start <= current.month <= month_end:
                week_end = min(current + datetime.timedelta(days=6), date_end)
                if week_end.month >= month_start and week_end.month <= month_end and current <= week_end:
                    ranges.append((current.isoformat(), week_end.isoformat()))
            current += datetime.timedelta(days=7)

    else:
        print('AOI >= 1100 km2 — splitting date range by day.')
        current = max(date_start, datetime.date(date_start.year, month_start, 1))
        while current < date_end:
            if month_start <= current.month <= month_end:
                ranges.append((current.isoformat(), (current + datetime.timedelta(days=1)).isoformat()))
            current += datetime.timedelta(days=1)

    print(f"Number of date ranges = {len(ranges)}")
    return ranges



def query_gee_for_imagery(dataset, aoi, date_start, date_end, month_start, month_end, fill_portion, mask_clouds):
    """
    Query GEE for imagery over study site. The function will return a collection of pre-processed, clipped images 
    that meet the search criteria. Images captured on the same day will be mosaicked together to increase spatial coverage.

    Parameters
    ----------
    dataset: str
        Image dataset name. Supported values: "Sentinel-2_TOA", "Sentinel-2_SR", or "Landsat".
    aoi: ee.Geometry
        Area of interest (AOI) to query for imagery.
    date_start: str
        Start date for the image search in the format 'YYYY-MM-DD'.
    date_end: str
        End date for the image search in the format 'YYYY-MM-DD'.
    month_start: int
        Start month for the image search (1-12).
    month_end: int
        End month for the image search (1-12).
    fill_portion: float
        Minimum percent coverage of the AOI required for an image to be included in the collection (0-100).
    mask_clouds: bool
        Whether to mask clouds in the imagery. If True, clouds will be masked using the dataset's cloud mask. 
        If False, no cloud masking will be applied.
    
    Returns
    ----------
    im_mosaics: ee.ImageCollection
        Image collection of pre-processed, clipped images that meet the search criteria. 
    """ 
    # Define image collection
    print(f'Querying GEE for {dataset} image collection')
    if dataset=='Landsat':
        im_col_l8 = gd.MaskedCollection.from_name('LANDSAT/LC08/C02/T1_L2').search(date_start, date_end, 
                                                                                    region=aoi, 
                                                                                    mask=mask_clouds).ee_collection
        im_col_l9 = gd.MaskedCollection.from_name('LANDSAT/LC09/C02/T1_L2').search(date_start, date_end, 
                                                                                    region=aoi, 
                                                                                    mask=mask_clouds).ee_collection
        im_col = im_col_l8.merge(im_col_l9)
        image_scaler = 1/2.75e-05
        refl_bands = ['SR_B1', 'SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7']
        rgb_bands = ['SR_B4', 'SR_B3', 'SR_B2']
        ndsi_bands = ['SR_B3', 'SR_B6']
        scale = 30
    elif 'Sentinel-2' in dataset:
        if dataset=='Sentinel-2_SR':
            im_col = gd.MaskedCollection.from_name('COPERNICUS/S2_SR_HARMONIZED').search(date_start, date_end, 
                                                                                        region=aoi, 
                                                                                        mask=mask_clouds).ee_collection
        elif dataset=='Sentinel-2_TOA':
            im_col = gd.MaskedCollection.from_name('COPERNICUS/S2_HARMONIZED').search(date_start, date_end, 
                                                                                        region=aoi, 
                                                                                        mask=mask_clouds).ee_collection
        image_scaler = 1e4
        refl_bands = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B11', 'B12']
        rgb_bands = ['B4', 'B3', 'B2']
        ndsi_bands = ['B3', 'B11']
        scale = 10

    # Clip to AOI
    def clip_to_aoi(im):
        return im.clip(aoi)
    im_col = im_col.map(clip_to_aoi)

    # Filter collection by month range
    im_col = im_col.filter(ee.Filter.calendarRange(month_start, month_end, 'month'))

    # Select needed bands
    im_col = im_col.select(refl_bands)
    
    # Divide by image scaler
    def divide_im_by_scaler(im):
        im_scaled = ee.Image(im.divide(image_scaler)).copyProperties(im, im.propertyNames()) 
        return im_scaled
    im_col = im_col.map(divide_im_by_scaler)

    # Calculate NDSI
    def calculate_ndsi(im):
        ndsi = im.normalizedDifference(ndsi_bands).rename('NDSI')
        im_with_ndsi = im.addBands(ndsi).copyProperties(im, im.propertyNames()) 
        return im_with_ndsi
    im_col = im_col.map(calculate_ndsi)

    # Mosaic images captured the same day to increase spatial coverage
    def make_daily_mosaics(collection):
        # modified from: https://gis.stackexchange.com/questions/280156/mosaicking-image-collection-by-date-day-in-google-earth-engine
        # Get the list of unique dates in the collection
        date_list = (collection.aggregate_array('system:time_start')
                    .map(lambda ts: ee.Date(ts).format('YYYY-MM-dd')))
        date_list = ee.List(date_list.distinct())

        def day_mosaics(date, new_list):
            date = ee.Date.parse('YYYY-MM-dd', date)
            new_list = ee.List(new_list)
            filtered = collection.filterDate(date, date.advance(1, 'day'))
            image = ee.Image(filtered.mosaic()).set('system:time_start', date.millis())
            return ee.List(ee.Algorithms.If(filtered.size(), new_list.add(image), new_list))

        return ee.ImageCollection(ee.List(date_list.iterate(day_mosaics, ee.List([]))))
    # call the function
    im_mosaics = make_daily_mosaics(im_col)

    # Filter by percent coverage of the AOI
    def calculate_percent_aoi_coverage(im):
        # Total pixels: use a constant image masked to the AOI
        total = ee.Image.constant(1).rename('total').clip(aoi)
    
        # Valid pixels: 1 where the band has data (mask is present), 0 where masked
        valid = im.select(rgb_bands[0]).mask().rename('valid').clip(aoi)
    
        # Stack both bands into one image and reduce in a single pass
        stacked = total.addBands(valid)
    
        counts = stacked.reduceRegion(
            reducer=ee.Reducer.sum(),
            geometry=aoi,
            scale=scale,
            maxPixels=1e9,
            bestEffort=True
        )
    
        total_count = ee.Number(counts.get('total'))
        valid_count = ee.Number(counts.get('valid'))
    
        percent_coverage = valid_count.divide(total_count).multiply(100)
    
        return im.copyProperties(im, im.propertyNames()).set({'percent_AOI_coverage': percent_coverage})
    
    im_mosaics = im_mosaics.map(calculate_percent_aoi_coverage)
    im_mosaics = im_mosaics.filter(ee.Filter.gte('percent_AOI_coverage', fill_portion))

    return ee.ImageCollection(im_mosaics)


def classify_image_collection(collection, dataset):
    """
    Classify the image collection using a pre-trained classifier. The classifier is trained on a set of training data
    that is specific to the dataset. 

    Parameters
    ----------
    collection: ee.ImageCollection
        Image collection to classify.
    dataset: str
        Image dataset name. Supported values: "Sentinel-2_TOA", "Sentinel-2_SR", or "Landsat".
    
    Returns
    ----------
    classified_collection: ee.ImageCollection
        Classified image collection.
    """

    print('Classifying image collection')

    # Retrain classifier
    if dataset=='Landsat':
        clf = ee.Classifier.smileKNN(3)
        training_data = ee.FeatureCollection("projects/ee-raineyaberle/assets/glacier-snow-cover-mapping/Landsat_training_data")
        feature_cols = ['SR_B1', 'SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7', 'NDSI']
    elif dataset=='Sentinel-2_SR':
        clf = ee.Classifier.libsvm()
        training_data = ee.FeatureCollection("projects/ee-raineyaberle/assets/glacier-snow-cover-mapping/Sentinel-2_SR_training_data")
        feature_cols = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B11', 'B12', 'NDSI']
    elif dataset=='Sentinel-2_TOA':
        clf = ee.Classifier.libsvm()
        training_data = ee.FeatureCollection("projects/ee-raineyaberle/assets/glacier-snow-cover-mapping/Sentinel-2_TOA_training_data")
        feature_cols = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B11', 'B12', 'NDSI']
    clf = clf.train(training_data, 'class', feature_cols)

    # Classify collection
    def classify(im):
        return im.classify(clf).copyProperties(im, im.propertyNames())
    classified_collection = collection.map(classify)

    return ee.ImageCollection(classified_collection)


def calculate_snow_cover_statistics(image_collection, dem, aoi, dataset,
                                    out_folder='glacier_snow_cover_exports', 
                                    file_name_prefix=f'snow_cover_stats_{current_datetime_str}'):
    """
    Calculate snow cover statistics for each image in the collection. The function will calculate the following
    statistics for each image: snow area, ice area, rock area, water area, glacier area, transient AAR, SLA,
    SLA upper bound, and SLA lower bound. 

    Parameters
    ----------
    image_collection: ee.ImageCollection
        Image collection to calculate statistics for.
    dem: ee.Image
        Digital elevation model (DEM) image to use for SLA calculations.
    aoi: ee.Geometry
        Area of interest (AOI) to calculate statistics for.
    out_folder: str
        Name of Google Drive Folder where statistics will be saved as CSV.
    file_name_prefix: str
        Prefix for output file name.
    
    Returns
    ----------
    task
    """

    print('Calculating snow cover statistics')

    # Determine image scale of dataset
    if dataset=='Landsat':
        scale = 30
    else:
        scale = 10

    def process_image(image):
        # Grab the image date
        date = ee.Date(image.get('system:time_start')).format('YYYY-MM-dd')

        # Create masks for each class
        snow_mask = image.eq(1).Or(image.eq(2))
        ice_mask = image.eq(3)
        rock_mask = image.eq(4)
        water_mask = image.eq(5)

        # Calculate areas of each mask
        def calculate_class_area(mask):
            return mask.multiply(ee.Image.pixelArea()).reduceRegion(
                reducer=ee.Reducer.sum(),
                geometry=aoi,
                scale=scale,
                maxPixels=1e9,
                bestEffort=True
            ).get('classification')
        snow_area = calculate_class_area(snow_mask)
        ice_area = calculate_class_area(ice_mask)
        rock_area = calculate_class_area(rock_mask)
        water_area = calculate_class_area(water_mask)

        # Calculate glacier area (snow + ice area)
        glacier_area = ee.Number(snow_area).add(ee.Number(ice_area))

        # Calculate transient AAR (snow area / glacier area)
        transient_aar = ee.Number(snow_area).divide(ee.Number(glacier_area))

        # Estimate snowline altitude (SLA) using the transient AAR and the DEM
        sla_percentile = (ee.Number(1).subtract(ee.Number(transient_aar)))
        sla = dem.reduceRegion(
            reducer=ee.Reducer.percentile(ee.List([ee.Number(sla_percentile).multiply(100).toInt()])),
            geometry=aoi,
            scale=scale,
            maxPixels=1e9,
            bestEffort=True
            ).get('elevation')

        # Estimate upper and lower bounds for the SLA
        # upper bound: snow-free pixels above the SLA
        snow_free_mask = image.eq(3).Or(image.eq(4)).Or(image.eq(5))
        above_sla_mask = dem.gt(ee.Number(sla))
        upper_mask = snow_free_mask.And(above_sla_mask)
        upper_mask_area = upper_mask.multiply(ee.Image.pixelArea()).reduceRegion(
            reducer=ee.Reducer.sum(),
            geometry=aoi,
            scale=scale,
            maxPixels=1e9,
            bestEffort=True
        ).get('classification')
        sla_upper_percentile = (ee.Number(sla_percentile)
                                .add(ee.Number(upper_mask_area)
                                     .divide(ee.Number(aoi.area()))))
        sla_upper = dem.reduceRegion(
            reducer=ee.Reducer.percentile([ee.Number(sla_upper_percentile).multiply(100).toInt()]),
            geometry=aoi,
            scale=scale,
            maxPixels=1e9,
            bestEffort=True
            ).get('elevation')
        # lower bound: snow-covered pixels below the SLA
        below_sla_mask = dem.lt(ee.Number(sla))
        lower_mask = snow_mask.And(below_sla_mask)
        lower_mask_area = lower_mask.multiply(ee.Image.pixelArea()).reduceRegion(
            reducer=ee.Reducer.sum(),
            geometry=aoi,
            scale=scale,
            maxPixels=1e9,
            bestEffort=True
        ).get('classification')
        sla_lower_percentile = (ee.Number(sla_percentile)
                                .subtract(ee.Number(lower_mask_area)
                                          .divide(ee.Number(aoi.area()))))
        sla_lower = dem.reduceRegion(
            reducer=ee.Reducer.percentile([ee.Number(sla_lower_percentile).multiply(100).toInt()]),
            geometry=aoi,
            scale=scale,
            maxPixels=1e9,
            bestEffort=True
            ).get('elevation')
        
        # Return feature
        feature = ee.Feature(None, {
            'date': date,
            'source': dataset,
            'spatial_scale_m': scale,
            'percent_AOI_coverage': image.get('percent_AOI_coverage'),
            'snow_area_m2': snow_area,
            'ice_area_m2': ice_area,
            'rock_area_m2': rock_area,
            'water_area_m2': water_area,
            'glacier_area_m2': glacier_area,
            'transient_AAR': transient_aar,
            'SLA_m': sla,
            'SLA_upper_bound_m': sla_upper,
            'SLA_lower_bound_m': sla_lower
        })

        return feature
    
    # Calculate statistics for each image in collection
    statistics = ee.FeatureCollection(image_collection.map(process_image))

    # Export to Google Drive folder
    task = ee.batch.Export.table.toDrive(
        collection=statistics, 
        description=file_name_prefix, 
        folder=out_folder, 
        fileNamePrefix=file_name_prefix, 
        fileFormat='CSV', 
        )
    task.start()
    print(f'Exporting snow cover statistics to {out_folder} Google Drive folder with file name: {file_name_prefix}')
    print('To monitor tasks, see your Google Cloud Console or GEE Task Manager: https://code.earthengine.google.com/tasks')

    return task


def run_classification_pipeline(aoi: ee.Geometry.Polygon, aoi_area: float | int, dem: ee.Image, 
                                dataset: str, date_start: str, date_end: str, month_start: int, month_end: int, 
                                min_aoi_coverage: int | float, mask_clouds: bool, out_folder: str, glac_id: str):
    """
    Run the classification pipeline for a given AOI and image dataset. 

    Parameters
    ----------
    aoi: ee.Geometry.Polygon
        Area of interest (AOI) to query for imagery.
    aoi_area: float | int
        Area of the AOI in meters squared. Can be calculated using: aoi.area().getInfo()
    dem: ee.Image
        Digital elevation model over the AOI, used for calculating snow cover statistics. 
    dataset: str
        Image dataset name. Supported values: "Sentinel-2_TOA", "Sentinel-2_SR", or "Landsat".
    date_start: str
        Start date for the image search in the format 'YYYY-MM-DD', inclusive.
    date_end: str
        End date for the image search in the format 'YYYY-MM-DD', inclusive.
    month_start: int
        Start month for the image search (1-12), inclusive.
    month_end: int
        End month for the image search (1-12), inclusive.
    min_aoi_coverage: float
        Minimum percent coverage of the AOI required for an image to be included in the collection (0-100).
    mask_clouds: bool
        Whether to mask clouds in the imagery. If True, clouds will be masked using the dataset's cloud mask. 
        If False, no cloud masking will be applied.
    out_folder: str
        Name of Google Drive folder where results will be exported. 
    glac_id: str
        Glacier ID used in output file names.

    Returns
    ----------
    None
    """
    # Make sure dataset is recognized
    if dataset not in ['Sentinel-2_TOA', 'Sentinel-2_SR', 'Landsat']:
        raise ValueError(
            f"Dataset not recognized: {dataset}. Please select from: 'Sentinel-2_TOA', 'Sentinel-2_SR', or 'Landsat'."
        )

    # Split date range into smaller date ranges as necessary
    date_ranges = split_date_range(aoi_area, dataset, date_start, date_end, month_start, month_end)
    
    # Run the workflow for each day in date range separately
    for date_range in date_ranges:
        print('\n', date_range)
    
        # Query GEE for imagery
        image_collection = query_gee_for_imagery(dataset, aoi, date_range[0], date_range[1], month_start, month_end, 
                                                 min_aoi_coverage, mask_clouds)
    
        # Classify image collection
        classified_collection = classify_image_collection(image_collection, dataset)
    
        # Calculate snow cover statistics, export to Google Drive
        task = calculate_snow_cover_statistics(classified_collection, dem, aoi, dataset, out_folder,
                                               file_name_prefix=f"{glac_id}_{dataset}_snow_cover_stats_{date_range[0]}_{date_range[1]}")


