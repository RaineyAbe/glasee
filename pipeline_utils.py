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


def determine_required_image_scale(aoi, dataset):
    """
    Estimate the appropriate spatial resolution (scale) in meters for processing satellite imagery
    in Google Earth Engine (GEE) based on the user memory limit of 10 MB.

    This function calculates the approximate size in MB of a single image over the given area of 
    interest (AOI) at its native scale. If the estimated image size exceeds the GEE user 
    memory limit, it calculates a coarser spatial scale to stay within the limit.

    Adapted from Daniel Wiell's approach:
    https://gis.stackexchange.com/questions/432948/print-ndvi-image-file-size-in-google-earth-engine

    Parameters
    ----------
    aoi : ee.Geometry.Polygon
        Area of interest over which images will be queried and processed.
    dataset : str
        Image dataset name. Supported values: "Sentinel-2_TOA", "Sentinel-2_SR", or "Landsat".

    Returns
    -------
    scale_required : int or float
        Required image scale in meters to keep image size under GEE's 10 MB user memory limit.
    """
    # Select dataset parameters
    if dataset in ['Sentinel-2_TOA', 'Sentinel-2_SR']:
        dataset_str = 'COPERNICUS/S2_HARMONIZED'
        refl_bands = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B11', 'B12']
        rgb_bands = ['B4', 'B3', 'B2']
    elif dataset == 'Landsat':
        dataset_str = 'LANDSAT/LC08/C02/T1_L2'
        refl_bands = ['SR_B1', 'SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7']
        rgb_bands = ['SR_B4', 'SR_B3', 'SR_B2']

    # Grab and clip a sample image
    image = (
        ee.ImageCollection(dataset_str)
        .filterDate('2022-05-01', '2022-05-31')
        .filterBounds(aoi)
        .first()
        .clip(aoi)
        .select(refl_bands)
    )

    # Get native scale of RGB band in meters
    scale = image.select(rgb_bands[0]).projection().nominalScale().getInfo()

    # Estimate the number of bits contained in each band
    def get_bits(band):
        data_type = ee.Dictionary(ee.Dictionary(band).get('data_type'))
        precision = ee.String(data_type.get('precision'))

        def int_bits():
            min_val = ee.Number(data_type.get('min'))
            max_val = ee.Number(data_type.get('max'))

            types = ee.FeatureCollection([
                ee.Feature(None, {'bits': 8, 'min': -2**7,   'max': 2**7}),
                ee.Feature(None, {'bits': 8, 'min': 0,       'max': 2**8}),
                ee.Feature(None, {'bits': 16, 'min': -2**15, 'max': 2**15}),
                ee.Feature(None, {'bits': 16, 'min': 0,      'max': 2**16}),
                ee.Feature(None, {'bits': 32, 'min': -2**31, 'max': 2**31}),
                ee.Feature(None, {'bits': 32, 'min': 0,      'max': 2**32}),
            ])

            match = (
                types
                .filter(ee.Filter.lte('min', min_val))
                .filter(ee.Filter.gt('max', max_val))
                .merge(ee.FeatureCollection([ee.Feature(None, {'bits': 64})]))
                .first()
            )
            return ee.Number(match.get('bits'))

        return ee.Algorithms.If(
            precision.equals('int'),
            int_bits(),
            ee.Algorithms.If(precision.equals('float'), ee.Number(32), ee.Number(64))
        )

    # Describe image and sum total bits across all bands
    image_description = ee.Dictionary(ee.Algorithms.Describe(image))
    bands = ee.List(image_description.get('bands'))
    total_bits = ee.Number(bands.map(get_bits).reduce(ee.Reducer.sum()))

    # Estimate image size in bytes: (bits / 8) * number of pixels
    pixel_count = aoi.area().divide(scale ** 2)
    im_size_bytes = ee.Number(total_bits.divide(8).multiply(pixel_count).ceil()).getInfo()
    im_size_mb = im_size_bytes / (1024**2)
    print(f"Estimated image size over AOI for {dataset}: {np.round(im_size_mb, 2)} MB")

    # Check if size exceeds GEE memory limit
    if im_size_mb > 10:
        # calculate resolution required for im_size_mb (rounded to the nearest 10 m)
        scale_required = int(round(scale * np.sqrt(im_size_mb), -1))     
        print(f"Image size exceeds GEE's 10 MB limit. Using scale = {scale_required} m")
    else:
        scale_required = scale
        print(f"Image size is within GEE limit. Using default scale = {scale_required} m")

    return scale_required


def split_date_range(aoi, dataset, date_start, date_end, month_start, month_end):
    """
    Split a date range into smaller chunks based on the area of the AOI:
        - AOIs < 1000 km²: split by year
        - AOIs ≥ 1000 km²: split by month

    Enforces dataset availability windows:
        - Sentinel-2_TOA: available from 2016
        - Sentinel-2_SR: available from 2019
        - Landsat 8/9: available from 2013

    Parameters
    ----------
    aoi : ee.Geometry
        Area of interest.
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
    # Convert input strings to date objects
    date_start = datetime.datetime.strptime(date_start, "%Y-%m-%d").date()
    date_end = datetime.datetime.strptime(date_end, "%Y-%m-%d").date()

    # Enforce dataset availability start years
    dataset_start_years = {
        "Sentinel-2_TOA": 2016,
        "Sentinel-2_SR": 2019,
        "Landsat": 2013,
    }

    min_year = dataset_start_years[dataset]
    year_start = max(date_start.year, min_year)
    year_end = date_end.year

    # Get AOI area in m2
    aoi_area = aoi.area().getInfo()
    print(f'AOI area = {int(aoi_area / 1e6)} km^2')

    # Initialize list of date ranges
    ranges = []

    if aoi_area < 1000e6:  
        print('AOI area < 1000 km2, splitting date range by year')
        for year in range(year_start, year_end + 1):
            start = datetime.date(year, month_start, 1)
            if month_end == 12:
                end = datetime.date(year + 1, 1, 1) - datetime.timedelta(days=1)
            else:
                end = datetime.date(year, month_end + 1, 1) - datetime.timedelta(days=1)

            start = max(start, date_start)
            end = min(end, date_end)
            if start <= end:
                ranges.append((start.isoformat(), end.isoformat()))

    else:  
        print('AOI area >= 1000 km2, splitting date range by month')
        for year in range(year_start, year_end + 1):
            for month in range(month_start, month_end+1):
                if (year == year_start and month < month_start) or (year == year_end and month > month_end):
                    continue

                start = datetime.date(year, month, 1)
                if month == 12:
                    end = datetime.date(year + 1, 1, 1) - datetime.timedelta(days=1)
                else:
                    end = datetime.date(year, month + 1, 1) - datetime.timedelta(days=1)

                start = max(start, date_start)
                end = min(end, date_end)
                if start <= end:
                    ranges.append((start.isoformat(), end.isoformat()))
    print(f"Number of date ranges = {len(ranges)}")

    return ranges


def query_gee_for_imagery(dataset, aoi, scale, date_start, date_end, month_start, month_end, fill_portion, mask_clouds):
    """
    Query GEE for imagery over study site. The function will return a collection of pre-processed, clipped images 
    that meet the search criteria. Images captured on the same day will be mosaicked together to increase spatial coverage.

    Parameters
    ----------
    dataset: str
        Image dataset name. Supported values: "Sentinel-2_TOA", "Sentinel-2_SR", or "Landsat".
    aoi: ee.Geometry
        Area of interest (AOI) to query for imagery.
    scale: int | float
        Image scale. 
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
    
    # Reproject to user-specified image scale
    crs = im_col.first().select(rgb_bands[0]).projection()
    def resample_scale(im):
        return im.reproject(crs=crs, scale=scale)
    im_col = im_col.map(resample_scale)

    # Filter collection by month range
    im_col = im_col.filter(ee.Filter.calendarRange(month_start, month_end, 'month'))
    
    # Clip to AOI
    def clip_to_aoi(im):
        return im.clip(aoi)
    im_col = im_col.map(clip_to_aoi)

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
        # Count total pixels in the AOI (use one band, first RGB band arbitrarily)
        pixel_count = im.select(rgb_bands[0]).unmask().reduceRegion(
            reducer = ee.Reducer.count(),
            geometry = aoi,
            scale = scale,
            maxPixels = 1e9,
            bestEffort = True
        ).get(rgb_bands[0])

        # Count unmasked pixels in the AOI (use one band, first RGB band arbitrarily)
        unmasked_pixel_count = im.select(rgb_bands[0]).reduceRegion(
            reducer = ee.Reducer.count(),
            geometry = aoi,
            scale = scale,
            maxPixels = 1e9,
            bestEffort = True
        ).get(rgb_bands[0])

        # Calculate percent coverage
        percent_coverage = ee.Number(pixel_count).divide(unmasked_pixel_count).multiply(100)

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


def calculate_snow_cover_statistics(image_collection, dem, aoi, scale=30, 
                                    out_folder='snow_cover_exports', 
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
    scale: int
        Scale to use for calculations (default is 30m).
    out_folder: str
        Name of Google Drive Folder where statistics will be saved as CSV.
    file_name_prefix: str
        Prefix for output file name.
    
    Returns
    ----------
    task
    """

    print('Calculating snow cover statistics')

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
                scale=10,
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
            scale=10,
            maxPixels=1e9,
            bestEffort=True
        ).get('classification')
        sla_upper_percentile = (ee.Number(sla_percentile)
                                .add(ee.Number(upper_mask_area)
                                     .divide(ee.Number(aoi.area()))))
        sla_upper = dem.reduceRegion(
            reducer=ee.Reducer.percentile([ee.Number(sla_upper_percentile).multiply(100).toInt()]),
            geometry=aoi,
            scale=10,
            maxPixels=1e9,
            bestEffort=True
            ).get('elevation')
        # lower bound: snow-covered pixels below the SLA
        below_sla_mask = dem.lt(ee.Number(sla))
        lower_mask = snow_mask.And(below_sla_mask)
        lower_mask_area = lower_mask.multiply(ee.Image.pixelArea()).reduceRegion(
            reducer=ee.Reducer.sum(),
            geometry=aoi,
            scale=10,
            maxPixels=1e9,
            bestEffort=True
        ).get('classification')
        sla_lower_percentile = (ee.Number(sla_percentile)
                                .subtract(ee.Number(lower_mask_area)
                                          .divide(ee.Number(aoi.area()))))
        sla_lower = dem.reduceRegion(
            reducer=ee.Reducer.percentile([ee.Number(sla_lower_percentile).multiply(100).toInt()]),
            geometry=aoi,
            scale=10,
            maxPixels=1e9,
            bestEffort=True
            ).get('elevation')
        
        # Return feature
        feature = ee.Feature(None, {
            'date': date,
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


def run_classification_pipeline(aoi: ee.Geometry.Polygon, dem: ee.Image, 
                                dataset: str, date_start: str, date_end: str, month_start: int, month_end: int, 
                                min_aoi_coverage: int | float, mask_clouds: bool, out_folder: str, glac_id: str):
    """
    Run the classification pipeline for a given AOI and image dataset. 

    Parameters
    ----------
    dataset: str
        Image dataset name. Supported values: "Sentinel-2_TOA", "Sentinel-2_SR", or "Landsat".
    aoi: ee.Geometry.Polygon
        Area of interest (AOI) to query for imagery.
    dem: ee.Image
        Digital elevation model over the AOI, used for calculating snow cover statistics. 
    date_start: str
        Start date for the image search in the format 'YYYY-MM-DD'.
    date_end: str
        End date for the image search in the format 'YYYY-MM-DD'.
    month_start: int
        Start month for the image search (1-12).
    month_end: int
        End month for the image search (1-12).
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
    
    # Calculate the required image spatial resolution (scale) to stay within the GEE user memory limit.
    scale_required = determine_required_image_scale(aoi, dataset)
    
    # Split the date range into separate date ranges
    date_ranges_list = split_date_range(aoi, dataset, date_start, date_end, month_start, month_end)
    
    # Run the workflow for each date range separately. 
    for date_range in date_ranges_list:
        print('\n', date_range)
    
        # Query GEE for imagery
        image_collection = query_gee_for_imagery(dataset, aoi, scale_required, date_range[0], date_range[1], month_start, month_end, 
                                                 min_aoi_coverage, mask_clouds)
    
        # Classify image collection
        classified_collection = classify_image_collection(image_collection, dataset)
    
        # Calculate snow cover statistics, export to Google Drive
        task = calculate_snow_cover_statistics(classified_collection, dem, aoi, scale=scale_required, out_folder=out_folder,
                                               file_name_prefix=f"{glac_id}_{dataset}_snow_cover_stats_{date_range[0]}_{date_range[1]}")


