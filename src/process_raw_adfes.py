"""process_raw_data.py: 

    I reached out to the Unviersity of Amsterdamn for access to the raw data used in the ADFES study.
    This script will process the still pictures data into a format that can be used in the Eigenfaces project.
    I've manually deleted 'Extras and HD Stills' directories and 'Prototype Stills' directories.
    I then moved all images in the 'Apex Stills' directories into directories named after the models' represented 
    geographical locations."""

__author__ = "Jacob Taylor Cassady"
__email__ = "jcassady@jh.edu"

# Built-in Libraries
from os import listdir
from os.path import join, dirname, basename
from typing import List

# External Libraries
from pandas import DataFrame, concat
from cv2 import imread, cvtColor, COLOR_BGR2GRAY

# Local Libraries
from adfes import ADFES_Image, GEOGRAPHIC_TAG

ADFES_DIRECTORY: str = join(dirname(__file__), '..','datasets','adfes')

def create_pandas_dataframe() -> None:
    dataframe: DataFrame = DataFrame()

    for model_geographical_tag in GEOGRAPHIC_TAG:
        geography_directory: str = join(ADFES_DIRECTORY, model_geographical_tag.value)

        # Get all images in the directory
        files_in_directory: List[str] = listdir(geography_directory)

        for file in files_in_directory:
            if file.endswith('.jpg'):
                image_path: str = join(geography_directory, file)
                image = imread(image_path)
                gray_image = cvtColor(image, COLOR_BGR2GRAY)
                processed_image: ADFES_Image = ADFES_Image(image_data=image, 
                                                           geographic_tag=model_geographical_tag, 
                                                           image_name=basename(image_path).replace('-Apex', '').replace('.jpg', ''))
                dataframe = concat([dataframe, DataFrame(processed_image.to_series()).T], ignore_index=True)

    return dataframe

if __name__ == "__main__":
    adfes_dataframe: DataFrame = create_pandas_dataframe()
    adfes_dataframe.to_pickle(join(ADFES_DIRECTORY, 'adfes_dataframe.pkl'))
    print(adfes_dataframe.head())
