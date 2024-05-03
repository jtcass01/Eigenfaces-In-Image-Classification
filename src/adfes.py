from __future__ import annotations
from enum import Enum
from parse import parse
from typing import Dict, List

from pandas import Series
from numpy import array

FILE_NAME_FORMAT: str = '{gender}{model_id}-{emotion}'
MAX_ID: int = 12

class ADFES_Image:
    HEIGHT: int = 576
    WIDTH: int = 720

    def __init__(self, image_data: List[int], geographic_tag: GEOGRAPHIC_TAG, image_name: str):
        self.image_data: List[int] = image_data
        self.geographic_tag: GEOGRAPHIC_TAG = geographic_tag

        # Get gender, emotion, and model_id from image_name
        parsed_metadata: Dict[str, str] = parse(FILE_NAME_FORMAT, image_name)
        self.gender: GENDER = GENDER(parsed_metadata['gender'])
        self.emotion: EMOTION = EMOTION(parsed_metadata['emotion'])
        self.model_id: int = \
              int(parsed_metadata['model_id'])\
            + self.geographic_tag.encode() * MAX_ID\
            + self.gender.encode() * MAX_ID * len(GEOGRAPHIC_TAG)
        self.id: int = \
              int(parsed_metadata['model_id'])\
            + self.geographic_tag.encode() * MAX_ID\
            + self.gender.encode() * MAX_ID * len(GEOGRAPHIC_TAG) \
            + self.emotion.encode() * MAX_ID * len(EMOTION) * len(GEOGRAPHIC_TAG) \
        
        str(int(parsed_metadata['model_id']))

    def to_series(self) -> Series:
        return Series({
            'geographic_tag': str(self.geographic_tag),
            'gender': str(self.gender),
            'emotion': str(self.emotion),
            'model_id': str(self.model_id),
            'id': str(self.id),
            'image': self.image_data})

class GENDER(Enum):
    MALE: str = 'M'
    FEMALE: str = 'F'

    def __str__(self):
        string_representations: Dict[GENDER, str] = {
            GENDER.MALE: 'male',
            GENDER.FEMALE: 'female'
        }

        if self in string_representations:
            return string_representations[self]
        else:
            raise NotImplementedError
        
    def encode(self) -> int:
        return list(GENDER).index(self)

class EMOTION(Enum):
    ANGER: str = 'Anger'
    CONTEMPT: str = 'Contempt'
    DISGUST: str = 'Disgust'
    EMBARRASSMENT: str = 'Embarrass'
    FEAR: str = 'Fear'
    JOY: str = 'Joy'
    NEUTRAL: str = 'Neutral'
    PRIDE: str = 'Pride'
    SADNESS: str = 'Sad'
    SURPRISE: str = 'Surprise'

    def __str__(self):
        string_representations: Dict[GENDER, str] = {
            EMOTION.ANGER: 'Anger',
            EMOTION.CONTEMPT: 'Contempt',
            EMOTION.DISGUST: 'Disgust',
            EMOTION.EMBARRASSMENT: 'Embarrassment',
            EMOTION.FEAR: 'Fear',
            EMOTION.JOY: 'Joy',
            EMOTION.NEUTRAL: 'Neutral',
            EMOTION.PRIDE: 'Pride',
            EMOTION.SURPRISE: 'Surprise',
            EMOTION.SADNESS: 'Sadness'
        }

        if self in string_representations:
            return string_representations[self]
        else:
            raise NotImplementedError

    def encode(self) -> int:
        return list(EMOTION).index(self)

class GEOGRAPHIC_TAG(Enum):
    NORTH_EUROPEAN: str = 'North-European'
    MEDITERRANEAN: str = 'Mediterranean'

    def __str__(self):
        string_representations: Dict[GENDER, str] = {
            GEOGRAPHIC_TAG.NORTH_EUROPEAN: 'North European',
            GEOGRAPHIC_TAG.MEDITERRANEAN: 'Mediterranean'
        }

        if self in string_representations:
            return string_representations[self]
        else:
            raise NotImplementedError

    def encode(self) -> int:
        return list(GEOGRAPHIC_TAG).index(self)
