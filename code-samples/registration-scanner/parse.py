"""
parse: Parse filtered image using cloud API, returning data as we can understand it.

Author: Nick Gable (nick@sourcemn.org)
"""

from google.cloud import vision
from .filter import get_bounding_boxes
from difflib import SequenceMatcher
from typing import Tuple
import json
import os
import argparse

RESOURCES_DIR = "resources"


def init_client():
    global client
    API_KEY = json.loads(open('secrets.json').read())['GOOGLE_API_KEY']
    GOOGLE_QUOTA_ID = json.loads(open('secrets.json').read())[
        'GOOGLE_QUOTA_ID']
    client = vision.ImageAnnotatorClient(client_options={
        "api_key": API_KEY,
        "quota_project_id": GOOGLE_QUOTA_ID
    })


def init_resources():
    """
    Load in global resources.
    """
    global cities
    global languages
    global let_to_num
    global language_translations
    global help_texts
    global bounding_boxes

    cities = open(os.path.join(RESOURCES_DIR, 'cities.txt')).read().split('\n')
    languages = open(os.path.join(RESOURCES_DIR, 'languages.txt'),
                     encoding='utf-8').read().split('\n')
    let_to_num = json.loads(
        open(os.path.join(RESOURCES_DIR, 'let_to_num.json')).read())
    language_translations = json.loads(open(os.path.join(
        RESOURCES_DIR, 'language_translations.json'), encoding='utf-8').read())
    help_texts = json.loads(
        open(os.path.join(RESOURCES_DIR, 'help_texts.json'), encoding='utf-8').read())
    bounding_boxes = json.loads(open(os.path.join(
        RESOURCES_DIR, 'bounding_boxes.json'), encoding='utf-8').read())


def detect_content(path: str):
    """
    Detect content using Google Cloud Vision, and return the response object. 
    """
    with open(path, 'rb') as file:
        content = file.read()

    image = vision.Image(content=content)
    response = client.document_text_detection(image=image)

    return response


def get_closest_label(bounds: Tuple[int, int, int, int], bounding_id: str = None) -> str:
    """
    Based off of the provided bounding box, return the string label representing the most likely
    field this text is in. TODO: Maybe use ratio based computing for this, and return None if the ratio is below
    a certain amount for bad data?
    """
    def shared_area(box1: Tuple[int, int, int, int], box2: Tuple[int, int, int, int]):
        """
        Compute the intersecting area of the two boxes. 
        """
        xa1 = box1[0]
        ya1 = box1[1]
        xa2 = box1[2]
        ya2 = box1[3]

        xb1 = box2[0]
        yb1 = box2[1]
        xb2 = box2[2]
        yb2 = box2[3]

        return max(
            0, min(xa2, xb2) - max(xa1, xb1)
        ) * max(
            0, min(ya2, yb2) - max(ya1, yb1)
        )

    best_label = ''
    best_area = 0
    bound_set = get_bounding_boxes(bounding_id)['bounds_set']
    for label, (x1, y1, x2, y2) in bound_set.items():
        area = shared_area((x1, y1, x2, y2), bounds)
        if area > best_area:
            best_area = area
            best_label = label

    return best_label


def assign_annotations(response, bounding_id: str = None) -> dict:
    """
    Given Google Vision response object, filter through annotations, assigning them to specific
    fields based off of bounding box information.
    """
    fields = {}
    for annotation in response.text_annotations[1:]:
        top_left = [10**10, 10**10]
        bottom_right = [0, 0]

        # construct rectangle from the polygon
        for vertex in annotation.bounding_poly.vertices:
            if vertex.x < top_left[0]:
                top_left[0] = vertex.x
            if vertex.y < top_left[1]:
                top_left[1] = vertex.y
            if vertex.x > bottom_right[0]:
                bottom_right[0] = vertex.x
            if vertex.y > bottom_right[1]:
                bottom_right[1] = vertex.y

        label = get_closest_label(
            top_left + bottom_right, bounding_id=bounding_id)
        if label not in fields.keys():
            # include x value for left right placement
            fields[label] = [(annotation.description, top_left[0])]
        else:
            fields[label] += [(annotation.description, top_left[0])]

    # sort fields by top_left
    for field in fields.keys():
        fields[field].sort(key=lambda x: x[1])

    # remove y values from fields, replacing with spaces
    for field in fields.keys():
        fields[field] = [desc for (desc, _) in fields[field]]
        fields[field] = ' '.join(fields[field])

    return fields


def match_similar(annotation: str, options: list[str], minimum_sim=0.2):
    """
    Match `annotation` to the closest value in `options`, or return `annotation`
    if the similarity value is less than `minimum_sim`.
    """
    most_similar = annotation
    best_similarity = 0.0

    for option in options:
        # similarity lowercases and removes spaces to try to maximize opportunity to catch
        similarity = SequenceMatcher(None, option.lower().replace(
            ' ', ''), annotation.lower().replace(' ', '')).ratio()
        if similarity > best_similarity:
            most_similar = option
            best_similarity = similarity

    if best_similarity >= minimum_sim:
        return most_similar
    else:
        return annotation


def clean_annotations(annotations: dict) -> dict:
    """
    Post-process the annotations provided, cleaning up output.
    """
    # name, address, and proxy_permission: title case
    for key in {'name', 'address', 'proxy_permission'}:
        if key in annotations:
            annotations[key] = annotations[key].title()

    if 'city' in annotations:
        # if city name contains common Minneapolis alternatives, set to Minneapolis
        if annotations['city'].lower() in ('msp', 'mps', 'mpl', 'mpls', 'mn'):
            annotations['city'] = "Minneapolis"
        # match to closest in our city list (should almost always be minneapolis)
        else:
            annotations['city'] = match_similar(annotations['city'], cities)

    # do similar thing for languages
    if 'language' in annotations:
        annotations['language'] = match_similar(
            annotations['language'], languages, minimum_sim=0.5)
        if annotations['language'] in language_translations['translations']:
            annotations['language'] = language_translations['translations'][annotations['language']]
            
        # remove languages not in permitted set
        # doing this here because we want the language field to be accurate, since it is used for auto-detecting the language
        if annotations['language'] not in language_translations['permitted_languages']:
            del annotations['language']

    # for state: if the city matches one in our city list, or there is an M, or an N, write MN
    if 'city' in annotations and annotations['city'] in cities:
        annotations['state'] = 'MN'

    if 'state' in annotations and annotations['state'] != 'MN':
        if 'M' in annotations['state'] or 'N' in annotations['state']:
            annotations['state'] = 'MN'

    # for number fields, go through and remove any non-numerical characters, convert to a number
    # also, use mapping of common letters to numbers to possibly correct some issues
    # if empty, drop the field
    number_fields = ['children', 'adults',
                     'elderly', 'phone_number', 'zip', 'total']
    digits = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    for field in number_fields:
        if field in annotations:
            new_val = ''
            for character in annotations[field]:
                if character in let_to_num:
                    new_val += str(let_to_num[character])
                if character in digits:
                    new_val += character

            if len(new_val) > 0:
                annotations[field] = int(new_val)
            else:
                del annotations[field]

    # TODO possibly use the google maps api to get better addresses?
    # we can see if the address reading is good enough as is or not, lol

    return annotations


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Parse information from a filtered form image."
    )
    parser.add_argument("in_file", type=str)

    args = parser.parse_args()
    init_client()
    init_resources()

    response = detect_content(args.in_file)

    print(
        clean_annotations(
            assign_annotations(response)
        )
    )