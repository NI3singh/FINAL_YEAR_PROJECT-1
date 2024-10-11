import cv2
import numpy as np

class DistanceCalculator:
    def __init__(self, focal_length, known_height):

        self.focal_length = focal_length
        self.known_height = known_height

    def calculate_distance(self, pixel_height):

        # Use the formula to estimate the distance
        distance = (self.focal_length * self.known_height) / pixel_height
        return distance
