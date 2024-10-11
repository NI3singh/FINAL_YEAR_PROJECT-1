# speech_output.py

import pyttsx3

# Initialize the speech engine
engine = pyttsx3.init()

# Create a set to keep track of spoken objects
spoken_objects = set()

def speak_object_details(objects_with_distances):

    for obj_name, distance in objects_with_distances:
        # Only speak if the object hasn't been spoken before
        if obj_name not in spoken_objects:
            text = f"{obj_name} detected at {distance:.2f} meters"
            engine.say(text)
            engine.runAndWait()
            # Mark the object as spoken
            spoken_objects.add(obj_name)
