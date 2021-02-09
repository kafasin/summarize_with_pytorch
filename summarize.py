import transformers
import torch

# helper function
def summarize_with_pipeline(text, pipeline, max_length=70, min_length=30):
  """
  Summarize the given text
  """
  return pipeline(text, max_length=max_length, min_length=min_length)[0]['summary_text']
  
# Create a summarize pipeline
summarizer = transformers.pipeline('summarization')

example_text = """
Apollo 14 (January 31 – February 9, 1971) was the eighth crewed mission in the United States Apollo 
program, and the third to land on the Moon. Commander Alan Shepard (pictured), Command Module Pilot 
Stuart Roosa, and Lunar Module Pilot Edgar Mitchell overcame a series of malfunctions en route to 
the Moon that, after the failure of Apollo 13, might have resulted in a second consecutive aborted 
mission, and possibly the premature end of the Apollo program. Shepard and Mitchell made their lunar 
landing on February 5 in the Fra Mauro formation, where they undertook two extravehicular activities 
(EVAs or moonwalks). In Apollo 14's most famous incident, Shepard hit two golf balls he had brought 
with him with a makeshift club. Roosa remained in lunar orbit, where he took photographs of the Moon 
and performed experiments. After liftoff from the surface and a successful docking, the mission 
returned to Earth, splashing down safely in the Pacific Ocean.
"""

print(summarize_with_pipeline(text, summarizer))

result = """
 Apollo 14 (January 31 – February 9, 1971) was the eighth crewed mission in the U.S. Apollo program . 
 It was the third mission to land on the Moon, with a series of malfunctions that might have resulted 
 in a second aborted mission . Commander Alan Shepard hit two golf balls he had brought with him with makeshift
"""
