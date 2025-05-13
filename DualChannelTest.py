from advancedImageOps import *
import cv2
import numpy as np
from pycromanager import Bridge
import matplotlib.pyplot as plt
import aotools
import json
from UtilitySPIM2 import matriarch
from tqdm import tqdm
import slmpy
import slmAberrationCorrection
from slmAberrationCorrection import make_now
from slmAberrationCorrection import better_iterate
import logging
import time 
logging.basicConfig(level=logging.DEBUG)

logging.info("initializing")

#Opens a JSON file containing all the configurations needed
with open('config.json', 'r') as f:
    config = json.load(f)
slmShape = config["slm_device"]["resolution"]
maskRadius= config["fourier_properties"]["radius"]
positionCH1 = config["fourier_properties"]["CH1_center"]
positionCH2 = config["fourier_properties"]["CH2_center"]
stretch = config["fourier_properties"]["stretch"]
degree = config["settings"]["zernike_modes"]
epsilon = config["settings"]["iteration_epsilon"]
g_0 = config["settings"]["iteration_gain0"]
laser = config["illumination_device"]["name"]
totalIterations = 30
guideStarSize, integralRadious = slmAberrationCorrection.make_now.calculate_guidestar_params(config["guide_star"]["microbead"], config["guide_star"]["binning"])

#Generates phase masks for the SLM
baseMask = generate_corkscrew_simple(maskRadius,  N=4, center=(0, 0), fact=(5/8))
display = np.zeros(slmShape)
monoMask = overlay_centered(display, baseMask, positionCH1)
dualMask = overlay_centered(monoMask, baseMask, positionCH2)
softMask = antialias_gaussian(dualMask, sigma=1.0, mode='reflect', truncate=4.0)

#Adaptive Optics
logging.info("Connecting to SLM")
slm = slmpy.SLMdisplay(monitor = config["slm_device"]["display"])
bridge = Bridge()
core = bridge.get_core()
slm.updateArray(display.astype('uint8'))
logging.info("Please check guide Star visibility before proceeding")
input("Press Enter to continue...")

slm.updateArray(softMask.astype('uint8'))
logging.info("Showing dual mask")
input("Press Enter to continue...")