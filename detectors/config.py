# import the necessary packages
import os
# define the base path to the input dataset and then use it to derive
# the path to the images directory and annotation CSV file
BASE_PATH = "dataset"
IMAGES_PATH = "/content/slikovnaBiometrija/data/ears/train"
ANNOTS_PATH = "/content/slikovnaBiometrija/data/ears/annotations/detection/train.txt"

# define the path to the base output directory
BASE_OUTPUT = "outputKeras"
# define the path to the output serialized model, model training plot,
# and testing image filenames
MODEL_PATH = os.path.sep.join([BASE_OUTPUT, "detector.h5"])
PLOT_PATH = os.path.sep.join([BASE_OUTPUT, "plot.png"])
TEST_FILENAMES = '/content/slikovnaBiometrija/detectors/your_super_detector/outputKeras/test_images.txt'

# initialize our initial learning rate, number of epochs to train
# for, and the batch size
INIT_LR = 1e-4
NUM_EPOCHS = 5
BATCH_SIZE = 32