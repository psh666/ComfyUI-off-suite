from .modules.image_loader import CachedLoadImageFromUrl
from .modules.image_tool import OFFCenterCrop, OFFCenterCropSEGS, OFFSEGSToImage, OFFImageResizeFit, OFFWatermark
from .modules.misc import GWNumFormatter


# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "Image Resize Fit":OFFImageResizeFit,
    "Image Crop Fit": OFFCenterCrop,
    "OFF SEGS to Image": OFFSEGSToImage,
    "Crop Center with SEGS" : OFFCenterCropSEGS,
    "Watermarking" : OFFWatermark,
    "GW Number Formatting": GWNumFormatter,
    "Cached Image Load From URL": CachedLoadImageFromUrl
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "Image Resize Fit" : "Image Resize Fit",
    "Image Crop Fit": "Image Crop Fit Node",
    "OFF SEGS to Image": "OFF SEGS to Image",
    "Crop Center with SEGS":"Crop Center with SEGS",
    "Watermarking" : "Watermarking",
    "GW Number Formatting": "GW Number Formatting Node",
    "Cached Image Load From URL": "Cached Image Load From URL"

}
