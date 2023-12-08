
class GWNumFormatter:

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):

        return {
            "required": {
                "input_number": ("INT", {
                    "default": 0,
                    "min": 0,  # Minimum value
                    "max": 100000000,  # Maximum value
                }),
                "width": ("INT", {
                    "default": 3,
                    "min": 0,
                    "max": 10,
                })
            },
        }

    RETURN_TYPES = ("STRING",)
    # RETURN_NAMES = ("image_output_name",)

    FUNCTION = "format"

    # OUTPUT_NODE = False

    CATEGORY = "GW"

    def format(self, input_number, width):
        return (f"%0{width}d" % (input_number),)