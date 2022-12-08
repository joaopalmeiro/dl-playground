import torch
from lavis.models import load_model_and_preprocess
from PIL import Image

IMAGE_FOLDER: str = "img"
IMAGE_FILE: str = "pie_chart.png"

# https://github.com/salesforce/LAVIS#image-captioning
# https://github.com/salesforce/LAVIS/blob/main/examples/blip_image_captioning.ipynb
if __name__ == "__main__":
    # from lavis.models import model_zoo
    # print(model_zoo)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print(device)

    # https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.Image.mode
    # https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.Image.convert
    raw_image = Image.open(f"{IMAGE_FOLDER}/{IMAGE_FILE}").convert("RGB")
    # print(raw_image.mode)

    model_type = "base_coco"
    # model_type = "large_coco"

    model, vis_processors, _ = load_model_and_preprocess(
        name="blip_caption", model_type=model_type, is_eval=True, device=device
    )

    # "eval" for validation/testing/inference
    image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)

    print(model.generate({"image": image}))
    # Nucleus sampling is non-deterministic
    # print(model.generate({"image": image}, use_nucleus_sampling=True, num_captions=3))
