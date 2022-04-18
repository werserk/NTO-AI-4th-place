import torch
import numpy as np

from ocr.transforms import InferenceTransform
from ocr.tokenizer import Tokenizer
from ocr.config import Config
from ocr.models import CRNN


def predict(images, model, tokenizer, device):
    model.eval()
    images = images.to(device)
    with torch.no_grad():
        output = model(images)
    pred = output.detach().cpu()
    pred = torch.argmax(pred, -1).permute(1, 0).numpy()
    text_preds = tokenizer.decode(pred)
    return text_preds

def predict_raw(images, model, device):
    model.eval()
    images = images.to(device)
    with torch.no_grad():
        output = model(images)
    pred = output.detach().cpu().permute(1, 0, 2)
    return pred


class OcrPredictor:
    def __init__(self, model_path, config_path, device='cuda', return_raw=False):
        config = Config(config_path)
        self.tokenizer = Tokenizer(config.get('alphabet'))
        self.device = torch.device(device)
        self.return_raw = return_raw
        # load model
        self.model = CRNN(number_class_symbols=self.tokenizer.get_num_chars())
        self.model.load_state_dict(torch.load(model_path))
        self.model.to(self.device)

        self.transforms = InferenceTransform(
            height=config.get_image('height'),
            width=config.get_image('width'),
        )

    def __call__(self, images):
        if isinstance(images, (list, tuple)):
            one_image = False
        elif isinstance(images, np.ndarray):
            images = [images]
            one_image = True
        else:
            raise Exception(f"Input must contain np.ndarray, "
                            f"tuple or list, found {type(images)}.")

        images = self.transforms(images)
        if not self.return_raw:
            pred = predict(images, self.model, self.tokenizer, self.device)
        else:
            pred = predict_raw(images, self.model, self.device)

        if one_image:
            return pred[0]
        else:
            return pred
