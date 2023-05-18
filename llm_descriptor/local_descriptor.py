import torch
class LocalDescriptor:
    def __init__(self) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def dense_pred_to_caption(self, predictions):
        boxes = predictions["instances"].pred_boxes if predictions["instances"].has("pred_boxes") else None
        object_description = predictions["instances"].pred_object_descriptions.data
        new_caption = ""
        for i in range(len(object_description)):
            new_caption += (object_description[i] + ": " + str([int(a) for a in boxes[i].tensor.cpu().detach().numpy()[0]])) + "; "
        return new_caption