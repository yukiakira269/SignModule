def __init__(self, config):
    self.model = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov5/best.pt', force_reload=True) 

def predict(self, payload):
        """
        Called once per request. Preprocesses the request payload (if necessary), 
        runs inference, and postprocesses the inference output (if necessary).
        Args:
            payload: The request payload
        Returns:
            Prediction or a batch of predictions.
        """

        # Convert url image to PIL format
        img = url_to_img(payload["url"])

        # Run a model
        results = self.model(img)

        # Draw boxes
        boxes = results.xyxy[0].cpu().numpy()
        box_img = draw_box(img, boxes)

        # Return an image in the base64 format
        return img_to_bytes(box_img)
