from groundingdino.util.inference import load_model, load_image, predict, annotate
import cv2
from matplotlib import pyplot as plt

model = load_model("/home/chinchinati/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py", "/home/chinchinati/GroundingDINO/weights/groundingdino_swint_ogc.pth")
IMAGE_PATH = "/home/chinchinati/GroundingDINO/weights/dog-3.jpeg"
TEXT_PROMPT = "chair . person . dog ."
BOX_TRESHOLD = 0.35
TEXT_TRESHOLD = 0.25

image_source, image = load_image(IMAGE_PATH)

boxes, logits, phrases = predict(
    model=model,
    image=image,
    caption=TEXT_PROMPT,
    box_threshold=BOX_TRESHOLD,
    text_threshold=TEXT_TRESHOLD
)

annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)
# cv2.imwrite("annotated_image.jpg", annotated_frame)
print(annotated_frame.shape) 
color = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
plt.imshow(color)
plt.title('Image')
plt.show()