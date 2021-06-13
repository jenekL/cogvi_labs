import cv2
import torch
from torchvision import transforms

model = torch.hub.load('pytorch/vision:v0.9.0', 'resnet34', pretrained=True)
model.eval()

preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((300, 300)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

image = cv2.imread("input/dog.jpg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
transformed_image = preprocess(image)
input_tensor = torch.tensor(transformed_image, dtype=torch.float32)
input_batch = input_tensor.unsqueeze(0)

if torch.cuda.is_available():
    input_batch = input_batch.to('cuda')
    model.to('cuda')

with torch.no_grad():
    output = model(input_batch)

# print(output[0])

probabilities = torch.nn.functional.softmax(output[0], dim=0)
# print(probabilities)

with open("config/rs34.names", "r") as f:
    categories = [s.strip() for s in f.readlines()]
top5_prob, top5_catid = torch.topk(probabilities, 5)
for i in range(top5_prob.size(0)):
    print(categories[top5_catid[i]], top5_prob[i].item())

#
# image_result = draw_bboxes(image, top5_prob, categories)
# cv2.imshow('Detections', image_result)
# cv2.waitKey(0)
# # save the image to disk
# save_name = "bla.jpg"
# cv2.imwrite(f"outputs/{save_name}", image_result)
