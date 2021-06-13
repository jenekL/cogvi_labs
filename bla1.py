import cv2
import torchvision
from PIL import Image
from matplotlib import pyplot as plt
from torchvision import transforms
import time

def get_prediction(img_path, threshold):
    with open("config/rs50.names", "r") as f:
        categories = [s.strip() for s in f.readlines()]

    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    model.eval()

    transform = transforms.Compose([transforms.ToTensor()])
    img = Image.open(img_path)  #comm
    img = transform(img)

    start_time = time.time()
    pred = model([img])
    print("--- %s seconds ---" % (time.time() - start_time))

    pred_class = [categories[i] for i in list(pred[0]['labels'].numpy())]
    pred_boxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(pred[0]['boxes'].detach().numpy())]
    pred_score = list(pred[0]['scores'].detach().numpy())
    pred_t = [pred_score.index(x) for x in pred_score if x > threshold][-1]
    pred_boxes = pred_boxes[:pred_t + 1]
    pred_class = pred_class[:pred_t + 1]
    return pred_boxes, pred_class, pred_score


def object_image_detection(img_path, threshold=0.5, rect_th=3, text_size=1, text_th=2):
    boxes, pred_class, pred_score = get_prediction(img_path, threshold)

    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    for i in range(len(boxes)):
        title = pred_class[i] + ' ' + str(pred_score[i])
        print(title)

        cv2.rectangle(img, (int(boxes[i][0][0]), int(boxes[i][0][1])), (int(boxes[i][1][0]), int(boxes[i][1][1])),
                      (255, 0, 0), rect_th)
        cv2.putText(img, title, (int(boxes[i][0][0]), int(boxes[i][0][1])), cv2.FONT_HERSHEY_SIMPLEX, text_size,
                    (0, 255, 0), thickness=text_th)
        plt.figure(figsize=(20, 30))
        plt.imshow(img)
        plt.xticks([])
        plt.yticks([])
        plt.show()

def object_video_detection(img_path, threshold=0.5, rect_th=3, text_size=1, text_th=2):
    boxes, pred_class, pred_score = get_prediction(img_path, threshold)

    img = cv2.imread(img_path) #comm
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    for i in range(len(boxes)):
        title = pred_class[i] + ' ' + str(pred_score[i])
        cv2.rectangle(img, (int(boxes[i][0][0]), int(boxes[i][0][1])), (int(boxes[i][1][0]), int(boxes[i][1][1])),
                      (255, 0, 0), rect_th)
        cv2.putText(img, title, (int(boxes[i][0][0]), int(boxes[i][0][1])), cv2.FONT_HERSHEY_SIMPLEX, text_size,
                    (0, 255, 0), thickness=text_th)
    return img


object_image_detection('input/image_2.jpg', threshold=0.8)
