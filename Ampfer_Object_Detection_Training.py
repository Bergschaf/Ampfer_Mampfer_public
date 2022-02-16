import numpy as np
import torch
import torch.utils.data
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from engine import train_one_epoch, evaluate
import utils
import transforms as T
import xml.etree.ElementTree as ET
import os
from PIL import ImageDraw, Image

# --------------------------------------------------------------------------------------------------------------------------------------------------

train = True  # Wenn diese Variable auf True gesetzt ist, dann wird das Netz Trainiert, sonst wird nur getestet.

# --------------------------------------------------------------------------------------------------------------------------------------------------


class Dataloader(torch.utils.data.Dataset):
    def __init__(self, transforms=None):
        self.Bilder = "drive/MyDrive/Object_detection_Data/Bilder/"
        self.pdir = sorted(os.listdir(self.Bilder))
        self.Label = "drive/MyDrive/Object_detection_Data/Label/"
        self.Data = {}
        self.Boxes = []
        self.Transforms = transforms

    def __getitem__(self, filename):
        """
        Mit der __getitem__ Funktion kann man mit der schreibweise Dataloader[filename] die Label und das Bild der
        Datei filename abrufen. (Format: [Label, RGB-Bild]
        """

        img = Image.open(self.Bilder + self.pdir[filename]).convert("RGB")  # Das Bild und
        box_list = self.get_Boxes(self.pdir[filename][:len(self.pdir[filename]) - 4])  # die Label werden abgerufen
        boxes = torch.as_tensor(box_list, dtype=torch.float32)
        num_objs = len(box_list)
        labels = torch.ones((num_objs,), dtype=torch.int64)
        image_id = torch.tensor([filename])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.Transforms is not None:
            img, target = self.Transforms(img, target)
        # Die Label und das Bild werden zu einem Tensor konvertiert.
        return img, target

    def __len__(self):
        return int(len(self.pdir))

    def get_Boxes(self, filename):
        """
        Die get_Boxes-Funktion gibt die Label für das Bild filename zurück.
        """
        f = open("drive/MyDrive/Object_detection_Data/Label/" + filename + ".xml")
        tree = ET.parse(f)
        root = tree.getroot()
        # Die xml-Datei mit den Labeln wird geöffnet.
        self.Boxes = []
        for i in range(len(root) - 6):
            self.Boxes.append([int(root[i + 6][4][0].text), int(root[i + 6][4][1].text), int(root[i + 6][4][2].text),
                               int(root[i + 6][4][3].text)])
            # Die Boxen werden in dem Format [[x,y,x,y][x,y,x,y],...] zurückgegeben
        return self.Boxes


def get_model(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    # Das pre-trained Model wird heruntergeladen und angepasst.

    if os.path.isfile("drive/MyDrive/Object_detection_Data/Netze/model_drohne_niedrig_V2.pt"):
        print("---loaded model---")
        model.load_state_dict(torch.load("drive/MyDrive/Object_detection_Data/Netze/model_drohne_niedrig_V2.pt"))
        # Falls bereits ein Netz trainiert wurde, wird dieses Netz geladen.
    return model


def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
        # Während dem Training werden die Bilder zufällig gedreht.
    return T.Compose(transforms)


dataset = Dataloader(transforms=get_transform(train=True))
dataset_test = Dataloader(transforms=get_transform(train=False))

torch.manual_seed(1)
indices = torch.randperm(len(dataset)).tolist()
dataset = torch.utils.data.Subset(dataset, indices[:-5])
dataset_test = torch.utils.data.Subset(dataset_test, indices[-5:])

data_loader = torch.utils.data.DataLoader(
    Dataloader(transforms=get_transform(train=True)), batch_size=2, shuffle=True, num_workers=2,
    collate_fn=utils.collate_fn)
data_loader_test = torch.utils.data.DataLoader(
    Dataloader(transforms=get_transform(train=False)), batch_size=1, shuffle=False, num_workers=2,
    collate_fn=utils.collate_fn)
# Die Trainings und Test Datensets werden definiert.
print("We have: {} examples, {} are training and {} testing".format(len(indices), len(dataset), len(dataset_test)))

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# Wenn möglich wird eine cuda-fähige Grafikkarte verwendet.

num_classes = 2
# Es gibt zwei Klassen, Ampfer und nicht Ampfer.
model = get_model(num_classes)
model.to(device)

params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.0003,
                            momentum=0.9, weight_decay=0.0005)
# Der Optimizer, der daas Netz trainiert, wird erstellt.
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                               step_size=3,
                                               gamma=0.5)
# Die Lernrate wird immer weiter gesenkt.

num_epochs = 20
if train:

    for epoch in range(num_epochs):
        train_one_epoch(model, optimizer, data_loader, device, epoch,
                        print_freq=10)

        lr_scheduler.step()
        # Nach jeder Traings Epoche wird die Lernrate aktualisiert.
        evaluate(model, data_loader_test, device=device)
        # und das Netz wird mit den Testdaten getestet.
        torch.save(model.state_dict(), "drive/MyDrive/Object_detection_Data/Netze/model_drohne_niedrig_V2.pt")
        # Das Netz wird gespeichert.

loaded_model = get_model(num_classes=2)

for idx in range(20):
    # Am Ende wird das Netz 20x getestet.
    img, _ = dataset_test[idx]
    label_boxes = np.array(dataset_test[idx][1]["boxes"])

    loaded_model.eval()
    with torch.no_grad():
        prediction = loaded_model([img])
    # Das Netz wird auf jedes Trainingsbild angewendet.
    image = Image.fromarray(img.mul(255).permute(1, 2, 0).byte().numpy())
    draw = ImageDraw.Draw(image)

    for elem in range(len(label_boxes)):
        draw.rectangle([(label_boxes[elem][0], label_boxes[elem][1]),
                        (label_boxes[elem][2], label_boxes[elem][3])],
                       outline="blue", width=10)
    for element in range(len(prediction[0]["boxes"])):
        boxes = prediction[0]["boxes"][element].cpu().numpy()
        score = np.round(prediction[0]["scores"][element].cpu().numpy(),
                         decimals=4)
        if score > 0.8:
            draw.rectangle([(boxes[0], boxes[1]), (boxes[2], boxes[3])],
                           outline="red", width=10)
            draw.text((boxes[0], boxes[1]), text=str(score))
    # Auf den Bildern werden dann die erkannten Ampfer und die Ampfer aus den Labeln eingezeichnet.
    image.save("drive/MyDrive/Object_detection_Data/Testbilder_Drohne_Ergebnis/Drohne_Niedrig_" + str(idx) + ".JPG")
# Tutorial: https://towardsdatascience.com/building-your-own-object-detector-pytorch-vs-tensorflow-and-how-to-even-get-started-1d314691d4ae
