import torch
import torch.utils.data
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from engine import evaluate
from PIL import Image
import utils
import transforms as T
import os
import math
import haversine
from haversine import inverse_haversine
from exif import Image as EImage
import json
import simplekml


bilderp = "drive/MyDrive/Bilder/"  # Ordner, in den die Bilder hochgeladen wurden
targetp = "drive/MyDrive/Koordinaten/"  # Ordner, in den die ermittelten Koordinaten gespeichert werden


class Dataloader(torch.utils.data.Dataset):
    def __init__(self, transforms=None):
        self.Bilder = bilderp
        self.pdir = sorted(os.listdir(self.Bilder))
        self.Data = {}
        self.Boxes = []
        self.Transforms = transforms

    def __getitem__(self, idx):
        """
        Zum Trainieren des Netzes soll diese Funktion, die Label des Bildes an der Position idx
        zurückgeben. Da hier das Netz nur angewendet und nicht trainiert wird, werden diese Label
        nicht benötigt. Anstelle der Label wird daher der Dateiname des Bildes zurückgegeben, sodass
        man die erkannten Ampfer später den Metadaten des Bildes zuordnen kann.
        """
        img = Image.open(self.Bilder + self.pdir[idx]).convert("RGB")
        filename = {"filename": self.pdir[idx]}
        if self.Transforms is not None:
            img, filename = self.Transforms(img, filename)
        return img, filename

    def __len__(self):
        return int(len(self.pdir))


def get_model(num_classes):
    """
    Diese Funktion ist dafür zuständig, das Netz zu laden.
    """
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)  # Das pretrained-Model von pytorch
    in_features = model.roi_heads.box_predictor.cls_score.in_features  # wird heruntergeladen und konfiguriert.
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    if os.path.isfile("drive/MyDrive/Object_detection_Data/Netze/model_drohne_niedrig_V2.pt"):
        print("---loaded model---")
        model.load_state_dict(torch.load("drive/MyDrive/Object_detection_Data/Netze/model_drohne_niedrig_V2.pt"))
    else:
        print("---model not found---")
    # Wenn bereits ein trainiertes Netz vorhanden ist, dann werden die entsprechenden Gewichtungen geladen:
    return model


class Img:
    """
    Die Img Klasse wird für Bilder erstellt, um die Überlappung der Boxen, die die Ampferpflanzen markieren, zu
    entfernen.
    """
    def __init__(self, filename: str, objects):
        self.filename = filename  # Dateiname des Bildes (ohne Dateiendung)
        self.objects = objects  # Die erkannten Positionen der Ampferpflanzen.

    def remove_overlap(self):
        """
        Diese Funktion überprüft, ob sich die Boxen überlappen.
        """
        for index, object in enumerate(self.objects):
            for index2, object2 in enumerate(self.objects[index:]):
                # Das Programm überorüft für jede Kombination von zwei Boxen, ob sie sihc überlappen.
                overlap, smaller = check_overlap(object, object2)  # Die check_overlap-Funktion gibt zurück, zu wie groß
                # die sich überlappende Fläche im Verlgeich zur Fläche der kleineren Box ist (in Prozent).
                if overlap > 0.8:  # Wenn sich die Boxen zu mehr als 80 % überlappen, dann wird die kleinere gelöscht.
                    try:
                        if smaller == 1:  # Die smaller Variable gibt an, welche Box kleiner ist.
                            self.objects.remove(object)
                        else:
                            self.objects.remove(object2)
                    except ValueError:  # Wenn die Box schon gelöscht wurde, dann wird nichts gemacht.
                        pass


def check_overlap(pos1, pos2):
    """
    Die check_overlap-Funktion gibt zurück, wie viel sich die zwei Boxen pos1 und pos2 überlappen.
    """
    difx = 0  # Die überlappung in x-Richtung.
    if pos2[0] < pos1[0] < pos2[2]:
        difx = pos2[2] - pos1[0]
        if pos2[0] < pos1[2] < pos2[2]:
            difx = pos2[2] - pos2[0]

    elif pos1[0] < pos2[0] < pos1[2]:
        difx = pos1[2] - pos2[0]
        if pos1[0] < pos2[2] < pos1[2]:
            difx = pos1[2] - pos1[0]

    dify = 0  # Die Überlappung in y-Richtung.
    if pos2[1] < pos1[1] < pos2[3]:
        dify = pos2[3] - pos1[1]
        if pos2[1] < pos1[3] < pos2[3]:
            dify = pos2[3] - pos2[1]

    elif pos1[1] < pos2[1] < pos1[3]:
        dify = pos1[3] - pos2[1]
        if pos1[1] < pos2[3] < pos1[3]:
            dify = pos1[3] - pos1[1]

    overlap = difx * dify  # Die Fläche der Überlappung.

    A_pos1 = (pos1[2] - pos1[0]) * (pos1[3] - pos1[1])  # Die Fläche von Box 1.
    A_pos2 = (pos2[2] - pos2[0]) * (pos2[3] - pos2[1])  # Die Fläche von Box 2.
    # Das Verhältniss der Fläche der Überlappung und der Fläche der kleineren Box wird in Prozent zurückgegeben.
    return (overlap / A_pos1, 1) if A_pos1 < A_pos2 else (overlap / A_pos2, 2)


def decimal_coords(coords, ref):
    """
    Diese Funktion konvertiert die Koorinaten coords (mit der Referenz ref) in Dezimalkoordinaten.
    """
    decimal_degrees = coords[0] + coords[1] / 60 + coords[2] / 3600
    if ref == "S" or ref == "W":
        decimal_degrees = -decimal_degrees
    return decimal_degrees


def get_flight_yaw_altitude_coords(image_path):
    """
    Diese Funktion liest die Höhe, die Koordinaten und den Winkel nach Norden der Drohen aus den Metadaten des Bildes
    an der Position image_path aus.
    """
    with open(image_path, "rb") as img:
        img_text = img.read()
        image = EImage(img_text)
        coords = (decimal_coords(image.gps_latitude,
                                 image.gps_latitude_ref),
                  decimal_coords(image.gps_longitude,
                                 image.gps_longitude_ref))
        # Die Koordinaten können aus den exif-Metadaten der Bilder ausgelesen werden.
        start_pos = img_text.find(b"FlightYawDegree=") + 17  # Der Winkel nach Norden und die Höhe der Drohne können
        # einfach aus den Bytes des Bildes ausgelesen werden.
        for i, b in enumerate(img_text[start_pos:]):
            if b == 34:
                end_pos = i
                break
        yaw = float(img_text[start_pos:end_pos + start_pos])
        start_pos = img_text.find(b"RelativeAltitude=") + 18
        for i, b in enumerate(img_text[start_pos:]):
            if b == 34:
                end_pos = i
                break
        alt = float(img_text[start_pos:end_pos + start_pos])
        return yaw, alt, coords


def get_Angle_from_Camera(x: int, y: int):
    """
    Diese Funktion gibt die Distanz des Bildpunktes x y zum Bildmittelpunkt und den Winkel zur y-Achse zurück.
    """
    new_x = x - 2000
    new_y = y - 1500
    dist = 0
    if new_x != 0 and new_y != 0:
        dist = math.sqrt(pow(new_x, 2) + pow(new_y, 2))
    return dist, math.atan2(y - 1500, x - 2000) + math.pi / 2


def get_Ampfer_Cords(filename, boxes, Img_dir):
    """
    Diese Funktion gibt eine Liste an Koordinaten zurück, an denen die Ampfer sind, deren Position auf dem Bild in der
    boxes Liste gespeichert ist.
    """
    IMG_PATH = f"{Img_dir}{filename}.JPG"
    ampfer_coords = []
    yaw, alt, coords = get_flight_yaw_altitude_coords(IMG_PATH)  # Die Metadaten werden ausgelesen.
    meter_per_pixel = (alt * math.tan(math.radians(39.5))) / 2500  # Ein Umrechnugsfaktor, wie viele Pixel im Bild
    # einem Meter entsprechen, wird berechnet.
    for x, y in boxes:
        dist_pixel, angle_from_middle = get_Angle_from_Camera(x, y)
        abs_angle = math.radians(yaw) + angle_from_middle  # Die Distanz zum Bildmittelpunkt und der Winkel nach
        dist = dist_pixel * meter_per_pixel  # Norden wird für jeden Ampfer berechnet.
        ampfer_coords.append(inverse_haversine(coords, dist, abs_angle, unit=haversine.Unit.METERS))
        # Dann werden die Koordinaten des Ampfers berechnet und an die amper_coords-Liste angehängt.
    return ampfer_coords


if __name__ == "__main__":
    dataset = Dataloader(transforms=T.Compose([T.ToTensor()]))
    torch.manual_seed(1)
    indices = torch.randperm(len(dataset)).tolist()
    dataset = torch.utils.data.Subset(dataset, indices)
    data_loader = torch.utils.data.DataLoader(
        Dataloader(transforms=T.Compose(T.ToTensor())), batch_size=2, shuffle=False, num_workers=2,
        collate_fn=utils.collate_fn)
    loaded_model = get_model(num_classes=2).eval()
    # Zuerst wird das Netz und der Dataloader initialisiert.

    for idx in range(len(os.listdir(bilderp))):

        img, filename = dataset[idx]

        with torch.no_grad():
            prediction = loaded_model([img])  # Das Netz wird dann auf jedes Bild angewendet.
        objects = []
        pic_name = filename["filename"]
        for element in range(len(prediction[0]["boxes"])):
            boxes = prediction[0]["boxes"][element].cpu().numpy()
            pos = (int(boxes[0]), int(boxes[1]), int(boxes[2]), int(boxes[3]))
            objects.append(pos)
        # Dann wird die boxes-Liste anhand der Netzausgabe erstellt.
        img = Img(pic_name[:-4], objects)
        img.remove_overlap()  # Mit der remove_overlap-Funtkion der Image-Klasse werden dann die Überlappungen entfernt.
        boxes = []
        for box in img.objects:
            boxes.append(((box[0] + box[2]) / 2, (box[1] + box[3]) / 2))
        cordslist = get_Ampfer_Cords(pic_name[:-4], boxes, bilderp)  # Anhand der Bildkoordinaten des Ampfers werden
        # dann die realen Koordinaten des Ampfers berechet.
        json.dumps(cordslist)
        with open(targetp + pic_name[:-4] + ".json", "w") as f:
            f.write(json.dumps(cordslist))
        kml = simplekml.Kml()
        for i in cordslist:
            kml.newpoint(name="Ampfer", coords=[(i[1], i[0])])
        kml.save(f"drive/MyDrive/Koordinaten/{pic_name[:-4]}.kml")
        # Diese Koordinaten werden dann in einer .json und in einer .kml Datei gespeichert.
        print(pic_name + " done!")
