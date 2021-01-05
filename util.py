import os
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
import json
import pandas as pd
import datetime

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
mtcnn = MTCNN(keep_all=False, device=device)
resnet = InceptionResnetV1(pretrained='vggface2', device=device).eval()


def get_photo():
    names = []
    photos = []
    for _, dirs, files in os.walk('face'):
        names = dirs
        break

    for name in names:
        photo = Image.open(os.path.join("face", name, name + ".jpg"))
        photos.append(photo)

    return photos, names


def get_embedding():
    faces = []
    photos, names = get_photo()
    for photo in photos:
        face, prob = mtcnn(photo, return_prob=True)
        faces.append(face)
    face_stack = torch.stack(faces).to(device)
    embedding_list = resnet(face_stack).detach().cpu()

    return embedding_list, names


def write_embedding():
    embedding_list, names = get_embedding()
    dicts = []
    with open('face/embedding.json', 'w') as f:
        for i in range(len(embedding_list)):
            embedding_dict = {'name': names[i], "embedding": embedding_list[i].numpy().tolist()}
            dicts.append(embedding_dict)
        json_str = json.dumps(dicts)
        f.write(json_str)


def load_embedding():
    json_path = os.path.join('face', 'embedding.json')
    with open(json_path, 'r') as f:
        embedding_list = json.load(f)
    return embedding_list

def save_check(name):
    time = datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d %H:%M:%S')
    check_dict = [{'name':name, 'time':time}]
    check_msg = pd.DataFrame(check_dict).to_csv('check.csv', mode='a')


if __name__ == '__main__':
    write_embedding()