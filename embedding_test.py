import os
import cv2
from classic_image_classification import ImageEmbedding


def main():
    path = "/home/fmuenke/datasets/gashole_data/test/outburst_0"
    opt = {
        "aggregator": "bag_of_words",
        "complexity": 32,
        "feature": "gray-kaze",
        "sampling_method": "kaze",
        "sampling_step": 16,
        "sampling_window": 8,
        "image_size": {
            "width": 128,
            "height": 128,
        }
    }
    img_emb = ImageEmbedding(opt)
    img_emb.fit(data_path=path, tag_type="cls")

    img_emb.save("/home/fmuenke/ai_models/emb_test")

    img_emb_2 = ImageEmbedding(opt)
    img_emb_2.load("/home/fmuenke/ai_models/emb_test")
    for img_f in os.listdir(os.path.join(path, "images")):
        img = cv2.imread(os.path.join(path, "images",img_f))
        x = img_emb_2.transform(img)
        print(x)



if __name__ == "__main__":
    main()
