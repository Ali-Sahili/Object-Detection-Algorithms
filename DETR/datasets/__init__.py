from datasets.coco import build as build_coco


def build_dataset(image_set, args):
    return build_coco(image_set, args)
