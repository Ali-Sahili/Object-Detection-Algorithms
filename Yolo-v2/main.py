import argparse


def get_args():
    parser = argparse.ArgumentParser("YOLO-v2 implementation on Pytorch")
    parser.add_argument("--image_size", type=int, default=448, help="common width and height.")
    parser.add_argument("--batch_size", type=int, default=10, help="batch size")
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--decay", type=float, default=0.0005)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--num_epoches", type=int, default=160)
    parser.add_argument("--test_interval", type=int, default=1, 
                                           help="Number of epoches between testing phases")
    parser.add_argument("--object_scale", type=float, default=1.0)
    parser.add_argument("--noobject_scale", type=float, default=0.5)
    parser.add_argument("--class_scale", type=float, default=1.0)
    parser.add_argument("--coord_scale", type=float, default=5.0)
    parser.add_argument("--reduction", type=int, default=32)
    parser.add_argument("--es_min_delta", type=float, default=0.0,
                        help="Early stopping's parameter: ")
    parser.add_argument("--es_patience", type=int, default=0,
                        help="Early stopping's parameter. Set to 0 to disable this technique.")
    parser.add_argument("--train_set", type=str, default="train",
                        help="For both VOC2007 and 2012, you could choose 3 different datasets")
    parser.add_argument("--test_set", type=str, default="val",
                        help="For both VOC2007 and 2012, you could choose 3 different datasets")
    parser.add_argument("--year", type=str, default="2007", 
                                            help="The year of dataset (2007 or 2012)")
    parser.add_argument("--data_path", type=str, default="../datasets/VOC2007_train/VOCdevkit", 
                                                 help="the root folder of dataset")
    parser.add_argument("--pre_trained_model_type", type=str, choices=["model", "params"], 
                                                              default="model")
    parser.add_argument("--pre_trained_model_path", type=str, 
                        default="trained_models/whole_model_trained_yolo_voc")
    parser.add_argument("--log_path", type=str, default="tensorboard/yolo_voc")
    parser.add_argument("--saved_path", type=str, default="trained_models")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    train(args)
