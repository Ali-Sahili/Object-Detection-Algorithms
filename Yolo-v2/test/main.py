
import argparse




def get_args():
    parser = argparse.ArgumentParser("YOLO-v2 implementation on Pytorch.")
    parser.add_argument("--image_size", type=int, default=448, help="common width and height.")
    parser.add_argument("--conf_threshold", type=float, default=0.35)
    parser.add_argument("--nms_threshold", type=float, default=0.5)
    parser.add_argument("--pre_trained_model_type", type=str, choices=["model", "params"], 
                                                              default="model")
    parser.add_argument("--pre_trained_model_path", type=str, 
                         default="trained_models/whole_model_trained_yolo_voc")
    parser.add_argument("--input", type=str, default="test_images")
    parser.add_argument("--output", type=str, default="test_images")

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_args()
    test(args)
