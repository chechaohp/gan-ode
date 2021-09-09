from torchvision.datasets import UCF101


def main():
    video_path = "E:\\UCF101\\videos_classified"
    annotation_path = "E:\\UCF101\\annotations"

    dataset = UCF101(video_path, annotation_path, 10)


if __name__ == "__main__":
    main()