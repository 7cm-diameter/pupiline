if __name__ == "__main__":
    from os import mkdir
    from os.path import exists

    required_directories = [
        "data",
        "data/analyzed",
        "data/video",
        "data/tracked",
        "data/area",
    ]

    for dir in required_directories:
        if not exists(dir):
            mkdir(dir)
