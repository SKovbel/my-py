import os
import json


def convert_classes(classes, start=1):
    msg = ''
    for id, name in enumerate(classes, start=start):
        msg = msg + "item {\n"
        msg = msg + "  id: " + str(id) + "\n"
        msg = msg + "  name: '" + name + "'\n}\n\n"
    return msg[:-1]


def main(train_json, save_dir):
    classes = []
    with open(train_json, 'r') as load_f:
        dict = json.load(load_f)
        categories = dict['categories']
        for c in categories:
            classes.append(c['name'])

    txt = convert_classes(classes)

    save_path = os.path.join(save_dir, 'label_map.pbtxt')
    with open(save_path, 'w') as dump_f:
        dump_f.write(txt)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description="labelme annotation to coco data json file.")
    parser.add_argument("train_json",
                        help="Output json file path.",
                        type=str)
    parser.add_argument("save_dir",
                        help="Output json file path.",
                        type=str)
    args = parser.parse_args()

    main(args.train_json, args.save_dir)