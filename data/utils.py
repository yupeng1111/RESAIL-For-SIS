import os
import re


class Root:

    def __init__(self, root):
        self.root = root
        path_list = []
        print('Indexing files from %s' % self.root)
        for root, _, file_names in sorted(os.walk(self.root)):
            for file_name in file_names:
                path_list.append(os.path.join(root, file_name))
        print("Index over!")
        self.path_list = path_list

    def filter_pth(self, re_string):
        def filter_fn(string):
            if re.match(re_string, string):
                return True
            else:
                return False
        return list(filter(filter_fn, self.path_list))


class ADE20KPath:
    def __init__(self, data_part, dataroot):
        root = Root(dataroot)
        self.image_paths = root.filter_pth(f'.*?/images/{data_part}/.*?jpg')
        self.class_label_paths = root.filter_pth(f'.*?/annotations/{data_part}/.*?png')
        self.instance_label_paths = root.filter_pth(f'.*?/annotations_instance/{data_part}/.*?')
        self.image_paths.sort(key=lambda string: string[-12: -4])
        self.class_label_paths.sort(key=lambda string: string[-12: -4])
        self.instance_label_paths.sort(key=lambda string: string[-12: -4])
        self.check_path()

    def check_path(self):
        assert len(self.instance_label_paths) > 0
        for p1, p2, p3 in zip(self.instance_label_paths, self.image_paths, self.class_label_paths):
            assert p1.replace('annotations_instance', '').replace('png', '') == \
                p2.replace('images', '').replace('jpg', ''), \
                '\n' + p1.replace('annotations_instance', '').replace('png', '') + \
                "\n" + p2.replace('images', '').replace('jpg', '')

            assert os.path.basename(p1) == os.path.basename(p3), f'{p1} does not match {p3}'

    def set_path(self, obj):
        setattr(obj, 'image_paths', self.image_paths)
        setattr(obj, 'class_label_paths', self.class_label_paths)
        setattr(obj, 'instance_label_paths', self.instance_label_paths)