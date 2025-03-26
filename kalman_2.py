# -----------------------------------
# 功能：数据统计(v1)、文件名的检查
# -----------------------------------
import os


def main(root_path, imgs_path='images', json_path='json'):
    img_path = os.path.join(root_path, imgs_path)
    # print('当前处理文件名：', img_path)

    if img_path.split('/')[-1] != 'images':  # 文件名是否为images
        print('Is not images')

    if os.path.exists(img_path):
        img_list = os.listdir(img_path)
        print('总的数据量为：{}!'.format(len(img_list)))
        return len(img_list)
    else:
        print('images文件名不对！')

    json_path = os.path.join(root_path, json_path)
    if os.path.exists(json_path):
        json_list = os.listdir(json_path)
        print('总的数据量为：{}!'.format(len(json_list)))
    else:
        print('json文件名不匹配！')

    # re1, re2 = int(len(img_list)), int(len(json_list))
    # return (re1, re2)


if __name__ == "__main__":
    data_path = r'D:\infiRay\2、可见光车道线'
    file_list = os.listdir(data_path)

    imgs_sum, jsons_num = 0, 0
    for filename in file_list:
        root_path = os.path.join(data_path, filename)
        result = main(root_path)
        imgs_sum += result[0]
        jsons_num += result[1]

    print('imgs_sum: ', imgs_sum)
    print('jsons_num: ', jsons_num)