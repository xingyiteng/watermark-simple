from PIL import Image


def main():
    # 加载彩色图片
    image = Image.open('D:\\workspace\\watermark\\watermark-project\\watermark-simple\\message.png')
    # 转换为灰度图像
    gray_image = image.convert('L')
    # 应用阈值转换为二值图像
    threshold = 128
    binary_image = gray_image.point(lambda p: p > threshold and 255)
    # 保存二值图像
    binary_image.save('message_binary.png')
    # 显示图像
    binary_image.show()


if __name__ == '__main__':
    main()
