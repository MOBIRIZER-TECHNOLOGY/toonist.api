import cv2
from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

def process():

    style = "artstyle"

    img_cartoon = pipeline(Tasks.image_portrait_stylization,model='damo/cv_unet_person-image-cartoon-3d_compound-models')
    
    if style == "anime":
        img_cartoon = pipeline(Tasks.image_portrait_stylization,model='damo/cv_unet_person-image-cartoon_compound-models')
    elif style == "3d":
        img_cartoon = pipeline(Tasks.image_portrait_stylization,model='damo/cv_unet_person-image-cartoon-3d_compound-models')
    elif style == "handdrawn":
        img_cartoon = pipeline(Tasks.image_portrait_stylization,model='damo/cv_unet_person-image-cartoon-handdrawn_compound-models')
    elif style == "sketch":
        img_cartoon = pipeline(Tasks.image_portrait_stylization,model='damo/cv_unet_person-image-cartoon-sketch_compound-models')
    elif style == "artstyle":
        img_cartoon = pipeline(Tasks.image_portrait_stylization,model='damo/cv_unet_person-image-cartoon-artstyle_compound-models')

    #Local Path
    img_path = 'input.png'
    # Server Path
    # img_path = 'https://invi-label.oss-cn-shanghai.aliyuncs.com/label/cartoon/image_cartoon.png'
    result = img_cartoon(img_path)

    cv2.imwrite(style + '_result.png', result[OutputKeys.OUTPUT_IMG])
    print('finished!')


if __name__ == '__main__':
    process()
